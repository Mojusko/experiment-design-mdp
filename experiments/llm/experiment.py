import os
import torch
import numpy as np
import datetime
import sys

import hydra
from omegaconf import DictConfig, OmegaConf # Added OmegaConf
from hydra.utils import to_absolute_path # Import Hydra path utility

from stpy.helpers.helper import cartesian
# Updated imports from doexpy.env.llm
from doexpy.env.llm import (
    LLMGrid, get_scorer_model, make_theta_star, generate_emissions, create_prompt # Added create_prompt
)
# Import embedder components
from components.embedder import BaseEmbedder, create_embedder
from components.feedback import FeedbackFactory
from components.solver import SolverFactory
from components.tester  import BaseTester, ImageGenerationTester
# Import specific saver types needed for validation and estimator creation
from components.saver   import BaseSaver, VisitsSaver, VisitsImageSaver, ConfSaver, LearnedEstimatorSaver
# Import estimator and related components for initialization
from stpy.regression.regularized_dictionary.regularized_multinomial_estimator import RegularizedMultinomialEstimator
from stpy.probability.multinomial_likelihood import MultinomialLikelihood
from stpy.regularization.regularizer import L2Regularizer
from stpy.embeddings.polynomial_embedding import CustomEmbedding # Corrected import path
import json # For loading feedback JSON
import re # For parsing filenames



class LLMExperiment:
    """
    High-level orchestrator. Steps:
      1) Load data & environment
      2) Initialize feedback+design+estimator
      3) Initialize solver (FrankWolfe, DP, random, etc.) -> Explorer
      4) Run exploration
      5) Fit/estimate
      6) Test & save results
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.rng = np.random.RandomState(int(cfg.seed))
        self.seed = int(cfg.seed) # Store seed as an integer attribute

        # Create results directory with timestamp
        timestamp = os.environ.get('TIMESTAMP', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        self.results_dir = f"{cfg.results_dir}-{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)

        # --- Initialize Embedder ---
        # The create_embedder factory reads cfg.embedder config group
        self.embedder: BaseEmbedder = create_embedder(cfg.embedder)
        print(f"Initialized Embedder: {self.embedder.__class__.__name__} with model {self.embedder.model_id}")

        # --- Load Data & Initialize Environment ---
        self.training_words, self.testing_words, self.model_words = self._load_data()
        # _init_env now uses self.embedder
        self.env = self._init_env() # This initializes self._scorer_model and self._theta_star

        # --- Initialize Core Components ---
        # Pass the embedder instance where needed (e.g., FeedbackFactory might need it)
        # Pass the scorer model as before
        self.feedback, self.design, self.estimator = FeedbackFactory.create(cfg, self.env, self._scorer_model, self.embedder)
        # Pass the same_first_action_in_episode flag to the solver factory
        self.explorer = SolverFactory.create(
            cfg,
            self.env,
            self.design,
            self.feedback,
            same_first_action_in_episode=cfg.get('same_first_action_in_episode', False) # Read from config
        )

        # For test-only mode, initialize estimator to None, will be loaded later (remains same)
        self.estimator = None
        
        # Build experiment_id with prefix if available
        experiment_id = str(self.cfg.experiment_id) if self.cfg.experiment_id is not None else ""
        if hasattr(self.cfg.experiment, 'id_prefix') and self.cfg.experiment.id_prefix:
            # Check if prefix is already present
            prefix_present = experiment_id.startswith(self.cfg.experiment.id_prefix) if experiment_id else False
            
            if not prefix_present:
                # Only prepend if not already present
                algorithm_code = self._get_algorithm_code()
                feedback_code = self._get_feedback_code()
                if experiment_id:
                    # If experiment_id is already set, use it as a suffix (typically seed number)
                    experiment_id = f"{self.cfg.experiment.id_prefix}-{algorithm_code}-{feedback_code}-{experiment_id}"
                else:
                    experiment_id = f"{self.cfg.experiment.id_prefix}-{algorithm_code}-{feedback_code}"

        self.experiment_id = experiment_id

        # Initialize testers and savers with results_dir and experiment_id
        # Pass embedder and env to testers/savers that might need it
        testers_config = self.cfg.get('tester') # Get the config value (could be list or None)
        self.testers = []
        if testers_config:
             for t_conf in testers_config:
                 try:
                     # Base arguments needed by all testers (via BaseTester)
                     init_args = {
                         'scorer_model': self._scorer_model,
                         'embedder': self.embedder,
                         # 'params' is handled by Hydra via t_conf
                     }
                     # Add arguments specific to certain testers if needed
                     # Example: If a tester needed 'env', add it here conditionally based on t_conf._target_
                     # if t_conf.get('_target_') == 'components.tester.SomeTesterNeedingEnv':
                     #     init_args['env'] = self.env

                     tester = hydra.utils.instantiate(
                         t_conf, # The tester's specific config
                         **init_args # Pass the dynamically built dictionary
                     )
                     self.testers.append(tester)
                     print(f"Successfully initialized tester: {t_conf._target_}")
                 except Exception as e:
                     print(f"Failed to initialize tester {t_conf._target_}: {e}")
                     raise

        # Initialize the savers with appropriate parameters
        self.savers = []
        for s_conf in self.cfg.savers: # Renamed loop variable for clarity
            try:
                # Instantiate the saver, passing necessary objects and the main config
                # BaseSaver and its children now expect explicit named arguments.
                # Hydra automatically handles arguments defined within s_conf (like 'params').
                # We explicitly pass the arguments NOT defined in the saver's YAML config,
                # only passing arguments relevant to the specific saver type.

                # Base arguments common to most savers
                init_args = {
                    'env': self.env,
                    'embedder': self.embedder,
                    'scorer_model': self._scorer_model,
                    'results_dir': self.results_dir,
                    'experiment_id': self.experiment_id
                    # 'params' and other config-specific args are handled by Hydra via s_conf
                }

                # Add arguments specific to VisitsImageSaver if it's the target
                if s_conf.get('_target_') == 'components.saver.VisitsImageSaver':
                    init_args['horizon'] = self.cfg.horizon
                    # init_args['horizon'] = self.cfg.horizon # Removed duplicate line
                    init_args['dense_feedback'] = self.cfg.get('dense_feedback', False)
                    init_args['verbose'] = self.cfg.get('verbose', False)
                    init_args['seed'] = self.seed # Pass the current seed
                    # Pass total repeats, default to 1 if not found in config
                    init_args['total_repeats'] = self.cfg.experiment.get('repeats', 1)
                # Add arguments specific to ReadableVisitsSaver
                elif s_conf.get('_target_') == 'components.saver.ReadableVisitsSaver':
                    init_args['horizon'] = self.cfg.horizon
                    init_args['dense_feedback'] = self.cfg.get('dense_feedback', False)

                # Instantiate the saver using the configuration and the constructed arguments
                saver = hydra.utils.instantiate(
                    s_conf, # The saver's specific config (contains _target_, params, etc.)
                    **init_args # Pass the dynamically built dictionary of arguments
                )
                self.savers.append(saver)
                print(f"Successfully initialized saver: {s_conf._target_}")
            except Exception as e:
                print(f"Failed to initialize saver {s_conf._target_}: {e}")
                raise  # Re-raise to stop execution as this is a critical component
        self.visits = [] if self.cfg.feedback.num_policies == 1 else [[] for _ in range(self.cfg.feedback.num_policies)]

        # --- Validate configuration for explore_only mode ---
        if self.cfg.get('explore_only', False):
            if self.testers:
                raise ValueError("Testers are not allowed in explore_only mode.")
            allowed_savers = (VisitsSaver, VisitsImageSaver, ConfSaver) # Allow ConfSaver too
            for saver in self.savers:
                if not isinstance(saver, allowed_savers):
                    raise ValueError(f"Saver type '{type(saver).__name__}' is not allowed in explore_only mode. "
                                     f"Only {', '.join(s.__name__ for s in allowed_savers)} are permitted.")

    def calculate_cosine_error(self, est_weight, gt_weight):
        """Calculate cosine error between two weight vectors"""
        est_weight = est_weight.cpu()
        gt_weight = gt_weight.cpu()
        # Compute cosine similarity and convert it to an error metric
        cos_sim = torch.nn.functional.cosine_similarity(est_weight.flatten(), gt_weight.flatten(), dim=0)
        return 1 - cos_sim.item()
    
    def run(self):
        total_episodes = self.cfg.experiment.episodes
        est_freq = self.cfg.feedback.adaptive_estimation_frequency
        est_start = self.cfg.feedback.adaptive_estimation_start
        num_policies = self.cfg.feedback.num_policies

        all_visits = [[] for _ in range(num_policies)]
        recent_visits_buffer = [[] for _ in range(num_policies)]

        def update_callback(ep_idx, new_visits_for_this_episode):
            for policy_idx, single_visit in enumerate(new_visits_for_this_episode):
                all_visits[policy_idx].append(single_visit)
                recent_visits_buffer[policy_idx].append(single_visit)

            if est_freq > 0 and ep_idx < total_episodes - 1 and ep_idx >= est_start and ep_idx % est_freq == 0:
                self.feedback.collect_labels(self.cfg, recent_visits_buffer, self._theta_star)
                self.feedback.fit_estimator()
                self.estimator = self.feedback.estimator  # Store the updated estimator
                self.design.update_estimator(self.estimator, self.env.emissions)
                
                # Calculate and print cosine error
                est_weight = self.estimator.theta_fit
                gt_weight = self._scorer_model.weight
                error = self.calculate_cosine_error(est_weight, gt_weight)
                print(f"Episode {ep_idx} partial re-fit complete. Cosine error: {error:.4f}")
                
                for p_i in range(num_policies):
                    recent_visits_buffer[p_i].clear()

        results = self.explorer.run(
            episodes=total_episodes,
            return_visitations=True,
            update_callback=update_callback
        )
        # Extract only the visitations (third element) from the results tuple
        # MdpExploreMultiPolicy.run returns (objective_values, opt, visitations_per_policy)
        self.visits = results[2] if isinstance(results, tuple) and len(results) == 3 else results

        if any(len(buf) > 0 for buf in recent_visits_buffer):
            self.feedback.collect_labels(self.cfg, recent_visits_buffer, self._theta_star)
        elif est_start == 0:
            self.feedback.collect_labels(self.cfg, all_visits, self._theta_star)

        self.feedback.fit_estimator()
        self.estimator = self.feedback.estimator  # Store the fitted estimator
        self.design.update_estimator(self.estimator, self.env.emissions)
        
        # Calculate and print final cosine error
        est_weight = self.estimator.theta_fit
        gt_weight = self._scorer_model.weight
        error = self.calculate_cosine_error(est_weight, gt_weight)
        print(f"Final estimation after all {total_episodes} episodes complete. Cosine error: {error:.4f}")

    def run_explore_only(self):
        """Runs only the exploration phase and saves visits/images."""
        total_episodes = self.cfg.experiment.episodes
        print(f"Running exploration for {total_episodes} episodes...")

        # Run exploration without estimation callback
        results = self.explorer.run(
            episodes=total_episodes,
            return_visitations=True,
            update_callback=None # No intermediate estimation
        )
        # Extract the actual visits (third element of the tuple)
        # The solver returns (objective_values, final_objective, visits)
        self.visits = results[2]
        self.estimator = None # Ensure estimator is None

        print("Exploration complete. Saving results...")
        self.save_results_explore_only()

    def save_results_explore_only(self):
        """Saves results specifically for explore_only mode (visits, images, config)."""
        from components.results import ExperimentResults
        from components.saver import VisitsSaver, VisitsImageSaver, ConfSaver # Import allowed savers

        results = ExperimentResults()
        results.set_visits(self.visits)
        results.set_estimator(None) # Explicitly set estimator to None

        # Ensure savers have the correct results_dir (might change in test_only)
        for saver in self.savers:
            if saver.results_dir != self.results_dir:
                print(f"Updating saver {type(saver).__name__} results_dir from {saver.results_dir} to {self.results_dir}")
                saver.results_dir = self.results_dir

        # Add essential metadata
        results.add_metadata('mode', 'explore_only')
        results.add_metadata('horizon', self.cfg.horizon)
        results.add_metadata('algorithm', self.cfg.algorithm)
        results.add_metadata('base_prompt', self.cfg.base_prompt)
        results.add_metadata('embedder_class', self.embedder.__class__.__name__)
        results.add_metadata('embedder_model_id', self.embedder.model_id)
        results.add_metadata('embedder_normalize', self.embedder.normalize)
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        results.add_metadata('config_dict', config_dict)

        # Run only the allowed savers
        allowed_savers = (VisitsSaver, VisitsImageSaver, ConfSaver)
        for saver in self.savers:
            if isinstance(saver, allowed_savers):
                print(f"Running saver: {type(saver).__name__}")
                saver.save_result(results)
            else:
                # This check is redundant due to __init__ validation, but safe
                print(f"Skipping disallowed saver: {type(saver).__name__}")

    def _init_env(self):
        """
        Builds token lists, sets up LLMGrid using the configured embedder,
        and initializes the ground-truth scoring model and theta_star.
        """
        # Embedder is already initialized in self.embedder

        # Create token_lists for training environment
        horizon = self.cfg.horizon
        if self.cfg.algorithm == "optim":
            # For optim, we need to create all possible combinations
            combos = cartesian(self.training_words)
            words_list = []
            for row in combos:
                tokens = [t for t in row if t != " "]
                words_list.append(", ".join(tokens))
            token_lists = [words_list]
        else:
            # Use the horizon-specific token lists
            token_lists = self.training_words


        # Build environment, passing the initialized embedder
        env = LLMGrid(
            list_of_text_tokens=token_lists, # Corrected keyword argument
            embedder=self.embedder, # Pass the embedder instance
            base_prompt=self.cfg.base_prompt,
            include_base_prompt_in_first_tokens=self.cfg.include_base_prompt_in_first_tokens,
            verbose=self.cfg.verbose
        )

        # --- Initialize Scorer Model and Theta Star (only if scorer_model is specified) ---
        scorer_model_name = self.cfg.experiment.get('scorer_model') # Use .get() for safety
        if scorer_model_name:
            print(f"Initializing ground truth scorer model: {scorer_model_name}")
            # Build scorer model using the environment and the embedder
            self._scorer_model = get_scorer_model(
                model_name=scorer_model_name,
                env=env,
                embedder=self.embedder # Pass embedder
            )
            # Create the ground truth function using the scorer model
            self._theta_star = make_theta_star(env, self._scorer_model, verbose=self.cfg.verbose)
            # Store scorer vector if needed (optional, depends on usage)
            # self.env._scorer_vector = self._scorer_model.weight
        else:
            print("Skipping ground truth scorer model initialization (scorer_model not specified or null).")
            self._scorer_model = None
            self._theta_star = None



        return env

    def load_estimator(self, estimator_path):
        """Load a pre-computed estimator from file
        
        Args:
            estimator_path: Path to the saved estimator file
            
        Returns:
            True if estimator loaded successfully, False otherwise
        """
        try:
            print(f"Loading estimator from {estimator_path}")
            if not os.path.exists(estimator_path):
                print(f"Error: Estimator file not found at {estimator_path}")
                return False
                
            # Load the theta tensor
            theta = torch.load(estimator_path)

            # Ensure theta is on the right device (use embedder's device)
            device = self.embedder.device
            theta = theta.to(device)

            # Setup feedback components with the loaded theta
            self.feedback.fit_estimator(preloaded_theta=theta)
            self.estimator = self.feedback.estimator
            self.design.update_estimator(self.estimator, self.env.emissions)
            
            print(f"Successfully loaded estimator with shape {theta.shape}")
            return True
        except Exception as e:
            print(f"Error loading estimator: {e}")
            return False

    def _process_human_feedback(self, visits_data, feedback_data):
        """
        Processes human feedback JSON and visits data to generate embeddings and labels
        suitable for training the RegularizedMultinomialEstimator.

        Args:
            visits_data: Loaded visits structure List[List[Tuple(states, actions)]].
            feedback_data: Dictionary loaded from feedback.json.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): (comparison_embeddings, labels)
                - comparison_embeddings: Shape [num_samples, num_policies, embedding_dim]
                - labels: Shape [num_samples, num_policies] (one-hot)
            Returns (None, None) if processing fails or no valid feedback is found.
        """
        print("Processing human feedback...")
        collected_comparison_embeddings = []
        collected_labels = []
        num_policies = len(visits_data) # Infer number of policies from visits structure

        if num_policies == 0:
             print("Error: Cannot process feedback, visits data has zero policies.")
             return None, None

        # Regex to extract episode and timestep from filename (adjust if format changes)
        # Example: images/episode_000_timestep_04.png
        filename_pattern = re.compile(r"episode_(\d+)_timestep_(\d+)\.png$")

        processed_count = 0
        skipped_count = 0
        error_count = 0

        for image_filename, preferred_policy_idx_1based in feedback_data.items():
            match = filename_pattern.search(image_filename)
            if not match:
                # print(f"Warning: Skipping feedback entry, could not parse filename: {image_filename}")
                skipped_count += 1
                continue

            try:
                episode_idx = int(match.group(1))
                timestep_h = int(match.group(2))

                # Validate preferred policy index (must be between 1 and num_policies)
                if not (1 <= preferred_policy_idx_1based <= num_policies):
                     print(f"Warning: Skipping feedback for {image_filename}. Invalid preferred_policy_idx: {preferred_policy_idx_1based} (num_policies={num_policies})")
                     skipped_count += 1
                     continue

                # Retrieve and truncate actions for all policies
                embeddings_for_this_comparison = []
                valid_comparison = True
                for k in range(num_policies):
                    try:
                        # visits_data[k][episode_idx] = (states, actions)
                        actions_policy_k = visits_data[k][episode_idx][1]
                        if isinstance(actions_policy_k, torch.Tensor):
                            actions_policy_k = actions_policy_k.cpu().numpy()
                        actions_policy_k = list(map(int, actions_policy_k)) # Ensure list of ints

                        # Check if timestep_h is valid for this action sequence length
                        if timestep_h > len(actions_policy_k):
                             print(f"Warning: Skipping feedback for {image_filename}, policy {k}. Timestep h={timestep_h} exceeds action length {len(actions_policy_k)}.")
                             valid_comparison = False
                             break # Skip this entire comparison if one policy is invalid

                        truncated_actions_policy_k = actions_policy_k[:timestep_h]

                        # Recreate prompt and embed
                        prompt_k = create_prompt(truncated_actions_policy_k, self.env)
                        embedding_k = self.embedder.embed_text(prompt_k) # Shape [1, dim]
                        embeddings_for_this_comparison.append(embedding_k.detach().cpu())

                    except IndexError:
                        print(f"Warning: Skipping feedback for {image_filename}. Missing visit data for policy {k}, episode {episode_idx}.")
                        valid_comparison = False
                        break # Skip this entire comparison
                    except Exception as e:
                         print(f"Error processing policy {k} for {image_filename}: {e}")
                         valid_comparison = False
                         error_count += 1
                         break # Skip this entire comparison

                if not valid_comparison:
                    skipped_count += 1
                    continue # Move to the next feedback item

                # Stack embeddings for this comparison
                comparison_tensor = torch.cat(embeddings_for_this_comparison, dim=0) # Shape [num_policies, dim]
                collected_comparison_embeddings.append(comparison_tensor)

                # Create one-hot label
                label_tensor = torch.zeros(num_policies)
                label_tensor[preferred_policy_idx_1based - 1] = 1 # Convert 1-based index to 0-based
                collected_labels.append(label_tensor)
                processed_count += 1

            except ValueError as e: # Catch potential int conversion errors
                print(f"Warning: Skipping feedback entry for {image_filename} due to parsing error: {e}")
                skipped_count += 1
            except Exception as e: # Catch other unexpected errors during processing
                 print(f"Error processing feedback entry for {image_filename}: {e}")
                 error_count += 1
                 skipped_count += 1

        print(f"Human feedback processing complete. Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}")

        if not collected_comparison_embeddings or not collected_labels:
            print("Error: No valid comparison data generated from human feedback.")
            return None, None

        # Stack final tensors
        final_comparison_embeddings = torch.stack(collected_comparison_embeddings, dim=0) # [num_samples, num_policies, dim]
        final_labels = torch.stack(collected_labels, dim=0) # [num_samples, num_policies]

        print(f"Generated training data shapes: Embeddings {final_comparison_embeddings.shape}, Labels {final_labels.shape}")
        return final_comparison_embeddings, final_labels


    def run_test_only(self, estimator_path):
        """
        Run in test-only mode. Behavior depends on provided paths:
        - estimator_path provided: Load estimator, test, save results.
        - visits_path provided, estimator_path=None: Load visits, inspect, save results.
        - visits_path and feedback_path provided, estimator_path=None: Load visits & feedback, train estimator, save estimator & results.
        Args:
            estimator_path: Path to the saved estimator file
            
        Returns:
            True if the test/save process was executed, False otherwise (e.g., input path missing).
        Args:
            estimator_path: Path to the saved estimator file (can be None).

        Returns:
            True if the process was executed successfully, False otherwise.
        """
        # --- Determine Mode based on Inputs ---
        visits_path = self.cfg.get('visits_path')
        feedback_path = self.cfg.get('feedback_path') # Get feedback path from config
        mode = None
        input_path_for_dir = None # Path used to determine results directory

        if estimator_path:
            mode = "test"
            input_path_for_dir = estimator_path
            print(f"--- Running in Test Mode (Loading Estimator: {estimator_path}) ---")
        elif visits_path and feedback_path:
            mode = "train_human_feedback"
            input_path_for_dir = visits_path # Use visits path for dir structure
            print(f"--- Running in Train Human Feedback Mode (Visits: {visits_path}, Feedback: {feedback_path}) ---")
        elif visits_path:
            mode = "inspect"
            input_path_for_dir = visits_path
            print(f"--- Running in Inspect Mode (Loading Visits: {visits_path}) ---")
        else:
            print("Error: Invalid combination of paths for test_only mode.")
            print("Provide either 'estimator_path', or 'visits_path', or both 'visits_path' and 'feedback_path'.")
            return False

        # --- Validate Input Paths ---
        absolute_input_path_for_dir = to_absolute_path(input_path_for_dir)
        if not os.path.exists(absolute_input_path_for_dir):
            print(f"Error: Input path for directory structure ('{absolute_input_path_for_dir}') not found.")
            return False
        # Specific checks for train_human_feedback mode
        if mode == "train_human_feedback":
            absolute_feedback_path = to_absolute_path(feedback_path)
            if not os.path.exists(absolute_feedback_path):
                 print(f"Error: Feedback path ('{absolute_feedback_path}') not found for train_human_feedback mode.")
                 return False

        # --- Setup Results Directory ---
        if not self.cfg.get('override_results_dir', False):
            original_dir = os.path.dirname(absolute_input_path_for_dir)
            if os.path.exists(original_dir):
                timestamp = os.environ.get('TIMESTAMP', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
                try:
                    algorithm = self._get_algorithm_code()
                    feedback_type = self._get_feedback_code()
                except ValueError: # Handle cases where cfg might not have algorithm/feedback
                    algorithm = "unknown_alg"
                    feedback_type = "unknown_fb"
                    print("Warning: Could not determine algorithm/feedback from config, using defaults for directory name.")

                # Use the determined mode ('test', 'inspect', 'train_human_feedback') in the directory name
                experiment_id_suffix = self.experiment_id or mode # Use existing ID or mode name

                # Check if the original_dir already ends with 'additional_tests'
                if os.path.basename(original_dir) == "additional_tests":
                    # If yes, use the original_dir itself as the base for new test folders
                    tests_base_dir = original_dir
                    print(f"Parent directory '{original_dir}' is already 'additional_tests'. Using it as base.")
                else:
                    # Otherwise, create 'additional_tests' inside the original_dir
                    tests_base_dir = os.path.join(original_dir, "additional_tests")
                    print(f"Using original results directory parent: {original_dir}")

                # Example: test-dsn-mult-..., inspect-dsn-mult-..., train_human_feedback-dsn-mult-...
                self.results_dir = os.path.join(tests_base_dir, f"{mode}-{algorithm}-{feedback_type}-{timestamp}")
                print(f"Saving {mode} results to: {self.results_dir}")
                os.makedirs(self.results_dir, exist_ok=True)

                # Update results_dir for all savers
                for saver in self.savers:
                    saver.results_dir = self.results_dir
                    print(f"Updated saver {type(saver).__name__} to use results_dir: {self.results_dir}")
            else:
                 print(f"Warning: Could not find original directory '{original_dir}'. Using default results_dir '{self.results_dir}'.")

        # Attempt to load estimator only if estimator_path is provided
        if estimator_path:
            print(f"Attempting to load estimator from: {estimator_path}")
            if not self.load_estimator(estimator_path):
                 print("Warning: Failed to load estimator, proceeding without it.")
                 self.estimator = None # Ensure estimator is None if loading failed
            else:
                 print("Successfully loaded estimator.")
        # --- Train Estimator from Human Feedback ---
        elif mode == "train_human_feedback":
             print("Loading visits and feedback data for training...")
             try:
                 # Load visits
                 loaded_visits = torch.load(absolute_input_path_for_dir) # Load from visits_path
                 # Load feedback JSON
                 with open(absolute_feedback_path, 'r') as f:
                     feedback_json = json.load(f)

                 # Process feedback to get training data
                 comparison_embeddings, labels = self._process_human_feedback(loaded_visits, feedback_json)

                 if comparison_embeddings is None or labels is None:
                      print("Error: Failed to generate training data from feedback. Cannot train estimator.")
                      return False # Stop execution

                 # Initialize Estimator (assuming Multinomial for now)
                 # TODO: Make estimator type configurable if needed
                 print("Initializing estimator for training...")
                 embed_dim = self.embedder.get_embedding_dim()
                 # Use a dummy identity embedding as fit takes embeddings directly
                 dummy_embedding = CustomEmbedding(embed_dim, lambda x: x, embed_dim)
                 likelihood = MultinomialLikelihood()
                 # Use lambda_est from config, fail if not present
                 lambda_est = self.cfg.feedback.lambda_est # Direct access, will error if missing
                 regularizer = L2Regularizer(lam=lambda_est)
                 estimator = RegularizedMultinomialEstimator(dummy_embedding, likelihood, regularizer)

                 # Fit Estimator
                 print(f"Fitting estimator with {comparison_embeddings.shape[0]} human feedback samples...")
                 estimator.fit(comparison_embeddings=comparison_embeddings, labels=labels)
                 self.estimator = estimator # Store the newly fitted estimator
                 self.visits = loaded_visits # Store loaded visits for potential saving
                 print("Estimator training complete.")

             except FileNotFoundError as e:
                  print(f"Error: Required file not found during training setup: {e}")
                  return False
             except Exception as e:
                  print(f"Error during estimator training from human feedback: {e}")
                  return False
        # --- Inspection Mode (Load Visits Only) ---
        elif mode == "inspect":
             print("No estimator path provided, proceeding without loading estimator (inspection mode).")
             self.estimator = None # Ensure estimator is None
             print(f"Attempting to load visits from: {absolute_input_path_for_dir}")
             try:
                 # Load visits directly into self.visits
                 self.visits = torch.load(absolute_input_path_for_dir)
                 # Basic validation after loading
                 if not isinstance(self.visits, list) or not self.visits or not self.visits[0]:
                      print(f"Warning: Loaded visits from {absolute_input_path_for_dir} appear empty or invalid.")
                      # Decide whether to proceed with empty visits or fail
                      # For now, let the validation in test_and_save handle it.
                 else:
                      print(f"Successfully loaded visits for inspection.")
             # Corrected Indentation for except blocks
             except FileNotFoundError:
                  print(f"Error: Visits file not found at {absolute_input_path_for_dir} during loading attempt.")
                  self.visits = None # Ensure visits is None if loading fails
                  return False # Stop execution if visits file is mandatory and not found
             except Exception as e:
                  print(f"Error loading visits from {absolute_input_path_for_dir}: {e}")
                  self.visits = None # Ensure visits is None if loading fails
                  return False # Stop execution on other loading errors

        # Ensure visits attribute exists, even if loading failed or wasn't attempted
        if not hasattr(self, 'visits'):
             print("Initializing empty visits list as it wasn't loaded or generated.")
             # Infer num_policies from config if possible, else default to 1
             num_policies = self.cfg.feedback.get('num_policies', 1)
             self.visits = [] if num_policies == 1 else [[] for _ in range(num_policies)]

        # Always proceed to test and save (unless loading/training failed above)
        print(f"Proceeding to test_and_save in {mode} mode (estimator is {'fitted' if mode == 'train_human_feedback' else ('loaded' if self.estimator else 'None')}, visits are {'loaded' if self.visits and self.visits[0] else 'not loaded/empty'}).")
        self.test_and_save(current_mode=mode) # Pass the mode to test_and_save
        return True # Indicate test/save process was executed

    def test_and_save(self, current_mode="full_run"): # Add current_mode argument with a default
        """
        Final estimation, testing and saving of results.

        Args:
            current_mode (str): The mode the experiment is running in
                                ('full_run', 'test', 'inspect', 'train_human_feedback').
                                Used for mode-specific validation and behavior.
        """
        # Create a container for all results
        from components.results import ExperimentResults
        results = ExperimentResults()
        
        # Set the estimator and visits
        results.set_estimator(self.estimator)
        # Ensure visits are valid before setting
        valid_visits = self.visits and isinstance(self.visits, list) and self.visits[0]
        results.set_visits(self.visits if valid_visits else None) # Set to None if invalid/empty

        # --- Pre-run Validation ---
        print("Validating requirements for configured testers and savers...")
        estimator_available = self.estimator is not None
        # Check if visits list exists, is not empty, and its first element is not empty
        visits_available = bool(self.visits and isinstance(self.visits, list) and self.visits[0])

        # Import tester/saver classes for isinstance checks
        from components.tester import PreferenceTester, CosineTester, ImageGenerationTester
        from components.saver import LearnedEstimatorSaver, VisitsSaver, VisitsImageSaver, ReadableVisitsSaver, ConfSaver

        # --- Mode-Specific Validation ---
        if current_mode == "train_human_feedback":
            # Ensure LearnedEstimatorSaver is present
            if not any(isinstance(s, LearnedEstimatorSaver) for s in self.savers):
                 raise ValueError(f"Mode '{current_mode}' requires LearnedEstimatorSaver to be configured, but it was not found.")
            # Ensure estimator was actually fitted
            if not estimator_available:
                 raise ValueError(f"Mode '{current_mode}' completed but the estimator is not available. Training likely failed.")
            # Testers are generally skipped in this mode, so no tester validation needed here.
            print(f"Validation for mode '{current_mode}': LearnedEstimatorSaver found.")
        # -----------------------------

        # Validate Testers (Skip if in train_human_feedback mode)
        if current_mode != "train_human_feedback":
            for tester in self.testers:
                tester_name = type(tester).__name__
                if isinstance(tester, (PreferenceTester, CosineTester)):
                    if not estimator_available:
                        raise ValueError(f"Tester '{tester_name}' requires an estimator, but it was not loaded or available.")
                elif isinstance(tester, ImageGenerationTester):
                     # ImageGenerationTester needs scorer_model OR estimator if use_estimator=True
                     if tester.use_estimator and not estimator_available:
                         raise ValueError(f"Tester '{tester_name}' is configured with use_estimator=True, but the estimator is not available.")
                     if not tester.use_estimator and self.scorer_model is None:
                          raise ValueError(f"Tester '{tester_name}' is configured to use the scorer_model, but it's not available.")
                     if self.embedder is None: # Also needs embedder
                          raise ValueError(f"Tester '{tester_name}' requires an embedder, but it's not available.")
                # Add checks for other testers if they have specific requirements
        else:
             print("Skipping tester validation in 'train_human_feedback' mode.")


        # Validate Savers (Common checks for all modes)
        for saver in self.savers:
            saver_name = type(saver).__name__
            if isinstance(saver, LearnedEstimatorSaver):
                if not estimator_available:
                    raise ValueError(f"Saver '{saver_name}' requires an estimator, but it was not loaded or available.")
            elif isinstance(saver, (VisitsSaver, VisitsImageSaver, ReadableVisitsSaver)):
                # More detailed check for visits availability
                if not visits_available:
                    error_reason = "Visit data is required but not available"
                    if self.visits is None:
                        error_reason = "Visit data was not loaded or generated (check visits_path or experiment run)"
                    elif not isinstance(self.visits, list):
                        error_reason = f"Visit data has unexpected type: {type(self.visits).__name__}"
                    elif not self.visits: # Check if the outer list is empty
                        error_reason = "Visit data list is empty"
                    elif not self.visits[0]: # Check if the first policy's list is empty
                         error_reason = "Visit data for the first policy is empty"
                    # Add context about the source path if inspection mode failed
                    if self.cfg.get('test_only', False) and not estimator_available and self.cfg.get('visits_path'):
                         error_reason += f". Attempted load from: {self.cfg.visits_path}"

                    raise ValueError(f"Saver '{saver_name}' requires visit data. Reason: {error_reason}.")
            # VisitsImageSaver and ReadableVisitsSaver also need env, checked in their __init__
            # ImageGenerationSaver needs results populated by ImageGenerationTester, implicitly checked by tester validation

        print("Validation successful.")
        # --- End Validation ---

        # Ensure all savers have the correct results_dir
        for saver in self.savers:
            if saver.results_dir != self.results_dir:
                print(f"Updating saver {type(saver).__name__} results_dir from {saver.results_dir} to {self.results_dir}")
                saver.results_dir = self.results_dir
        
        # Add experiment metadata
        results.add_metadata('horizon', self.cfg.horizon)
        results.add_metadata('algorithm', self.cfg.algorithm)
        results.add_metadata('base_prompt', self.cfg.base_prompt)
        # Add embedder info to metadata
        results.add_metadata('embedder_class', self.embedder.__class__.__name__)
        results.add_metadata('embedder_model_id', self.embedder.model_id)
        results.add_metadata('embedder_normalize', self.embedder.normalize)

        # Add the resolved config as a plain dictionary for the ConfSaver
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        results.add_metadata('config_dict', config_dict)

        # Run all testers and collect metrics (Skip if in train_human_feedback mode)
        all_tester_results = {} # Initialize dictionary to accumulate results
        if current_mode != "train_human_feedback":
            print("Running testers...")
            for tester in self.testers:
                print(f"Running tester: {type(tester).__name__}")
                # Pass visits=self.visits if needed by any tester in the future
                tester_results = tester.run_test( # Get results from the current tester
                    cfg=self.cfg,
                    env=self.env,
                    estimator=self.estimator, # Can be None if tester doesn't need it (but validation would have caught it if it did)
                    theta_star=self._theta_star,
                    training_words_list=self.training_words,
                    testing_words_list=self.testing_words
                    # visits=self.visits # Pass visits if any tester needs them
                )
                # Update the accumulated results dictionary
                if tester_results: # Ensure tester returned something
                    all_tester_results.update(tester_results)
        else:
            print("Skipping testers in 'train_human_feedback' mode.")
        # --- Removed duplicated code block here ---

        # Add all accumulated metrics to the results container
        if all_tester_results:
            results.add_metrics(all_tester_results)

        # Use all savers to save the results (validation ensures requirements are met)
        print("Running savers...")
        # Removed duplicated lines causing IndentationError here
        
        # Use all savers to save the results
        for saver in self.savers:
            # Pass the results object containing potentially loaded estimator/visits
            print(f"Running saver: {type(saver).__name__}")
            saver.save_result(results)

    def _get_algorithm_code(self):
        """Get a short code for the algorithm type"""
        algorithm = self.cfg.algorithm.lower()
        if algorithm == "design":
            return "dsn"
        elif algorithm == "random":
            return "rand"
        elif algorithm == "optim":
            return "opt"
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Expected one of: design, random, optim")
            
    def _get_feedback_code(self):
        """Get a short code for the feedback type"""
        feedback = self.cfg.feedback.name.lower()
        if feedback == "numerical":
            return "num"
        elif feedback == "multinomial":
            return "mult"
        else:
            return feedback[:3]  # First 3 chars as fallback
            
    def _load_data(self):
        """Returns training_words_lists and testing_words_lists for each horizon step"""
        vocab_files = self.cfg.experiment.vocabulary
        
        # Check if horizon matches the number of vocabulary files
        # Make sure vocab_files is a flat list, not a list of lists
        if len(vocab_files) != self.cfg.horizon:
            raise ValueError(f"Number of vocabulary files ({len(vocab_files)}) must match horizon ({self.cfg.horizon})")
        
        rng = np.random.RandomState(42)
        
        training_words_lists = []
        testing_words_lists = []
        
        # Process each vocabulary file separately
        for path in vocab_files:
            # Load words from file
            with open(path, 'r') as f:
                full_list = [line.strip() for line in f]
            full_list = list(dict.fromkeys(full_list))  # Remove duplicates
            
            # Cap vocabulary if needed
            if len(full_list) > self.cfg.experiment.vocab_size:
                print(f"Capping data from {path} at {self.cfg.experiment.vocab_size} items")
                full_list = list(rng.choice(full_list, self.cfg.experiment.vocab_size, replace=False))
            
            # Create 75-25 split for this file
            n_total = len(full_list)
            n_train = int(0.75 * n_total)
            
            indices = rng.permutation(n_total)
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            
            # Add to lists
            training_words_lists.append([full_list[i] for i in train_idx])
            testing_words_lists.append([full_list[i] for i in test_idx])
        
        return training_words_lists, testing_words_lists, []
