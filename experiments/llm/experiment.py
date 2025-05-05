import os
import re # Added import for regular expressions
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
from components.tester import BaseTester, ImageGenerationTester
# Import specific saver types needed for validation and estimator creation
# Removed LearnedEstimatorSaver from direct import here, will handle skipping later
from components.saver import BaseSaver, VisitsSaver, VisitsImageSaver, ConfSaver
# Removed unused estimator/likelihood/regularizer imports here, they are used within factories/components
# from stpy.regression.regularized_dictionary.regularized_multinomial_estimator import RegularizedMultinomialEstimator
# from stpy.probability.multinomial_likelihood import MultinomialLikelihood
# from stpy.regularization.regularizer import L2Regularizer
from stpy.embeddings.polynomial_embedding import CustomEmbedding # Corrected import path
# Removed unused json and re imports

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
    def __init__(self, cfg: DictConfig, derived_results_dir: str = None): # Add derived_results_dir argument
        self.cfg = cfg
        self.rng = np.random.RandomState(int(cfg.seed))
        self.seed = int(cfg.seed) # Store seed as an integer attribute

        # --- Determine and Create Results Directory ---
        if derived_results_dir:
            # Use the path derived in run_exp.py directly (already includes timestamp)
            self.results_dir = derived_results_dir
            print(f"Using derived results directory: {self.results_dir}")
        else:
            # Use default logic: path from config + timestamp
            timestamp = os.environ.get('TIMESTAMP', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
            self.results_dir = f"{cfg.results_dir}-{timestamp}"
            print(f"Using default results directory logic: {self.results_dir}")
        os.makedirs(self.results_dir, exist_ok=True)
        # ---------------------------------------------

        # --- Initialize Embedder ---
        # The create_embedder factory reads cfg.embedder config group
        self.embedder: BaseEmbedder = create_embedder(cfg.embedder)
        print(f"Initialized Embedder: {self.embedder.__class__.__name__} with model {self.embedder.model_id}")

        # --- Handle Scorer Model Configuration ---
        scorer_model_config = cfg.experiment.get('scorer_model')
        if isinstance(scorer_model_config, str):
            self.scorer_model_names = [scorer_model_config]
            print(f"Using single scorer model: {self.scorer_model_names}")
        elif isinstance(scorer_model_config, list):
            self.scorer_model_names = scorer_model_config
            print(f"Using multiple scorer models: {self.scorer_model_names}")
        elif OmegaConf.is_list(scorer_model_config): # Handle OmegaConf ListConfig
             self.scorer_model_names = OmegaConf.to_container(scorer_model_config, resolve=True)
             print(f"Using multiple scorer models (from ListConfig): {self.scorer_model_names}")
        else:
            raise ValueError(f"scorer_model in config must be a string or a list, got {type(scorer_model_config)}")
        self.num_scorer_models = len(self.scorer_model_names)

        # --- Load Data & Initialize Environment ---
        self.training_words, self.testing_words, self.model_words = self._load_data()
        # _init_env now uses self.embedder and populates _scorer_models and _theta_stars
        self.env = self._init_env() # Initializes self._scorer_models and self._theta_stars

        # --- Initialize Core Components (Lists for multiple models) ---
        self.feedbacks = []
        self.designs = []
        self.estimators = [] # Will hold estimator instances for each model

        # Create feedback, design, and estimator for each scorer model
        for i, model_name in enumerate(self.scorer_model_names):
            print(f"Initializing components for scorer model: {model_name}")
            # Pass the specific scorer model instance and name to the factory
            feedback, design, estimator = FeedbackFactory.create(
                cfg,
                self.env,
                self._scorer_models[i], # Pass the specific model instance
                self.embedder,
                scorer_model_name=model_name # Pass the name for lambda lookup
            )
            self.feedbacks.append(feedback)
            self.designs.append(design)
            self.estimators.append(estimator) # Add the initial estimator instance

        # --- Initialize Solver ---
        # The explorer uses the design and feedback from the *first* scorer model
        # The exploration path is the same, but estimation differs per model later
        print(f"Initializing explorer using components from the first model: {self.scorer_model_names[0]}")
        self.explorer = SolverFactory.create(
            cfg,
            self.env,
            self.designs[0], # Use first design
            self.feedbacks[0], # Use first feedback
            same_first_action_in_episode=cfg.get('same_first_action_in_episode', False) # Read from config
        )

        # Note: self.estimator is now self.estimators (a list)
        # For test-only mode, estimators will be loaded into this list later.
        
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
                     # Base arguments needed by testers (via BaseTester)
                     # scorer_model is NOT passed here; it will be passed to run_test
                     init_args = {
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
                # scorer_model is NOT passed here; savers that need it access it via results or don't need it.
                init_args = {
                    'env': self.env,
                    'embedder': self.embedder,
                    # 'scorer_model': self._scorer_model, # REMOVED
                    'results_dir': self.results_dir,
                    'experiment_id': self.experiment_id
                    # 'params' and other config-specific args are handled by Hydra via s_conf
                }

                # Add arguments specific to VisitsImageSaver if it's the target
                # No longer needed - saver derives from env and cfg passed via BaseSaver
                # if s_conf.get('_target_') == 'components.saver.VisitsImageSaver':
                #     pass # init_args['dense_feedback'] = self.cfg.get('dense_feedback', False) etc.

                # Add arguments specific to ReadableVisitsSaver
                # No longer needed - saver derives from env and cfg passed via BaseSaver
                # elif s_conf.get('_target_') == 'components.saver.ReadableVisitsSaver':
                #     pass # init_args['dense_feedback'] = self.cfg.get('dense_feedback', False)

                # Instantiate the saver using the configuration and the constructed arguments
                # Pass the main config `cfg` itself, so savers can access necessary top-level keys
                init_args['cfg'] = self.cfg # Pass the main config object
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
            # Import saver classes locally for isinstance check
            from components.saver import VisitsSaver, VisitsImageSaver, ConfSaver
            allowed_savers = (VisitsSaver, VisitsImageSaver, ConfSaver) # Allow ConfSaver too
            for saver in self.savers:
                if not isinstance(saver, allowed_savers):
                    raise ValueError(f"Saver type '{type(saver).__name__}' is not allowed in explore_only mode. "
                                     f"Only {', '.join(s.__name__ for s in allowed_savers)} are permitted.")

    def calculate_cosine_error(self, est_weight, gt_weight):
        """Calculate cosine error between two weight vectors."""
        # Ensure weights are tensors before moving to CPU
        if isinstance(est_weight, torch.Tensor):
            est_weight = est_weight.cpu()
        if isinstance(gt_weight, torch.Tensor):
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

            # --- Adaptive Estimation Loop (if enabled) ---
            if est_freq > 0 and ep_idx < total_episodes - 1 and ep_idx >= est_start and (ep_idx + 1) % est_freq == 0:
                print(f"\n--- Running Adaptive Estimation at Episode {ep_idx + 1} ---")
                # Loop through each scorer model to collect labels and fit estimator
                for i in range(self.num_scorer_models):
                    feedback = self.feedbacks[i]
                    theta_star = self._theta_stars[i]
                    scorer_model = self._scorer_models[i]
                    model_name = self.scorer_model_names[i]

                    # Check if components are valid for this model
                    if feedback is None or theta_star is None or scorer_model is None:
                        print(f"Skipping adaptive estimation for model '{model_name}': Missing components.")
                        continue

                    print(f"Collecting labels for model: {model_name}")
                    feedback.collect_labels(self.cfg, recent_visits_buffer, theta_star)
                    print(f"Fitting estimator for model: {model_name}")
                    feedback.fit_estimator()
                    self.estimators[i] = feedback.estimator # Update the specific estimator

                    # Calculate and print cosine error for this model
                    if self.estimators[i] and hasattr(self.estimators[i], 'theta_fit') and hasattr(scorer_model, 'weight'):
                        est_weight = self.estimators[i].theta_fit
                        gt_weight = scorer_model.weight
                        error = self.calculate_cosine_error(est_weight, gt_weight)
                        print(f"Episode {ep_idx + 1} partial re-fit complete for model '{model_name}'. Cosine error: {error:.4f}")
                    else:
                        print(f"Could not calculate cosine error for model '{model_name}' (estimator or ground truth weight missing).")

                # Clear the buffer after processing all models for this frequency step
                print("Clearing recent visits buffer.")
                for p_i in range(num_policies):
                    recent_visits_buffer[p_i].clear()
                print("-----------------------------------------------------\n")

        results = self.explorer.run(
            episodes=total_episodes,
            return_visitations=True,
            update_callback=update_callback
        )
        # Extract only the visitations (third element) from the results tuple
        # MdpExploreMultiPolicy.run returns (objective_values, opt, visitations_per_policy)
        # Ensure results is a tuple and has 3 elements before accessing index 2
        if isinstance(results, tuple) and len(results) == 3:
            self.visits = results[2]
        else:
            # Handle cases where explorer might return something else (e.g., just visits)
            # Or log a warning/error if the structure is unexpected
            print(f"Warning: Unexpected explorer result structure: {type(results)}. Assuming it contains visits.")
            self.visits = results # Assign directly, hoping it's the visits

        # --- Final Estimation Loop ---
        print("\n--- Running Final Estimation ---")
        # Determine which visits buffer to use for final collection
        final_visits_to_process = None
        if any(len(buf) > 0 for buf in recent_visits_buffer):
            print("Using remaining recent visits buffer for final estimation.")
            final_visits_to_process = recent_visits_buffer
        elif est_start == 0: # If estimation never happened adaptively, use all visits
            print("Using all visits for final estimation (adaptive estimation start was 0).")
            final_visits_to_process = all_visits
        else:
            print("No remaining visits in buffer and adaptive estimation occurred. Final estimation based on last adaptive fit.")
            # In this case, estimators are already fitted, just print final errors below.

        # Loop through each scorer model for final label collection (if needed) and fitting
        for i in range(self.num_scorer_models):
            feedback = self.feedbacks[i]
            theta_star = self._theta_stars[i]
            scorer_model = self._scorer_models[i]
            model_name = self.scorer_model_names[i]

            # Check if components are valid
            if feedback is None or theta_star is None or scorer_model is None:
                print(f"Skipping final estimation for model '{model_name}': Missing components.")
                continue

            # Collect labels only if there are visits to process from this run
            if final_visits_to_process:
                print(f"Collecting final labels for model: {model_name}")
                feedback.collect_labels(self.cfg, final_visits_to_process, theta_star)
                print(f"Fitting final estimator for model: {model_name}")
                feedback.fit_estimator()
                self.estimators[i] = feedback.estimator # Update the specific estimator
            else:
                # If no new visits, the estimator should be the one from the last adaptive step
                # Ensure self.estimators[i] exists from previous steps
                if i >= len(self.estimators) or self.estimators[i] is None:
                     print(f"Warning: Estimator for model '{model_name}' not found from previous steps.")
                     continue # Skip error calculation if no estimator exists

            # Calculate and print final cosine error for this model
            if self.estimators[i] and hasattr(self.estimators[i], 'theta_fit') and hasattr(scorer_model, 'weight'):
                est_weight = self.estimators[i].theta_fit
                gt_weight = scorer_model.weight
                error = self.calculate_cosine_error(est_weight, gt_weight)
                print(f"Final estimation for model '{model_name}' after {total_episodes} episodes complete. Cosine error: {error:.4f}")
            else:
                print(f"Could not calculate final cosine error for model '{model_name}' (estimator or ground truth weight missing).")
        print("------------------------------\n")

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
        results.add_metadata('horizon', self.env.max_episode_length) # Log horizon from env
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
        # horizon = self.cfg.horizon # REMOVED - Horizon is determined by len(list_of_text_tokens)
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

        # Initialize lists for models and thetas
        self._scorer_models = []
        self._theta_stars = []

        # Loop through configured scorer model names
        for model_name in self.scorer_model_names:
            scorer_model_instance = None # Initialize for this iteration
            if model_name:
                print(f"Initializing ground truth scorer model: {model_name}")
                # Build scorer model using the environment and the embedder
                scorer_model_instance = get_scorer_model(
                    model_name=model_name,
                    env=env,
                    embedder=self.embedder # Pass embedder
                )
                # Create the ground truth function using the scorer model
                theta_star_instance = make_theta_star(env, scorer_model_instance, verbose=self.cfg.verbose)
            else:
                # Handle case where a model name might be null/empty in the list
                print("Warning: Encountered null/empty scorer_model name. Skipping ground truth initialization for this entry.")
                scorer_model_instance = None
                theta_star_instance = None

            # Append the instances (or None) to the lists
            self._scorer_models.append(scorer_model_instance)
            self._theta_stars.append(theta_star_instance)

        # Store scorer vector if needed (optional, depends on usage) - This might need adjustment if used
        # Example: Store the first model's weight if needed elsewhere
        # if self._scorer_models and self._scorer_models[0] is not None:
        #     self.env._scorer_vector = self._scorer_models[0].weight

        return env

    def load_estimator(self, estimator_path):
        """
        Load a pre-computed estimator from file.
        NOTE: Currently only supports loading a *single* estimator for the *first* model
              when multiple scorer models are configured.
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

            # --- Basic Multi-Model Handling ---
            if self.num_scorer_models > 1:
                print(f"Warning: Loading estimator from {estimator_path} for the *first* model ({self.scorer_model_names[0]}) only.")

            # Setup feedback components for the first model with the loaded theta
            # Ensure lists are populated before accessing index 0
            if not self.feedbacks or not self.estimators:
                 print("Error: Feedback/Estimator lists not initialized before loading estimator.")
                 return False

            self.feedbacks[0].fit_estimator(preloaded_theta=theta)
            self.estimators[0] = self.feedbacks[0].estimator # Update the first estimator instance
            # Skip design update: self.designs[0].update_estimator(self.estimators[0], self.env.emissions)

            print(f"Successfully loaded estimator for model '{self.scorer_model_names[0]}' with shape {theta.shape}")
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

        # --- Access the nested 'preferences' dictionary ---
        preferences_dict = feedback_data.get("preferences")
        if not preferences_dict or not isinstance(preferences_dict, dict):
            print("Error: 'preferences' key not found or is not a dictionary in feedback data.")
            return None, None
        # -------------------------------------------------

        # Iterate through the items in the preferences dictionary
        for image_filename, preferred_policy_idx_1based in preferences_dict.items():
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

    # Removed unused _process_human_feedback method

    def run_test_only(self, mode: str, estimator_path: str = None, visits_path: str = None, feedback_path: str = None) -> bool:
        """
        Run in test-only mode. Behavior depends on provided paths:
        - estimator_path provided: Load estimator, test, save results. Optionally use feedback_path to override base_prompt.
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
        # Mode is now passed directly as an argument
        print(f"--- Running Test-Only Mode: {mode} ---")

        # --- Setup Results Directory (Handled by run_exp.py now) ---
        # The results_dir is set in __init__ and potentially overridden by run_exp.py

        # --- Basic Multi-Model Handling ---
        if self.num_scorer_models > 1:
            print(f"Warning: Test-only mode currently has limited support for multiple scorer models.")
            if mode == "load_estimator":
                print("-> Will load estimator for the first model only.")
            elif mode == "estimate_from_visits":
                print("-> Will estimate for the first model only.")
            elif mode == "load_estimator_and_feedback":
                 print("-> Will load estimator for the first model only. Feedback processing might affect environment for all.")
            # 'train_human_feedback' mode needs significant changes to support multiple models, likely unsupported for now.
            if mode == "train_human_feedback":
                 print("Error: Mode 'train_human_feedback' is not supported with multiple scorer models.")
                 return False

        # --- Mode 1: Load Estimator ---
        if mode == "load_estimator":
            #     mode = "test"
            #     ...
            # elif visits_path and feedback_path:
            #     mode = "train_human_feedback"
            #     ...
            # elif visits_path:
            #     mode = "inspect"
            #     ...
            # else:
            #     ...
            #     return False

            # Indentation added for the block below
            print(f"--- Running Test-Only Mode: {mode} ---") # This line was moved inside the if block

            # --- Setup Results Directory (Handled by run_exp.py now) ---
            # The results_dir is set in __init__ and potentially overridden by run_exp.py
        # We just need to ensure savers use the final self.results_dir before saving.

        # --- Mode 1: Load Estimator ---
        if mode == "load_estimator": # This if statement needs the block below indented
            # Indentation added for the block below
            if not estimator_path or not os.path.exists(estimator_path):
                print(f"Error: Estimator file not found or not provided: {estimator_path}")
                return False
            print(f"Loading estimator from: {estimator_path}")
            if not self.load_estimator(estimator_path):
                 print("Error: Failed to load estimator.")
                 return False
            # Proceed to testing
            self.test_and_save(current_mode=mode)
            return True

        # --- Mode 2: Estimate from Visits ---
        elif mode == "estimate_from_visits":
            if not visits_path or not os.path.exists(visits_path):
                print(f"Error: Visits file not found or not provided: {visits_path}")
                return False
            print(f"Loading visits from: {visits_path}")
            # Load visits directly into self.visits
            try:
                self.visits = torch.load(visits_path)
                if not isinstance(self.visits, list) or not self.visits or not self.visits[0]:
                     print(f"Warning: Loaded visits from {visits_path} appear empty or invalid.")
                     # Proceed, test_and_save validation will handle it if savers need visits
            except Exception as e:
                 print(f"Error loading visits from {visits_path}: {e}")
                 return False # Stop execution on loading errors

            print("Estimating/Training estimator using visits...")
            # Train the estimator using the loaded visits.
            # Assumes the configured feedback mechanism (e.g., PairwiseFeedback)
            # can handle feedback_data=None or raises an appropriate error.
            # If NumericalFeedback is used, this might just return the scorer_model.
            try:
                # Step 1: Collect labels using visits and ground truth scorer
                print("Collecting labels from visits using ground truth scorer...")
                # Ensure _theta_star is available
                # Use the first model's theta_star, consistent with limited multi-model support
                if not self._theta_stars or self._theta_stars[0] is None:
                     raise ValueError("Ground truth scorer (_theta_stars[0]) is not available. Cannot collect labels.")
                # Use the first feedback mechanism and first theta_star
                self.feedbacks[0].collect_labels(self.cfg, self.visits, self._theta_stars[0])

                # Step 2: Fit the estimator using the collected labels
                print("Fitting estimator with collected labels...")
                self.feedbacks[0].fit_estimator() # Uses internally stored data

                # Step 3: Retrieve the fitted estimator (for the first model)
                self.estimators[0] = self.feedbacks[0].estimator # Get the estimator instance

            except NotImplementedError as e:
                 print(f"Error: The configured feedback mechanism ({self.feedbacks[0].__class__.__name__}) does not support training from visits alone.")
                 print(e)
                 return False
            except AttributeError as e: # Catch the specific error if collect_labels/fit_estimator are missing
                 print(f"Error: Method missing in feedback class ({self.feedbacks[0].__class__.__name__}): {e}")
                 return False
            except Exception as e:
                 print(f"Error during estimator training from visits (model 0): {e}")
                 # Optionally re-raise for more detail: raise e
                 return False

            # Check if the first estimator was successfully fitted/retrieved
            if self.estimators and self.estimators[0] and getattr(self.estimators[0], 'fitted', False):
                print("Estimator for the first model trained/obtained successfully.")
                # Proceed to testing (will handle multiple models internally)
                self.test_and_save(current_mode=mode)
                return True
            else:
                print("Error: Feedback processing did not return a valid estimator from visits.")
                return False

        # --- Mode 3: Load Estimator and Feedback ---
        elif mode == "load_estimator_and_feedback":
            if not estimator_path or not os.path.exists(estimator_path):
                print(f"Error: Estimator file not found or not provided: {estimator_path}")
                return False
            if not feedback_path or not os.path.exists(feedback_path):
                print(f"Error: Feedback file not found or not provided: {feedback_path}")
                return False

            print(f"Loading estimator from: {estimator_path}")
            if not self.load_estimator(estimator_path):
                 print("Error: Failed to load estimator.")
                 return False

            print(f"Loading feedback data from: {feedback_path}")
            # Load feedback data using the feedback component's method
            # Store it for potential use by testers/savers (e.g., base prompt override)
            try:
                # Use the first feedback mechanism
                self.feedback_data = self.feedbacks[0].load_feedback(feedback_path)
                if self.feedback_data is None:
                    print("Error: Failed to load feedback data (returned None).")
                    return False
                print("Feedback data loaded.")
                # --- Optional: Re-initialize environment if user_prompt is found ---
                user_prompt = self.feedback_data.get("user_prompt")
                if user_prompt is not None and isinstance(user_prompt, str):
                    print(f"Found user_prompt: '{user_prompt}'. Re-initializing environment without bases.txt.")
                    # Prepare new vocab list excluding bases.txt
                    new_vocab_files = [vf for vf in self.cfg.experiment.vocabulary if 'bases.txt' not in vf]
                    if len(new_vocab_files) == len(self.cfg.experiment.vocabulary):
                        print("Warning: 'bases.txt' not found in original vocabulary list. Environment not changed.")
                        # If bases.txt wasn't there, still use the user_prompt as base_prompt
                        # but keep the original vocabulary and set include_base_prompt_in_first_tokens=False
                        # Re-initialize env with modified base_prompt, keeping original vocab
                        # Need to modify _init_env or create a helper to handle this
                        print("Warning: Environment re-initialization with modified base_prompt not fully implemented yet.")
                        # For now, just update the env's base_prompt attribute directly
                        self.env.base_prompt = user_prompt
                        self.env.include_base_prompt_in_first_tokens = False
                        # Note: Emissions might need regeneration if base_prompt changes significantly
                        print(f"Updated env.base_prompt to '{user_prompt}'. Emissions not regenerated.")
                    else:
                        # Re-initialize env with the user_prompt as base_prompt and the reduced vocabulary.
                        # Horizon is implicitly set by len(new_vocab_files).
                       # Set include_base_prompt_in_first_tokens=False as the base is now explicit.
                       # Need to modify _init_env or create a helper to handle this
                       print("Warning: Environment re-initialization with modified vocabulary not fully implemented yet.")
                       # For now, just update the env's base_prompt attribute directly
                       self.env.base_prompt = user_prompt
                       self.env.include_base_prompt_in_first_tokens = False
                       # Note: Emissions and token lists need regeneration
                       print(f"Updated env.base_prompt to '{user_prompt}'. Vocabulary/Emissions not regenerated.")

                   # Update components dependent on env.emissions (Skip design update)
                   # if hasattr(self, 'designs') and self.designs and hasattr(self.designs[0], 'update_estimator'):
                   #      # Ensure estimator is available before updating design
                    #      if self.estimators and self.estimators[0]:
                    #          # self.designs[0].update_estimator(self.estimators[0], self.env.emissions)
                    #          # print("Design objective updated with new environment emissions.")
                    #          pass # Skipping design update
                    #      else:
                    #          print("Warning: Estimator not available when trying to update design objective.")
                    # else:
                    #      print("Warning: Could not update design objective after environment re-initialization.")
                    print("Environment re-initialized.")
                else:
                    print("Optional 'user_prompt' not found in feedback data or not a string. Using environment initialized from config.")
                # -----------------------------------------------------------------
            except Exception as e:
                 print(f"Error loading feedback data or re-initializing environment: {e}")
                 return False

            # Proceed to testing with potentially modified environment
            self.test_and_save(current_mode=mode)
            return True
        else:
             # Should not happen if logic in run_exp.py is correct
             print(f"Error: Unknown test_only mode '{mode}' received by experiment runner.")
             return False

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

        # Set the list of estimators and visits
        results.set_estimators(self.estimators) # Use the list of estimators
        # Ensure visits are valid before setting
        valid_visits = self.visits and isinstance(self.visits, list) and self.visits[0]
        results.set_visits(self.visits if valid_visits else None) # Set to None if invalid/empty

        # --- Pre-run Validation ---
        print("Validating requirements for configured testers and savers...")
        # Check if *at least one* estimator is available in the list
        estimators_available = self.estimators and any(est is not None for est in self.estimators)
        # Check if visits list exists, is not empty, and its first element is not empty
        visits_available = bool(self.visits and isinstance(self.visits, list) and self.visits[0])

        # Import tester/saver classes for isinstance checks
        from components.tester import PreferenceTester, CosineTester, ImageGenerationTester
        from components.saver import LearnedEstimatorSaver, VisitsSaver, VisitsImageSaver, ReadableVisitsSaver, ConfSaver

        # --- Mode-Specific Validation ---
        if current_mode == "train_human_feedback":
            # Ensure LearnedEstimatorSaver is present
            # Check if LearnedEstimatorSaver is present (it will be skipped if num_models > 1)
            has_les = any(isinstance(s, LearnedEstimatorSaver) for s in self.savers)
            if not has_les:
                 print(f"Warning: Mode '{current_mode}' typically uses LearnedEstimatorSaver, but it was not found in config.")
            # Ensure at least one estimator was actually fitted
            if not estimators_available:
                 raise ValueError(f"Mode '{current_mode}' completed but no estimator is available. Training likely failed.")
            # Testers are generally skipped in this mode, so no tester validation needed here.
            print(f"Validation for mode '{current_mode}': Estimator(s) available.")
        # -----------------------------

        # Validate Testers (Skip if in train_human_feedback mode)
        if current_mode != "train_human_feedback":
            for tester in self.testers:
                tester_name = type(tester).__name__
                # Check if *any* estimator is available if the tester needs one
                if isinstance(tester, (PreferenceTester, CosineTester)):
                    if not estimators_available:
                        raise ValueError(f"Tester '{tester_name}' requires an estimator, but none were loaded or available.")
                elif isinstance(tester, ImageGenerationTester):
                     # ImageGenerationTester needs scorer_model OR estimator if use_estimator=True
                     if tester.use_estimator and not estimators_available:
                         raise ValueError(f"Tester '{tester_name}' is configured with use_estimator=True, but no estimator is available.")
                     # Check if *any* scorer model is available if needed
                     scorer_models_available = self._scorer_models and any(sm is not None for sm in self._scorer_models)
                     if not tester.use_estimator and not scorer_models_available:
                          raise ValueError(f"Tester '{tester_name}' is configured to use the scorer_model, but none are available.")
                     if self.embedder is None: # Also needs embedder
                          raise ValueError(f"Tester '{tester_name}' requires an embedder, but it's not available.")
                # Add checks for other testers if they have specific requirements
        else:
             print("Skipping tester validation in 'train_human_feedback' mode.")


        # Validate Savers (Common checks for all modes)
        for saver in self.savers:
            saver_name = type(saver).__name__
            if isinstance(saver, LearnedEstimatorSaver):
                # Check if *any* estimator is available if LearnedEstimatorSaver is used (will be skipped later if num_models > 1)
                if not estimators_available:
                    raise ValueError(f"Saver '{saver_name}' requires an estimator, but none were loaded or available.")
            elif isinstance(saver, (VisitsSaver, VisitsImageSaver, ReadableVisitsSaver)):
                # More detailed check for visits availability (remains the same)
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
                    # Use the correct variable name 'estimators_available'
                    if self.cfg.get('test_only', False) and not estimators_available and self.cfg.get('visits_path'):
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
        results.add_metadata('horizon', self.env.max_episode_length) # Log horizon from env
        results.add_metadata('algorithm', self.cfg.algorithm)
        results.add_metadata('base_prompt', self.cfg.base_prompt)
        # Add embedder info to metadata
        results.add_metadata('embedder_class', self.embedder.__class__.__name__)
        results.add_metadata('embedder_model_id', self.embedder.model_id)
        results.add_metadata('embedder_normalize', self.embedder.normalize)
        # Add scorer model names to metadata
        results.add_metadata('scorer_model_names', self.scorer_model_names)

        # Add the resolved config as a plain dictionary for the ConfSaver
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        results.add_metadata('config_dict', config_dict)

        # --- Run Testers and Collect Metrics (Looping through models) ---
        # Initialize lists to store metrics across all models
        all_preference_errors = []
        all_cosine_errors = []
        # Add lists for other potential metrics here
        # ...

        if current_mode != "train_human_feedback":
            print("\n--- Running Testers for Each Model ---")
            for i in range(self.num_scorer_models):
                model_name = self.scorer_model_names[i]
                estimator = self.estimators[i] if self.estimators and i < len(self.estimators) else None
                theta_star = self._theta_stars[i] if self._theta_stars and i < len(self._theta_stars) else None
                scorer_model = self._scorer_models[i] if self._scorer_models and i < len(self._scorer_models) else None

                print(f"\nTesting Model: {model_name}")

                # Check if essential components for this model are available
                if theta_star is None or scorer_model is None:
                    print(f"Skipping testing for model '{model_name}': Ground truth components missing.")
                    continue
                # Note: Estimator might be None if fitting failed, testers should handle this

                for tester in self.testers:
                    tester_name = type(tester).__name__
                    print(f"  Running tester: {tester_name}")

                    # Check if this tester requires an estimator and if it's available for *this* model
                    if isinstance(tester, (PreferenceTester, CosineTester)) and estimator is None:
                        print(f"  Skipping {tester_name} for model '{model_name}': Estimator not available.")
                        continue
                    if isinstance(tester, ImageGenerationTester) and tester.use_estimator and estimator is None:
                         print(f"  Skipping {tester_name} (use_estimator=True) for model '{model_name}': Estimator not available.")
                         continue

                    # --- Skip ImageGenerationTester for subsequent models (i > 0) ---
                    if isinstance(tester, ImageGenerationTester) and i > 0:
                        print(f"  Skipping {tester_name} for model '{model_name}' (run only for the first model).")
                        continue
                    # -------------------------------------------------------------

                    try:
                        # Run the test with the components for the current model
                        tester_results = tester.run_test(
                           cfg=self.cfg,
                           env=self.env,
                           estimator=estimator, # Pass the specific estimator for this model
                           theta_star=theta_star, # Pass the specific theta_star for this model
                           scorer_model=scorer_model, # RE-ADDED - Pass the specific scorer_model for this iteration
                           training_words_list=self.training_words,
                           testing_words_list=self.testing_words
                       )

                        # Process and store results for this model
                        if tester_results:
                            for key, value in tester_results.items():
                                print(f"    Model '{model_name}' - {key}: {value:.4f}")
                                # Append to corresponding list
                                if key == "preference_error":
                                    all_preference_errors.append(value)
                                elif key == "cosine_error":
                                    all_cosine_errors.append(value)
                                # Add elif for other metrics...

                    except Exception as e:
                        print(f"  Error running tester {tester_name} for model '{model_name}': {e}")

            print("--------------------------------------\n")
        else:
            print("Skipping testers in 'train_human_feedback' mode.")

        # --- Calculate Averaged Metrics ---
        averaged_metrics = {}
        if all_preference_errors:
            avg_pref_error = np.mean(all_preference_errors)
            averaged_metrics["preference_error"] = avg_pref_error
            print(f"Average Preference Error across models: {avg_pref_error:.4f}")
        if all_cosine_errors:
            avg_cosine_error = np.mean(all_cosine_errors)
            averaged_metrics["cosine_error"] = avg_cosine_error
            print(f"Average Cosine Error across models: {avg_cosine_error:.4f}")
        # Calculate averages for other metrics...

        # Add averaged metrics to the results container
        if averaged_metrics:
            results.add_metrics(averaged_metrics)

        # --- Run Savers ---
        print("\n--- Running Savers ---")
        for saver in self.savers:
            saver_name = type(saver).__name__

            # Skip LearnedEstimatorSaver if multiple models were used
            if isinstance(saver, LearnedEstimatorSaver) and self.num_scorer_models > 1:
                print(f"Skipping saver: {saver_name} (multiple scorer models configured)")
                continue

            # Pass the results object containing potentially loaded estimators/visits
            print(f"Running saver: {saver_name}")
            try:
                saver.save_result(results)
            except Exception as e:
                 print(f"Error running saver {saver_name}: {e}")
                 # Decide if we should continue or stop? For now, continue.

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
        
        # REMOVED Horizon Check: Horizon is now implicitly len(vocab_files)
        # if len(vocab_files) != self.cfg.horizon:
        #     raise ValueError(f"Number of vocabulary files ({len(vocab_files)}) must match horizon ({self.cfg.horizon})")
        
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
