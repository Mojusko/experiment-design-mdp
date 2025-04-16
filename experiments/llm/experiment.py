import os
import torch
import numpy as np
import datetime
import sys

import hydra
from omegaconf import DictConfig, OmegaConf # Added OmegaConf

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
# Import specific saver types needed for validation
from components.saver   import BaseSaver, VisitsSaver, VisitsImageSaver, ConfSaver



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
        self.explorer = SolverFactory.create(cfg, self.env, self.design, self.feedback)

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
                     # Pass env and embedder explicitly if needed by the tester
                     # Hydra handles args defined in t_conf (like 'params')
                     init_args = {
                         'scorer_model': self._scorer_model,
                         'embedder': self.embedder,
                         'env': self.env # Pass env, needed by VisitsTester
                         # 'params' is handled by Hydra via t_conf
                     }
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
                    init_args['dense_feedback'] = self.cfg.get('dense_feedback', False)
                    init_args['verbose'] = self.cfg.get('verbose', False)

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
        self.visits = results

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
        # Store scorer vector if needed (optional, depends on usage)
        # self.env._scorer_vector = self._scorer_model.weight

        # Build scorer model using the environment and the embedder
        self._scorer_model = get_scorer_model(
            model_name=self.cfg.experiment.scorer_model,
            env=env,
            embedder=self.embedder # Pass embedder
        )

        # Create the ground truth function using the scorer model
        self._theta_star = make_theta_star(env, self._scorer_model, verbose=self.cfg.verbose)



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
    
    def run_test_only(self, estimator_path):
        """Run only the testing and saving parts with a pre-loaded estimator
        
        Args:
            estimator_path: Path to the saved estimator file
            
        Returns:
            True if successful, False otherwise
        """
        # Use original results directory if not explicitly specified
        if not self.cfg.get('override_results_dir', False):
            # Extract original results directory from estimator path
            original_dir = os.path.dirname(estimator_path)
            if os.path.exists(original_dir):
                # Create a timestamp-based subdirectory under additional_tests
                timestamp = os.environ.get('TIMESTAMP', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
            
                # Get algorithm and feedback type for directory name
                algorithm = self._get_algorithm_code()
                feedback_type = self._get_feedback_code()
                experiment_id = self.experiment_id or "test"
            
                # Create directory structure: original_dir/additional_tests/test-algorithm-feedback-timestamp
                tests_base_dir = os.path.join(original_dir, "additional_tests")
                self.results_dir = os.path.join(tests_base_dir, f"test-{algorithm}-{feedback_type}-{timestamp}")
            
                print(f"Using original results directory: {original_dir}")
                print(f"Saving test results to: {self.results_dir}")
                os.makedirs(self.results_dir, exist_ok=True)
            
                # Update results_dir for all savers
                for saver in self.savers:
                    saver.results_dir = self.results_dir
                    print(f"Updated saver {type(saver).__name__} to use results_dir: {self.results_dir}")
        
        if not self.load_estimator(estimator_path):
            return False

        print("Running tests with pre-loaded estimator (skipping optimization)")

        # Empty visits if needed (will be passed to testers)
        if not hasattr(self, 'visits') or self.visits is None:
            self.visits = [] if self.cfg.feedback.num_policies == 1 else [[] for _ in range(self.cfg.feedback.num_policies)]

        # Run tests and save results
        self.test_and_save()
        return True

    def run_visits_only(self, visits_path):
        """Loads visits and runs only the testing and saving parts."""
        print(f"--- Running in Visits-Only Inspection Mode ---")
        print(f"Loading visits from: {visits_path}")
        if not os.path.exists(visits_path):
            print(f"Error: Visits file not found at {visits_path}")
            return False
        try:
            # Load visits - they will be passed to testers via test_and_save
            self.visits = torch.load(visits_path)
            print(f"Successfully loaded visits.")
            # Ensure estimator is None for this mode
            self.estimator = None
            # Use original results directory logic if not overridden
            if not self.cfg.get('override_results_dir', False):
                original_dir = os.path.dirname(visits_path)
                if os.path.exists(original_dir):
                    timestamp = os.environ.get('TIMESTAMP', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
                    algorithm = self._get_algorithm_code() # Might need adjustment based on filename
                    feedback_type = self._get_feedback_code() # Might need adjustment
                    experiment_id = self.experiment_id or "inspect"
                    tests_base_dir = os.path.join(original_dir, "additional_tests")
                    self.results_dir = os.path.join(tests_base_dir, f"inspect-{algorithm}-{feedback_type}-{timestamp}")
                    print(f"Using original results directory: {original_dir}")
                    print(f"Saving inspection results to: {self.results_dir}")
                    os.makedirs(self.results_dir, exist_ok=True)
                    for saver in self.savers:
                        saver.results_dir = self.results_dir
                        print(f"Updated saver {type(saver).__name__} to use results_dir: {self.results_dir}")

            # Run testers (like VisitsTester) and savers
            self.test_and_save()
            return True
        except Exception as e:
            print(f"Error loading or processing visits: {e}")
            return False
    def test_and_save(self):
        """Final estimation, testing and saving of results"""
        # Create a container for all results
        from components.results import ExperimentResults
        results = ExperimentResults()
        
        # Set the estimator and visits
        results.set_estimator(self.estimator)
        results.set_visits(self.visits)
        
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

        # Run all testers and collect metrics (only if estimator exists)
        if self.estimator is None:
            print("Skipping testers as estimator is None (likely explore_only or failed estimation).")
        else:
            print("Running testers...")
            for tester in self.testers:
                # Check if tester requires an estimator (most do)
                # Simple check for now: assume all testers need it unless specified otherwise
                requires_estimator = True # Default assumption
                # Example of how to add exceptions later:
                # if isinstance(tester, SomeTesterThatDoesNotNeedEstimator):
                #     requires_estimator = False

                if requires_estimator and self.estimator is None:
                     print(f"Skipping tester {type(tester).__name__} because estimator is missing.")
                     continue

                print(f"Running tester: {type(tester).__name__}")
                tester_results = tester.run_test(
                    cfg=self.cfg,
                    env=self.env,
                    estimator=self.estimator, # Can be None
                    theta_star=self._theta_star,
                    training_words_list=self.training_words,
                    testing_words_list=self.testing_words,
                    visits=self.visits # Pass the visits data
                )
                # Add metrics to results container
                if tester_results: # Ensure tester returned something
                    results.add_metrics(tester_results)

        # Use all savers to save the results (savers should handle None estimator if needed)
        print("Running savers...")
        # Removed duplicated lines causing IndentationError here
        
        # Use all savers to save the results
        for saver in self.savers:
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
