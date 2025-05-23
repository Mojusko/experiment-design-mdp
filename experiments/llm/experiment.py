import os
import re # Added import for regular expressions
import torch
import torch.nn.functional as F # Added for cosine_similarity
import numpy as np
import datetime
import sys

import hydra
from omegaconf import DictConfig, OmegaConf # Added OmegaConf
from hydra.utils import to_absolute_path # Import Hydra path utility

from stpy.helpers.helper import cartesian
# Updated imports from doexpy.env.llm
from doexpy.env.llm import (
    LLMGrid, generate_emissions, create_prompt # Removed get_scorer_model, make_theta_star
)
# Import scorer model functions from their new location
from experiments.llm.models.scorer_models import get_scorer_model, make_theta_star
# Import embedder components
from components.embedder import BaseEmbedder, create_embedder
from components.feedback import FeedbackFactory
from components.solver import SolverFactory
from components.tester import BaseTester, ImageGenerationTester
# Import specific saver types needed for validation and estimator creation
# Removed LearnedEstimatorSaver from direct import here, will handle skipping later
from components.saver import BaseSaver, VisitsSaver, VisitsImageSaver, ConfSaver, MetricsSaver # Added MetricsSaver
# Removed unused estimator/likelihood/regularizer imports here, they are used within factories/components
# from stpy.regression.regularized_dictionary.regularized_multinomial_estimator import RegularizedMultinomialEstimator
# from stpy.probability.multinomial_likelihood import MultinomialLikelihood
# from stpy.regularization.regularizer import L2Regularizer
from stpy.embeddings.polynomial_embedding import CustomEmbedding # Corrected import path
# Removed unused json and re imports
from doexpy.functionals.doe_adaptive_functionals import AdaptiveOrigDesignC # Added import
from doexpy.mdpexplore import MdpExplore, MdpExploreMultiPolicy # Added for type checking
from components.solver import TwoPhaseExplorer # Added for type checking
import logging # Added import

logger = logging.getLogger(__name__) # Added logger instance

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
        elif scorer_model_config is None:
            print("No scorer_model configured. Will proceed without ground truth scorer.")
            self.scorer_model_names = []
        else:
            raise ValueError(f"scorer_model in config must be a string, a list, or None, got {type(scorer_model_config)}")
        self.num_scorer_models = len(self.scorer_model_names)

        # --- Load Data & Initialize Environment ---
        self.training_words, self.testing_words, self.model_words = self._load_data()
        # _init_env now uses self.embedder and populates _scorer_models and _theta_stars
        # self._scorer_models and self._theta_stars will be empty if self.scorer_model_names is empty
        self.env = self._init_env() 

        # --- Initialize Core Components (Lists for multiple models) ---
        self.feedbacks = []
        self.designs = []
        self.estimators = [] # Will hold estimator instances

        if self.num_scorer_models > 0:
            # Create feedback, design, and estimator for each configured scorer model
            for i, model_name in enumerate(self.scorer_model_names):
                print(f"Initializing components for scorer model: {model_name}")
                feedback, design, estimator = FeedbackFactory.create(
                    cfg,
                    self.env,
                    self._scorer_models[i], # Pass the specific model instance
                    self.embedder,
                    scorer_model_name=model_name
                )
                self.feedbacks.append(feedback)
                self.designs.append(design)
                self.estimators.append(estimator)
        else:
            # No scorer models configured (e.g., for human feedback training without GT comparison)
            # Initialize a single set of components without a specific scorer model
            print("Initializing a single set of feedback/design/estimator components (no scorer model).")
            feedback, design, estimator = FeedbackFactory.create(
                cfg,
                self.env,
                None, # No scorer model instance
                self.embedder,
                scorer_model_name=None # No specific model name
            )
            self.feedbacks.append(feedback)
            self.designs.append(design)
            self.estimators.append(estimator)

        # --- Initialize Solver ---
        # The explorer uses the design and feedback from the *first* available set of components.
        if not self.designs or not self.feedbacks:
            raise RuntimeError("Cannot initialize explorer: No design or feedback components were created.")
        
        first_model_name_for_explorer = self.scorer_model_names[0] if self.scorer_model_names else "N/A (no scorer model)"
        print(f"Initializing explorer using components (design[0], feedback[0]). Associated model (if any): {first_model_name_for_explorer}")
        self.explorer = SolverFactory.create(
            cfg,
            self.env,
            self.designs[0], 
            self.feedbacks[0], 
            same_first_action_in_episode=cfg.get('same_first_action_in_episode', False)
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

        self.experiment_id = experiment_id # Initial experiment_id (e.g., from Makefile)
        self.previous_theta_fits = [None] * self.num_scorer_models # For logging previous estimator performance

        # --- Override experiment_id and algorithm if in test/inspect mode ---
        if self.cfg.get('test_only', False):
            input_path = self.cfg.get('visits_path') or self.cfg.get('estimator_path')
            if input_path:
                try:
                    from hydra.utils import to_absolute_path
                    abs_input_path = to_absolute_path(input_path)
                    filename = os.path.basename(abs_input_path)
                    print(f"Attempting to parse original info from input filename: {filename}")

                    # Regex to capture prefix, alg, feed, optional episode, and seed from visits/estimator files
                    # Example: visits-feedback-rand-mult-ep30-1.pkl
                    # Example: visits-feedback-dsn-num-5.pkl (no episode part)
                    # Example: estimator-feedback-dsn-num-ep50-5.pt
                    pattern = re.compile(r"^(?:visits|estimator)-(.+?)-(\w+)-(\w+)(?:-ep(\d+))?-(\d+)\.(?:pkl|pt)$")
                    match = pattern.match(filename)

                    if match:
                        original_prefix_part = match.group(1) # e.g., "feedback"
                        original_alg_code = match.group(2)    # e.g., "rand"
                        original_feed_code = match.group(3)   # e.g., "mult"
                        original_episodes = match.group(4)    # e.g., "30" or None
                        original_seed = match.group(5)        # e.g., "1"

                        # Reconstruct the original experiment ID
                        parsed_original_id = f"{original_prefix_part}-{original_alg_code}-{original_feed_code}"
                        if original_episodes:
                            parsed_original_id += f"-ep{original_episodes}"
                        parsed_original_id += f"-{original_seed}"

                        # Map alg_code back to algorithm name
                        alg_map = {"dsn": "design", "rand": "random", "opt": "optim"}
                        parsed_original_id = f"{original_prefix_part}-{original_alg_code}-{original_feed_code}"
                        if original_episodes:
                            parsed_original_id += f"-ep{original_episodes}"
                        parsed_original_id += f"-{original_seed}"
                        
                        print(f"  Parsed Original ID from filename: {parsed_original_id}")
                        print(f"  Overriding self.experiment_id from '{self.experiment_id}' to '{parsed_original_id}'")
                        self.experiment_id = parsed_original_id # Always update if filename parsed

                        # Map alg_code back to algorithm name
                        alg_map = {"dsn": "design", "rand": "random", "opt": "optim"}
                        parsed_original_algorithm = alg_map.get(original_alg_code)

                        if parsed_original_algorithm:
                            print(f"  Parsed Original Algorithm: {parsed_original_algorithm} (from code '{original_alg_code}')")
                            print(f"  Overriding self.cfg.algorithm from '{self.cfg.algorithm}' to '{parsed_original_algorithm}'")
                            self.cfg.algorithm = parsed_original_algorithm
                        else:
                            print(f"  Warning: Could not map parsed algorithm code '{original_alg_code}' to a known algorithm name.")
                            print(f"  Setting self.cfg.algorithm to 'unknown'. Original value was '{self.cfg.algorithm}'.")
                            self.cfg.algorithm = "unknown" # Set to "unknown" if not mapped
                    else:
                        print(f"  Warning: Could not parse original ID and algorithm from filename '{filename}' using pattern.")
                except Exception as e:
                    print(f"  Warning: Error during parsing of input path '{input_path}': {e}")
        # -----------------------------------------------------------------

        # Initialize testers and savers with potentially overridden results_dir and experiment_id
        testers_config = self.cfg.get('tester') # Get the config value (could be list or None)
        self.testers = []
        if testers_config:
             for t_conf in testers_config:
                 try:
                     # Instantiate tester, passing core objects explicitly.
                     # Hydra handles 'params' from t_conf automatically.
                     tester = hydra.utils.instantiate(
                         t_conf,          # The tester's specific config
                         env=self.env,          # Pass env explicitly
                         embedder=self.embedder # Pass embedder explicitly
                         # No other common args needed based on current BaseTester/subclasses
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

                # Instantiate saver, passing core objects and common config explicitly.
                # Hydra handles 'params' from s_conf automatically.
                # Conditional logic for specific saver types is removed.
                saver = hydra.utils.instantiate(
                    s_conf,             # The saver's specific config (_target_, params, etc.)
                    env=self.env,             # Pass env explicitly
                    embedder=self.embedder,     # Pass embedder explicitly
                    results_dir=self.results_dir, # Pass results_dir explicitly
                    experiment_id=self.experiment_id, # Pass experiment_id explicitly
                    seed=self.seed,           # Pass seed explicitly
                    total_repeats=self.cfg.experiment.get('repeats', 1), # Pass total_repeats explicitly
                    algorithm=self.cfg.algorithm, # Pass algorithm explicitly
                    num_policies=self.cfg.feedback.num_policies # Pass num_policies for savers that need it
                    # scorer_model is not passed here
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
        if not isinstance(total_episodes, int) or total_episodes <= 0:
            raise ValueError(f"LLMExperiment.run() requires cfg.experiment.episodes to be a positive integer, but got: {total_episodes}")
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
                    
                    # Print theta_fit id before fitting
                    if self.estimators[i] and hasattr(self.estimators[i], 'theta_fit') and self.estimators[i].theta_fit is not None:
                        print(f"  Model '{model_name}': Estimator theta_fit id BEFORE fit: {id(self.estimators[i].theta_fit)}")
                    else:
                        print(f"  Model '{model_name}': Estimator theta_fit BEFORE fit: N/A (no estimator or theta_fit yet)")

                    print(f"Fitting estimator for model: {model_name}")
                    feedback.fit_estimator()
                    self.estimators[i] = feedback.estimator # Update the specific estimator

                    # Print theta_fit id after fitting - REMOVED THIS PRINT BLOCK
                    # if self.estimators[i] and hasattr(self.estimators[i], 'theta_fit') and self.estimators[i].theta_fit is not None:
                    #     print(f"  Model '{model_name}': Estimator theta_fit id AFTER fit: {id(self.estimators[i].theta_fit)}")
                    # else:
                    #     print(f"  Model '{model_name}': Estimator theta_fit AFTER fit: N/A (no estimator or theta_fit)")

                    # Calculate and print cosine error for this model
                    # Check if theta can be retrieved from feedback and if scorer_model has weight
                    current_est_weight = self.feedbacks[i].get_learned_theta()
                    if current_est_weight is not None and hasattr(scorer_model, 'weight'):
                        current_gt_weight = scorer_model.weight # Renamed for clarity within this block
                        current_estimator_error = self.calculate_cosine_error(current_est_weight, current_gt_weight)
                        # current_l2_norm = torch.linalg.norm(current_est_weight).item() # Keep calculation for potential future use

                        log_msg_parts = [
                            f"Episode {ep_idx + 1} partial re-fit for model '{model_name}':",
                            f"Current Estimator Cosine Error: {current_estimator_error:.4f}" # Removed L2 Norm from print
                        ]

                        # Previous estimator's performance (if available)
                        previous_theta = self.previous_theta_fits[i]
                        if previous_theta is not None:
                            try:
                                previous_estimator_error = self.calculate_cosine_error(previous_theta, current_gt_weight)
                                log_msg_parts.append(f"Previous Estimator Cosine Error: {previous_estimator_error:.4f}")
                            except Exception as e:
                                logger.warning(f"Could not calculate Previous Estimator cosine error for model '{model_name}': {e}")
                                log_msg_parts.append("Previous Estimator Cosine Error: N/A")
                        else:
                            log_msg_parts.append("Previous Estimator Cosine Error: N/A (first fit or not available)")
                        
                        # Update design if applicable (using the current estimator object from feedback)
                        current_design = self.designs[i]
                        if isinstance(current_design, AdaptiveOrigDesignC):
                            if self.num_scorer_models > 1:
                                raise ValueError(
                                    "Adaptive C-optimal design with estimator updates (adaptive_estimation_frequency > 0) "
                                    "is not supported when multiple scorer models are configured."
                                )
                            # The design update uses self.feedbacks[i].estimator
                            current_design.update_estimator(self.feedbacks[i].estimator, self.env.emissions)
                            # No C-vector specific logging added to log_msg_parts here.
                        
                        elif est_freq > 0: # Log if adaptive estimation is on but design is not AdaptiveOrigDesignC
                             logger.info(f"Adaptive estimation is active for model '{model_name}', but design {type(current_design).__name__} is not AdaptiveOrigDesignC. Design's C vector not updated from estimator.")
                        
                        print(" ".join(log_msg_parts))

                    else:
                        # This else corresponds to: if current_est_weight is not None and hasattr(scorer_model, 'weight'):
                        print(f"Could not calculate cosine error for model '{model_name}' (learned theta or ground truth weight missing). Design's C not updated.")
                    
                    # Store the current theta_fit as the "previous" for the next adaptive step for this model
                    # current_est_weight already holds the learned theta or None
                    if current_est_weight is not None:
                        self.previous_theta_fits[i] = current_est_weight.detach().clone()
                    else:
                        self.previous_theta_fits[i] = None

                # Clear the buffer after processing all models for this frequency step
                print("Clearing recent visits buffer.")
                for p_i in range(num_policies):
                    recent_visits_buffer[p_i].clear()
                print("-----------------------------------------------------\n")

        run_kwargs = {
            'episodes': total_episodes,
            'return_visitations': True
        }

        if isinstance(self.explorer, MdpExploreMultiPolicy):
            run_kwargs['update_callback'] = update_callback
            run_kwargs['start_ep_idx'] = 0 # Defaulting to 0 as it wasn't passed before
        elif isinstance(self.explorer, TwoPhaseExplorer):
            run_kwargs['update_callback'] = update_callback
        # For MdpExplore, no additional arguments beyond episodes and return_visitations are passed.
            
        results = self.explorer.run(**run_kwargs)
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
        # This variable will hold data in the format List[List[Tuple(s,a)]]
        visits_for_final_feedback_collection = None 

        if isinstance(self.explorer, MdpExplore): # MdpExplore does not use update_callback
            print("Using self.visits for final estimation (MdpExplore was used).")
            # self.visits is List[Tuple(s,a)]. Wrap it to List[List[Tuple(s,a)]]
            # Ensure self.visits is not None before wrapping
            visits_for_final_feedback_collection = [self.visits] if self.visits is not None else [[]]
        elif any(len(buf) > 0 for buf in recent_visits_buffer): # Check if MdpExploreMultiPolicy callback populated buffer
            print("Using remaining recent visits buffer for final estimation.")
            visits_for_final_feedback_collection = recent_visits_buffer
        elif est_start == 0: # Fallback for MdpExploreMultiPolicy if buffer is empty but no adaptive est ran
            print("Using all_visits for final estimation (adaptive estimation start was 0).")
            visits_for_final_feedback_collection = all_visits
        # If none of the above, visits_for_final_feedback_collection remains None (no new data to process)


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

            # Collect labels only if there are new visits to process for this model
            # Check outer list, then index i, then if policy i's visit list is non-empty
            if visits_for_final_feedback_collection and \
               i < len(visits_for_final_feedback_collection) and \
               visits_for_final_feedback_collection[i]: 
                print(f"Collecting final labels for model: {model_name}")
                # feedback.collect_labels expects List[List[Tuple(s,a)]]
                feedback.collect_labels(self.cfg, visits_for_final_feedback_collection, theta_star)
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
            current_est_weight = self.feedbacks[i].get_learned_theta() # This is the estimator after the final fit
            if current_est_weight is not None and hasattr(scorer_model, 'weight'):
                gt_weight = scorer_model.weight
                current_estimator_error = self.calculate_cosine_error(current_est_weight, gt_weight)
                
                log_msg_parts = [
                    f"Final estimation for model '{model_name}' after {total_episodes} episodes complete:",
                    f"Current Estimator Cosine Error: {current_estimator_error:.4f}"
                ]

                # Previous estimator's performance (from the last adaptive step)
                previous_theta = self.previous_theta_fits[i]
                if previous_theta is not None:
                    try:
                        previous_estimator_error = self.calculate_cosine_error(previous_theta, gt_weight)
                        log_msg_parts.append(f"Previous Estimator Cosine Error: {previous_estimator_error:.4f}")
                    except Exception as e:
                        logger.warning(f"Could not calculate Previous Estimator cosine error for final log (model '{model_name}'): {e}")
                        log_msg_parts.append("Previous Estimator Cosine Error: N/A")
                else:
                    # This case might occur if adaptive estimation never ran (e.g., est_freq=0 or est_start >= total_episodes)
                    log_msg_parts.append("Previous Estimator Cosine Error: N/A (no prior adaptive fit)")
                
                print(" ".join(log_msg_parts))

            else:
                print(f"Could not calculate final cosine error for model '{model_name}' (estimator or ground truth weight missing).")
        print("------------------------------\n")

    # Removed _log_c_vector_similarity_vs_gt helper method

    def run_explore_only(self):
        """Runs only the exploration phase and saves visits/images."""
        total_episodes = self.cfg.experiment.episodes
        if not isinstance(total_episodes, int) or total_episodes <= 0:
            raise ValueError(f"LLMExperiment.run_explore_only() requires cfg.experiment.episodes to be a positive integer, but got: {total_episodes}")
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
            verbose=self.cfg.verbose,
            rng=self.rng # Pass the experiment's rng to LLMGrid
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

        # --- Access the 'preferences' list ---
        preferences_list = feedback_data.get("preferences")
        if not preferences_list or not isinstance(preferences_list, list):
            print("Error: 'preferences' key not found or is not a list in feedback data.")
            return None, None
        # -------------------------------------------------

        # Iterate through the items in the preferences list
        for preference_entry in preferences_list:
            if not isinstance(preference_entry, dict):
                print(f"Warning: Skipping invalid preference entry (not a dict): {preference_entry}")
                skipped_count += 1
                continue

            image_filename = preference_entry.get("filename")
            preferred_policy_idx_1based = preference_entry.get("preference")

            if image_filename is None or preferred_policy_idx_1based is None:
                print(f"Warning: Skipping preference entry with missing 'filename' or 'preference': {preference_entry}")
                skipped_count += 1
                continue
            
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

        # --- Mode 3: Inspect Visits ---
        elif mode == "inspect_visits":
            if not visits_path or not os.path.exists(visits_path):
                print(f"Error: Visits file not found or not provided for inspection: {visits_path}")
                return False
            print(f"Loading visits for inspection from: {visits_path}")
            # Load visits directly into self.visits
            try:
                self.visits = torch.load(visits_path)
                if not isinstance(self.visits, list) or not self.visits or not self.visits[0]:
                     print(f"Warning: Loaded visits from {visits_path} appear empty or invalid.")
                     # Proceed, test_and_save validation will handle it if savers need visits
            except Exception as e:
                 print(f"Error loading visits from {visits_path}: {e}")
                 return False # Stop execution on loading errors

            # Ensure estimator is None for inspection mode
            self.estimators = [None] * self.num_scorer_models
            print("Skipping estimator training/loading in inspection mode.")
            # Proceed directly to saving (which includes VisitsImageSaver)
            self.test_and_save(current_mode=mode)
            return True

        # --- Mode 4: Load Estimator and Feedback ---
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
        
        # --- Pre-tester setup for modes that might need feedback_path or visits_path ---
        # This block runs for 'load_estimator_and_feedback' and if HumanFeedbackBenchmarkTester is active.
        from components.tester import HumanFeedbackBenchmarkTester # Import locally for isinstance check
        is_hf_benchmark_tester_active = any(isinstance(t, HumanFeedbackBenchmarkTester) for t in self.testers)

        if mode == "load_estimator_and_feedback" or is_hf_benchmark_tester_active:
            if not feedback_path or not os.path.exists(to_absolute_path(feedback_path)):
                print(f"Error: Feedback file not found or not provided ({feedback_path}), but required for mode '{mode}' or HumanFeedbackBenchmarkTester.")
                return False
            
            abs_feedback_path = to_absolute_path(feedback_path)
            print(f"Loading feedback data from: {abs_feedback_path} for mode '{mode}' or HumanFeedbackBenchmarkTester.")
            try:
                import json
                with open(abs_feedback_path, 'r') as f:
                    loaded_feedback_json = json.load(f)
                
                user_prompt_from_feedback = loaded_feedback_json.get("user_prompt")
                if user_prompt_from_feedback is not None and isinstance(user_prompt_from_feedback, str):
                    print(f"  Updating env.base_prompt to '{user_prompt_from_feedback}' from feedback file.")
                    self.env.base_prompt = user_prompt_from_feedback
                    # Note: If include_base_prompt_in_first_tokens was true during LLMGrid init,
                    # this change alone might not be enough if vocab/emissions depend on it.
                    # However, config_inference.yaml has include_base_prompt_in_first_tokens: false.
                else:
                    print(f"  'user_prompt' not found in {abs_feedback_path} or not a string. Using existing env.base_prompt: '{self.env.base_prompt}'")
                
                # Store the full feedback_data if other components might need it (HFBenchmarkTester loads it itself)
                # self.feedback_data = loaded_feedback_json 

            except Exception as e:
                print(f"Error loading feedback data from {abs_feedback_path} or updating env: {e}")
                return False

        if is_hf_benchmark_tester_active: # Also load visits if HFBenchmarkTester is active
            if not visits_path or not os.path.exists(to_absolute_path(visits_path)):
                print(f"Error: Visits file not found or not provided ({visits_path}), but required for HumanFeedbackBenchmarkTester.")
                return False
            abs_visits_path = to_absolute_path(visits_path)
            print(f"Loading visits from: {abs_visits_path} for HumanFeedbackBenchmarkTester.")
            try:
                self.visits = torch.load(abs_visits_path) # Load into self.visits
                if not isinstance(self.visits, list) or not self.visits or \
                   not (isinstance(self.visits[0], list) and self.visits[0]) or \
                   not (isinstance(self.visits[0][0], tuple)): # Check structure List[List[Tuple(s,a)]]
                    print(f"Warning: Loaded visits from {abs_visits_path} appear empty or invalid for HumanFeedbackBenchmarkTester. Expected List[List[Tuple(s,a)]].")
                    # Allow proceeding, tester will handle it or error more specifically.
            except Exception as e:
                print(f"Error loading visits from {abs_visits_path} for HumanFeedbackBenchmarkTester: {e}")
                return False
        # --- End pre-tester setup ---


        # --- Mode 5: Train Human Feedback ---
        elif mode == "train_human_feedback":
            if not visits_path or not os.path.exists(visits_path): # Path existence checked by to_absolute_path earlier if HFBenchmarkTester active
                abs_visits_path = to_absolute_path(visits_path) if visits_path else "None"
                if not os.path.exists(abs_visits_path): # Re-check if not loaded above
                    print(f"Error: Visits file not found or not provided: {abs_visits_path}")
                    return False
            if not feedback_path or not os.path.exists(to_absolute_path(feedback_path)):
                abs_feedback_path = to_absolute_path(feedback_path) if feedback_path else "None"
                if not os.path.exists(abs_feedback_path): # Re-check
                    print(f"Error: Feedback file not found or not provided: {abs_feedback_path}")
                    return False
            
            # Paths are now absolute
            abs_visits_path = to_absolute_path(visits_path)
            abs_feedback_path = to_absolute_path(feedback_path)


            print(f"Loading visits from: {abs_visits_path}")
            if not feedback_path or not os.path.exists(feedback_path):
                print(f"Error: Feedback file not found or not provided: {feedback_path}")
                return False

            print(f"Loading visits from: {visits_path}")
            try:
                visits_data = torch.load(visits_path)
                if not isinstance(visits_data, list) or not visits_data: # Basic check
                     print(f"Warning: Loaded visits from {visits_path} appear empty or invalid.")
                     # Allow proceeding, _process_human_feedback will handle empty/invalid visits
            except Exception as e:
                 print(f"Error loading visits from {visits_path}: {e}")
                 return False

            print(f"Loading human feedback data from: {feedback_path}")
            try:
                import json # Ensure json is imported
                with open(feedback_path, 'r') as f:
                    self.feedback_data = json.load(f) # Store loaded feedback data
                if not self.feedback_data:
                    print("Error: Loaded feedback data is empty.")
                    return False
            except Exception as e:
                print(f"Error loading feedback data from {feedback_path}: {e}")
                return False

            print("Processing human feedback to generate training data...")
            comparison_embeddings, labels = self._process_human_feedback(visits_data, self.feedback_data)

            if comparison_embeddings is None or labels is None:
                print("Error: Failed to process human feedback into training data. Skipping estimator fitting.")
                return False
            
            if not self.feedbacks or self.feedbacks[0] is None:
                print("Error: Feedback component (self.feedbacks[0]) not initialized. Cannot fit estimator.")
                return False

            print("Fitting estimator using processed human feedback...")
            try:
                # Populate _collected_data for the first feedback component
                self.feedbacks[0]._collected_data = [(comparison_embeddings, labels)]
                self.feedbacks[0].fit_estimator()
                # Update the main estimator list
                self.estimators[0] = self.feedbacks[0].estimator
                print("Estimator fitted successfully with human feedback.")
            except Exception as e:
                print(f"Error fitting estimator with human feedback: {e}")
                return False
            
            # Proceed to testing and saving (which includes benchmark evaluation)
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

        # Set the list of estimators, feedbacks and visits
        results.set_estimators(self.estimators) # Use the list of estimators
        results.set_feedbacks(self.feedbacks)   # Set the list of feedback objects
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
                     # ImageGenerationTester needs an estimator if prompt_ranking_model is not 'gt'
                     if tester.prompt_ranking_model != 'gt' and not estimators_available:
                         raise ValueError(f"Tester '{tester_name}' is configured with prompt_ranking_model='{tester.prompt_ranking_model}', but no estimator is available.")
                     # Check if *any* scorer model is available if prompt_ranking_model is 'gt'
                     scorer_models_available = self._scorer_models and any(sm is not None for sm in self._scorer_models)
                     if tester.prompt_ranking_model == 'gt' and not scorer_models_available:
                          raise ValueError(f"Tester '{tester_name}' is configured with prompt_ranking_model='gt', but no ground truth scorer models are available.")
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
        # Add all GT scorer models to metadata for ImageGenerationSaver
        results.add_metadata('all_gt_scorer_models', self._scorer_models)


        # Add the resolved config as a plain dictionary for the ConfSaver
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        results.add_metadata('config_dict', config_dict)

        # --- Run Testers and Collect Metrics (Looping through models) ---
        # Dictionary to store metrics per model for individual saving
        per_model_metrics_collection = {}
        # Averaging lists (all_preference_errors, all_cosine_errors) are removed.
        # Add lists for other potential metrics here if they need different handling
        # ...

        # Skip testers if in inspection or human feedback training mode
        if current_mode == "inspect_visits":
            print("Skipping testers in 'inspect_visits' mode.")
        elif current_mode == "train_human_feedback":
            print("Skipping testers in 'train_human_feedback' mode.")
        else:
            # Run testers only for other modes (full_run, load_estimator, estimate_from_visits, load_estimator_and_feedback)
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
                    # For ImageGenerationTester, check if it needs an estimator (prompt_ranking_model != 'gt') and if that estimator is available
                    if isinstance(tester, ImageGenerationTester) and tester.prompt_ranking_model != 'gt' and estimator is None:
                         print(f"  Skipping {tester_name} (prompt_ranking_model='{tester.prompt_ranking_model}') for model '{model_name}': Estimator not available.")
                         continue

                    # --- Skip ImageGenerationTester for subsequent models (i > 0) ---
                    if isinstance(tester, ImageGenerationTester) and i > 0:
                        print(f"  Skipping {tester_name} for model '{model_name}' (run only for the first model).")
                        continue
                    # -------------------------------------------------------------

                    try:
                        # Prepare arguments for tester.run_test()
                        test_args = {
                            'cfg': self.cfg,
                            'env': self.env,
                            'estimator': estimator, # Pass the specific estimator for this model
                            'theta_star': theta_star, # Pass the specific theta_star for this model
                            'scorer_model': scorer_model, # Pass the specific scorer_model
                            'training_words_list': self.training_words,
                            'testing_words_list': self.testing_words,
                            'visits': self.visits if hasattr(self, 'visits') else None # Pass loaded visits
                        }
                        # Add feedback object if the tester is CosineTester
                        if isinstance(tester, CosineTester):
                            test_args['feedback'] = self.feedbacks[i]
                        
                        # For ImageGenerationTester, pass all estimators and GT models
                        if isinstance(tester, ImageGenerationTester):
                            test_args['all_estimators'] = self.estimators # Pass the full list
                            test_args['all_gt_scorer_models'] = self._scorer_models # Pass the full list

                        tester_results = tester.run_test(**test_args)

                        # Process and store results for this model
                        if tester_results:
                            for key, value in tester_results.items():
                                if key == "image_generation" and isinstance(value, dict):
                                    # Summarize image_generation results
                                    best_count = len(value.get("best_prompts", []))
                                    worst_count = len(value.get("worst_prompts", []))
                                    print(f"    Model '{model_name}' - {key}: (Best: {best_count}, Worst: {worst_count})")
                                elif isinstance(value, float):
                                    print(f"    Model '{model_name}' - {key}: {value:.4f}")
                                else:
                                    # For other non-float values, print as is (or consider summarizing if too verbose)
                                    print(f"    Model '{model_name}' - {key}: {value}")

                                # If the current tester is ImageGenerationTester,
                                # add its results directly to the main results object's metrics.
                                # These are not averaged across models as IGT runs only for the first model.
                                if isinstance(tester, ImageGenerationTester):
                                    results.add_metric(key, value)
                                # For other testers, collect for per-model saving
                                else:
                                    if model_name not in per_model_metrics_collection:
                                        per_model_metrics_collection[model_name] = {}
                                    per_model_metrics_collection[model_name][key] = value

                    except Exception as e:
                        print(f"  Error running tester {tester_name} for model '{model_name}': {e}")
                        # Optionally, re-raise or log traceback for critical errors
                        # import traceback
                        # print(traceback.format_exc())

            print("--------------------------------------\n")
        # Removed redundant else block here

        # --- Averaged Metrics Calculation Removed ---
        # The main results.metrics will now only contain non-model-specific metrics
        # (e.g., from ImageGenerationTester if it runs for the first model).

        # --- Benchmark Episode Evaluation (for human feedback training mode) ---
        # This is now handled by HumanFeedbackBenchmarkTester if it's active in the config.
        # The results from that tester will be added to results.metrics directly by the tester loop.
        # No specific code needed here for train_human_feedback mode regarding benchmark evaluation.
        # --------------------------------------------------------------------

        # If there's only one model, merge its metrics from per_model_metrics_collection
        # into the main results.metrics so MetricsSaver can pick them up.
        if self.num_scorer_models == 1 and per_model_metrics_collection:
            single_model_name = self.scorer_model_names[0]
            if single_model_name in per_model_metrics_collection:
                print(f"Merging metrics from single model '{single_model_name}' into main results for saving.")
                for key, value in per_model_metrics_collection[single_model_name].items():
                    results.add_metric(key, value)

        # --- Run Savers ---
        print("\n--- Running Savers ---")
        
        # Find the MetricsSaver instance first to handle per-model metric saving
        metrics_saver_instance = None
        for saver_instance in self.savers:
            if isinstance(saver_instance, MetricsSaver): # Ensure MetricsSaver is imported
                metrics_saver_instance = saver_instance
                break
        
        # Save per-model metrics if MetricsSaver exists and there are per-model metrics
        if metrics_saver_instance and per_model_metrics_collection and self.num_scorer_models > 1:
            print("Saving per-model metrics...")
            original_metrics_saver_exp_id = metrics_saver_instance.experiment_id
            for model_name, model_metrics_dict in per_model_metrics_collection.items():
                if not model_metrics_dict: # Skip if no metrics for this model
                    continue

                # Create a model-specific experiment_id
                model_specific_exp_id = f"{original_metrics_saver_exp_id}-{model_name}" if original_metrics_saver_exp_id else model_name
                metrics_saver_instance.experiment_id = model_specific_exp_id
                
                temp_model_results = ExperimentResults()
                temp_model_results.add_metrics(model_metrics_dict) # Add this model's specific metrics
                
                print(f"  Saving metrics for '{model_name}' with experiment_id '{model_specific_exp_id}'")
                try:
                    metrics_saver_instance.save_result(temp_model_results)
                except Exception as e:
                    print(f"  Error saving metrics for model '{model_name}': {e}")
            
            metrics_saver_instance.experiment_id = original_metrics_saver_exp_id # Restore original experiment_id
            print("Finished saving per-model metrics.")

        # Now run all savers (MetricsSaver will run again for averaged metrics if it's in the list)
        for saver in self.savers:
            saver_name = type(saver).__name__

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
            current_training_words = [full_list[i] for i in train_idx]
            current_testing_words = [full_list[i] for i in test_idx] # Original line for test set
            
            if not current_testing_words:
                print(f"Warning: LLMExperiment._load_data: Generated empty testing_words_list for vocab file '{path}'. This may cause issues in testers like ImageGenerationTester.")

            training_words_lists.append(current_training_words)
            testing_words_lists.append(current_testing_words)
            # testing_words_lists.append([full_list[i] for i in train_idx]) # TEMPORARY: Use training set for testing
        
        return training_words_lists, testing_words_lists, []
