import os
import re # Added import for regular expressions
import torch
import torch.nn.functional as F # Added for cosine_similarity
import numpy as np
import datetime
import sys
import random
import hashlib
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf # Added OmegaConf
from hydra.utils import to_absolute_path # Import Hydra path utility

from stpy.helpers.helper import cartesian
# Updated imports from doexpy.env.llm
from doexpy.env.llm import (
    LLMGrid, generate_emissions, create_prompt # Removed get_scorer_model, make_theta_star
)
# Import scorer model functions from their new location
from models.scorer_models import get_scorer_model, make_theta_star
# Import embedder components
from components.embedder import BaseEmbedder, create_embedder
from components.feedback import FeedbackFactory
from components.solver import SolverFactory
from components.tester import BaseTester, ImageGenerationTester
# Import specific saver types needed for validation and estimator creation
# Removed LearnedEstimatorSaver from direct import here, will handle skipping later
from components.saver import BaseSaver, VisitsSaver, VisitsImageSaver, ConfSaver, MetricsSaver, ImageGenerationSaver # Added MetricsSaver and ImageGenerationSaver
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
            # This path is derived by run_exp.py and already includes a timestamp.
            # It takes precedence.
            self.results_dir = derived_results_dir
            print(f"Using derived results directory: {self.results_dir}")
        elif cfg.get('override_results_dir', False):
            # This path is passed directly from the command line (e.g., Makefile)
            # and should be used as-is, without adding a timestamp.
            self.results_dir = to_absolute_path(cfg.results_dir)
            print(f"Using overridden results directory: {self.results_dir}")
        else:
            # Default logic: path from config + timestamp for a new run.
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

        self._estimation_episode_keys = None

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

        # --- Override experiment_id if in test/inspect mode ---
        if self.cfg.get('test_only', False):
            # The logic to correct experiment_id for inspection mode has been removed.
            # The Makefile now generates the correct ID from the start.
            # The logic below is for non-inspection test modes.
            if not self.cfg.get('inspection_mode', False):
                input_path = self.cfg.get('visits_path') or self.cfg.get('estimator_path')
                if input_path:
                    try:
                        abs_input_path = to_absolute_path(input_path)
                        filename = os.path.basename(abs_input_path)
                        print(f"Attempting to parse original info from input filename: {filename}")

                        # Regex to capture prefix, alg, feed, optional episode, and seed from visits/estimator files
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
                            
                            print(f"  Parsed Original ID from filename: {parsed_original_id}")
                            print(f"  Overriding self.experiment_id from '{self.experiment_id}' to '{parsed_original_id}'")
                            self.experiment_id = parsed_original_id # Always update if filename parsed
                        else:
                            print(f"  Warning: Could not parse original ID from filename '{filename}' using pattern.")
                    except Exception as e:
                        print(f"  Warning: Error during parsing of input path '{input_path}': {e}")

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

        # --- Auto-configure for explore_only mode ---
        if self.cfg.get('explore_only', False):
            # Clear all testers in explore_only mode
            if self.testers:
                print(f"explore_only=true: Removing {len(self.testers)} tester(s)")
                self.testers = []

            # Filter savers to only allowed types
            from components.saver import VisitsSaver, VisitsImageSaver, ConfSaver, ReadableVisitsSaver
            allowed_savers = (VisitsSaver, VisitsImageSaver, ConfSaver, ReadableVisitsSaver)
            original_saver_count = len(self.savers)
            self.savers = [s for s in self.savers if isinstance(s, allowed_savers)]
            removed_count = original_saver_count - len(self.savers)
            if removed_count > 0:
                print(f"explore_only=true: Removed {removed_count} incompatible saver(s), kept {len(self.savers)}")

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
        from components.saver import VisitsSaver, VisitsImageSaver, ConfSaver, ReadableVisitsSaver # Import allowed savers

        results = ExperimentResults()
        results.set_visits(self.visits)
        # No estimator in explore-only mode; ensure structure is consistent
        results.set_estimators([])

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
        allowed_savers = (VisitsSaver, VisitsImageSaver, ConfSaver, ReadableVisitsSaver)
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

    def _process_human_feedback(self, visits_data, feedback_data, benchmark_keys_to_exclude=None, estimation_keys_to_include=None):
        """
        Processes human feedback JSON and visits data to generate embeddings and labels
        suitable for training the RegularizedMultinomialEstimator.

        Args:
            visits_data: Loaded visits structure List[List[Tuple(states, actions)]].
            feedback_data: Dictionary loaded from feedback.json.
            benchmark_keys_to_exclude (set): A set of episode keys (e.g., "design-15") to exclude from training.
            estimation_keys_to_include (set|None): Optional set of episode keys (e.g., "design-5") allowed for training.
        """
        print("Processing human feedback...")
        if benchmark_keys_to_exclude is None:
            benchmark_keys_to_exclude = set()
        else:
            benchmark_keys_to_exclude = set(benchmark_keys_to_exclude)

        if estimation_keys_to_include is not None:
            estimation_keys_to_include = set(estimation_keys_to_include)

        collected_comparison_embeddings = []
        collected_labels = []
        num_policies = len(visits_data) # Infer number of policies from visits structure
        embedding_cache = {}  # Cache prompt embeddings to avoid recomputing duplicates

        if num_policies == 0:
             print("Error: Cannot process feedback, visits data has zero policies.")
             return None, None

        # Regex to extract episode and timestep from filename (adjust if format changes)
        # Example: images/alg-design_episode_015_timestep_01.png
        filename_pattern = re.compile(r"alg-([a-zA-Z0-9_]+)_episode_(\d+)_timestep_(\d+)\.png$")
        episode_key_pattern = re.compile(r"([a-zA-Z0-9_]+)-(\d+)") # For "alg-ep_idx"

        processed_count = 0
        skipped_count = 0
        error_count = 0
        excluded_for_benchmark_count = 0
        excluded_for_estimation_count = 0
        # Only train on feedback that matches the current algorithm for this run
        current_algorithm = str(self.cfg.algorithm).lower()

        # --- Access the 'preferences' list ---
        preferences_list = feedback_data.get("preferences")
        if not preferences_list or not isinstance(preferences_list, list):
            print("Error: 'preferences' key not found or is not a list in feedback data.")
            return None, None
        # -------------------------------------------------

        # Progress tracking helpers
        total_preferences = len(preferences_list)
        progress_bar = None
        if total_preferences > 0:
            try:
                from tqdm.auto import tqdm
                progress_bar = tqdm(total=total_preferences, desc="Processing feedback", unit="item")
            except Exception:
                progress_bar = None
                print(f"Processing feedback entries: 0/{total_preferences}")

        # Iterate through the items in the preferences list
        for preference_entry in preferences_list:
            if not isinstance(preference_entry, dict):
                print(f"Warning: Skipping invalid preference entry (not a dict): {preference_entry}")
                skipped_count += 1
                if progress_bar:
                    progress_bar.update(1)
                elif total_preferences:
                    processed_so_far = processed_count + skipped_count + excluded_for_benchmark_count + excluded_for_estimation_count + error_count
                    if processed_so_far % 50 == 0:
                        print(f"Processing feedback entries: {processed_so_far}/{total_preferences}")
                continue

            image_filename = preference_entry.get("filename")
            preferred_policy_idx_1based = preference_entry.get("preference")

            if image_filename is None or preferred_policy_idx_1based is None:
                print(f"Warning: Skipping preference entry with missing 'filename' or 'preference': {preference_entry}")
                skipped_count += 1
                if progress_bar:
                    progress_bar.update(1)
                elif total_preferences:
                    processed_so_far = processed_count + skipped_count + excluded_for_benchmark_count + excluded_for_estimation_count + error_count
                    if processed_so_far % 50 == 0:
                        print(f"Processing feedback entries: {processed_so_far}/{total_preferences}")
                continue

            match = filename_pattern.search(image_filename)
            if not match:
                # print(f"Warning: Skipping feedback entry, could not parse filename: {image_filename}")
                skipped_count += 1
                if progress_bar:
                    progress_bar.update(1)
                elif total_preferences:
                    processed_so_far = processed_count + skipped_count + excluded_for_benchmark_count + excluded_for_estimation_count + error_count
                    if processed_so_far % 50 == 0:
                        print(f"Processing feedback entries: {processed_so_far}/{total_preferences}")
                continue

            try:
                alg_name, ep_str, ts_str = match.groups()
                # Filter out entries that don't match the current algorithm (design/random)
                if str(alg_name).lower() != current_algorithm:
                    skipped_count += 1
                    if progress_bar:
                        progress_bar.update(1)
                    elif total_preferences:
                        processed_so_far = processed_count + skipped_count + excluded_for_benchmark_count + excluded_for_estimation_count + error_count
                        if processed_so_far % 50 == 0:
                            print(f"Processing feedback entries: {processed_so_far}/{total_preferences}")
                    continue
                episode_idx = int(ep_str)
                timestep_h = int(ts_str)

                # --- Check if this preference belongs to a benchmark episode ---
                episode_key = f"{alg_name}-{episode_idx}"
                if episode_key in benchmark_keys_to_exclude:
                    excluded_for_benchmark_count += 1
                    continue # Skip this entry, it's for benchmarking
                if estimation_keys_to_include is not None and episode_key not in estimation_keys_to_include:
                    excluded_for_estimation_count += 1
                    continue
                # -------------------------------------------------------------

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

                        if prompt_k in embedding_cache:
                            embedding_cpu = embedding_cache[prompt_k]
                        else:
                            embedding_raw = self.embedder.embed_text(prompt_k) # Shape [1, dim]
                            embedding_cpu = embedding_raw.detach().cpu()
                            embedding_cache[prompt_k] = embedding_cpu

                        embeddings_for_this_comparison.append(embedding_cpu)

                    except IndexError:
                        print(f"Warning: Missing visit data for policy {k}, episode {episode_idx}.")
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
            finally:
                if progress_bar:
                    progress_bar.update(1)
                elif total_preferences:
                    processed_so_far = processed_count + skipped_count + excluded_for_benchmark_count + excluded_for_estimation_count + error_count
                    if processed_so_far % 50 == 0 or processed_so_far == total_preferences:
                        print(f"Processing feedback entries: {processed_so_far}/{total_preferences}")

        if progress_bar:
            progress_bar.close()
        if estimation_keys_to_include is not None and processed_count == 0:
            print("Error: Estimation episode specification left no data for training.")
            return None, None

        print(f"Human feedback processing complete. Used for Training: {processed_count}, Excluded for Benchmark: {excluded_for_benchmark_count}, Excluded by Estimation Spec: {excluded_for_estimation_count}, Skipped (other): {skipped_count}, Errors: {error_count}")

        if not collected_comparison_embeddings or not collected_labels:
            print("Error: No valid comparison data generated from human feedback.")
            return None, None

        # Stack final tensors
        final_comparison_embeddings = torch.stack(collected_comparison_embeddings, dim=0) # [num_samples, num_policies, dim]
        final_labels = torch.stack(collected_labels, dim=0) # [num_samples, num_policies]

        print(f"Generated training data shapes: Embeddings {final_comparison_embeddings.shape}, Labels {final_labels.shape}")
        return final_comparison_embeddings, final_labels

    def _parse_benchmark_episode_spec(self, spec: str, available_algorithms) -> dict:
        """Parse textual benchmark episode specification into per-algorithm episode sets."""
        if not spec:
            return {}

        # Normalize available algorithms to lowercase strings for matching
        available = {str(alg).lower(): set() for alg in available_algorithms}
        if not available:
            return {}

        # Split on commas or whitespace, discard empty tokens
        tokens = [token for token in re.split(r"[\s,]+", spec) if token]
        if not tokens:
            return {}

        global_episodes = set()

        for raw_token in tokens:
            token = raw_token.strip()
            if not token:
                continue

            # Strip optional surrounding brackets or parentheses
            while len(token) > 1 and token[0] in "[(" and token[-1] in ")]":
                token = token[1:-1].strip()
            if not token:
                continue

            algorithm_key = None
            range_part = token
            if ':' in token:
                algorithm_key, range_part = token.split(':', 1)
                algorithm_key = algorithm_key.strip().lower()
                range_part = range_part.strip()
            else:
                range_part = range_part.strip()

            if not range_part:
                continue

            values = set()
            if '-' in range_part:
                start_str, end_str = range_part.split('-', 1)
                try:
                    start = int(start_str)
                    end = int(end_str)
                except ValueError:
                    print(f"Warning: Unable to parse benchmark episode token '{raw_token}'.")
                    continue
                if end < start:
                    start, end = end, start
                values.update(range(start, end + 1))
            else:
                try:
                    values.add(int(range_part))
                except ValueError:
                    print(f"Warning: Unable to parse benchmark episode token '{raw_token}'.")
                    continue

            if algorithm_key is None or algorithm_key in {'all', '*'}:
                global_episodes.update(values)
            else:
                if algorithm_key not in available:
                    print(f"Warning: Benchmark episode spec references unknown algorithm '{algorithm_key}'.")
                    continue
                available[algorithm_key].update(values)

        # Apply global selections to each algorithm present
        for alg in available:
            if global_episodes:
                available[alg].update(global_episodes)

        # Drop algorithms with no specified episodes
        return {alg: eps for alg, eps in available.items() if eps}

    def _select_benchmark_episodes(self, feedback_path: str):
        """Choose benchmark episodes.

        Simplified policy:
        - If the input feedback already contains 'benchmark_episode_keys', use them verbatim.
        - Otherwise, select the last N episodes per algorithm present in the preferences
          (N = cfg.num_benchmark_episodes, default 10), and record only 'benchmark_episode_keys'.
        """
        if not getattr(self, 'feedback_data', None):
            return

        self._estimation_episode_keys = None

        episode_spec = str(self.cfg.get('benchmark_episode_spec') or '').strip()
        estimation_spec = str(self.cfg.get('estimation_episode_spec') or '').strip()

        stored_keys = self.feedback_data.get('benchmark_episode_keys') if isinstance(self.feedback_data.get('benchmark_episode_keys'), list) else []
        has_stored_keys = bool(stored_keys)

        preferences = self.feedback_data.get("preferences")
        if not preferences:
            if has_stored_keys:
                stored_sorted = sorted(dict.fromkeys(stored_keys))
                self.feedback_data['benchmark_episode_keys'] = stored_sorted
                print(f"Using benchmark_episode_keys from feedback file (count={len(stored_sorted)}).")
            else:
                self.feedback_data.pop('benchmark_episode_keys', None)
            if estimation_spec:
                print(f"Warning: Estimation episode spec '{estimation_spec}' ignored because no feedback preferences are available.")
            self.feedback_data.pop('estimation_episode_keys', None)
            return

        num_benchmark = self.cfg.get('num_benchmark_episodes', 10)
        try:
            num_benchmark = int(num_benchmark)
        except (TypeError, ValueError):
            num_benchmark = 10
        if num_benchmark <= 0:
            self.feedback_data.pop('estimation_episode_keys', None)
            return

        # Collect unique episodes per algorithm
        episodes_by_alg = {}
        for entry in preferences:
            alg = entry.get('algorithm')
            ep = entry.get('episode')
            if alg is None or ep is None:
                continue
            try:
                ep_idx = int(ep)
            except (TypeError, ValueError):
                continue
            episodes_by_alg.setdefault(str(alg).lower(), set()).add(ep_idx)

        if not episodes_by_alg:
            self.feedback_data.pop('estimation_episode_keys', None)
            return

        final_benchmark_keys = None
        benchmark_source = None

        if episode_spec:
            overrides = self._parse_benchmark_episode_spec(episode_spec, episodes_by_alg.keys())
            if overrides:
                specified_keys = []
                for alg, episodes in overrides.items():
                    available = episodes_by_alg.get(alg, set())
                    if not available:
                        print(f"Warning: Benchmark spec references algorithm '{alg}' with no episodes in data; skipping.")
                        continue
                    matching = sorted(ep for ep in episodes if ep in available)
                    if not matching:
                        print(f"Warning: Benchmark spec for algorithm '{alg}' did not match available episodes; skipping.")
                        continue
                    for ep in matching:
                        specified_keys.append(f"{alg}-{ep}")
                specified_keys = sorted(dict.fromkeys(specified_keys))
                if specified_keys:
                    final_benchmark_keys = specified_keys
                    benchmark_source = ('spec', episode_spec)
                else:
                    print(f"Warning: Benchmark episode spec '{episode_spec}' did not match any episodes; falling back to automatic selection.")
            else:
                print(f"Warning: Benchmark episode spec '{episode_spec}' did not match any episodes; falling back to existing keys or automatic selection.")

        if final_benchmark_keys is None and has_stored_keys:
            final_benchmark_keys = sorted(dict.fromkeys(stored_keys))
            benchmark_source = ('stored', len(final_benchmark_keys))

        if final_benchmark_keys is None:
            benchmark_episode_keys = []
            for alg, eps in sorted(episodes_by_alg.items()):
                sorted_eps = sorted(eps)
                if not sorted_eps:
                    continue
                take = min(num_benchmark, len(sorted_eps))
                last_eps = sorted_eps[-take:]
                for ep in last_eps:
                    benchmark_episode_keys.append(f"{alg}-{ep}")
            final_benchmark_keys = sorted(dict.fromkeys(benchmark_episode_keys))
            benchmark_source = ('auto', num_benchmark)

        self.feedback_data['benchmark_episode_keys'] = final_benchmark_keys
        if benchmark_source and benchmark_source[0] == 'spec':
            print(f"Using benchmark episodes from spec '{episode_spec}': {final_benchmark_keys}")
        elif benchmark_source and benchmark_source[0] == 'stored':
            print(f"Using benchmark_episode_keys from feedback file (count={len(final_benchmark_keys)}).")
        else:
            print(f"Using last {num_benchmark} episodes per algorithm for benchmarking: {final_benchmark_keys}")

        # Process optional estimation episode specification
        self.feedback_data.pop('estimation_episode_keys', None)
        if estimation_spec:
            overrides = self._parse_benchmark_episode_spec(estimation_spec, episodes_by_alg.keys())
            if overrides:
                specified_keys = []
                for alg, episodes in overrides.items():
                    available = episodes_by_alg.get(alg, set())
                    if not available:
                        print(f"Warning: Estimation spec references algorithm '{alg}' with no episodes in data; skipping.")
                        continue
                    matching = sorted(ep for ep in episodes if ep in available)
                    if not matching:
                        print(f"Warning: Estimation spec for algorithm '{alg}' did not match available episodes; skipping.")
                        continue
                    for ep in matching:
                        specified_keys.append(f"{alg}-{ep}")
                specified_keys = sorted(dict.fromkeys(specified_keys))
                if specified_keys:
                    estimation_set = set(specified_keys)
                    benchmark_set = set(final_benchmark_keys)
                    overlap = estimation_set & benchmark_set
                    if overlap:
                        print(f"Warning: Estimation spec includes benchmark episodes {sorted(overlap)}; excluding them from training.")
                        estimation_set -= overlap
                    self._estimation_episode_keys = estimation_set
                    estimation_list = sorted(estimation_set)
                    self.feedback_data['estimation_episode_keys'] = estimation_list
                    if estimation_list:
                        print(f"Using estimation episodes from spec '{estimation_spec}': {estimation_list}")
                    else:
                        print(f"Warning: Estimation episode spec '{estimation_spec}' left no episodes after excluding benchmarks.")
                else:
                    print(f"Warning: Estimation episode spec '{estimation_spec}' did not match any episodes; using full training set.")
            else:
                print(f"Warning: Estimation episode spec '{estimation_spec}' did not match any episodes; using full training set.")
        else:
            self._estimation_episode_keys = None

    def run_test_only(self, mode: str, estimator_path: str = None, visits_path: str = None, feedback_path: str = None) -> bool:
        """
        Run in test-only mode. Behavior depends on provided paths and mode.
        """
        print(f"--- Running Test-Only Mode: {mode} ---")

        # --- Centralized Data Loading ---
        # Load visits if visits_path is provided.
        if visits_path:
            abs_visits_path = to_absolute_path(visits_path)
            if not os.path.exists(abs_visits_path):
                print(f"Error: Visits file not found: {abs_visits_path}")
                return False
            print(f"Loading visits from: {abs_visits_path}")
            try:
                self.visits = torch.load(abs_visits_path)
            except Exception as e:
                print(f"Error loading visits from {abs_visits_path}: {e}")
                return False
        
        # Load feedback data if feedback_path is provided.
        if feedback_path:
            abs_feedback_path = to_absolute_path(feedback_path)
            if not os.path.exists(abs_feedback_path):
                print(f"Error: Feedback file not found: {abs_feedback_path}")
                return False
            print(f"Loading feedback data from: {abs_feedback_path}")
            try:
                import json
                with open(abs_feedback_path, 'r') as f:
                    self.feedback_data = json.load(f)
                # Feedback metadata (e.g., style_description) is stored for reference only.
                self._select_benchmark_episodes(abs_feedback_path)
                if self.feedback_data.get('benchmark_episode_keys'):
                    override_feedback_path = Path(self.results_dir) / 'feedback_with_benchmarks.json'
                    try:
                        with open(override_feedback_path, 'w') as override_file:
                            json.dump(self.feedback_data, override_file, indent=2)
                        self.cfg.feedback_path = str(override_feedback_path)
                        print(f"Saved feedback with benchmark selection to: {override_feedback_path}")
                    except Exception as e:
                        print(f"Warning: Failed to persist benchmark-adjusted feedback: {e}")
            except Exception as e:
                print(f"Error loading or parsing feedback data from {abs_feedback_path}: {e}")
                return False
        # --- End Centralized Data Loading ---
        
        # (Reverted) No secondary visits loading; benchmarking uses the current run's visits

        # --- Mode Logic ---
        if mode == "load_estimator":
            if not estimator_path:
                print(f"Error: Estimator file not provided for mode '{mode}'.")
                return False
            print(f"Loading estimator from: {estimator_path}")
            if not self.load_estimator(estimator_path):
                 print("Error: Failed to load estimator.")
                 return False
            self.test_and_save(current_mode=mode)
            return True

        elif mode == "estimate_from_visits":
            if self.visits is None:
                print(f"Error: Visits data not loaded, but required for mode '{mode}'.")
                return False
            
            print("Estimating/Training estimator using visits...")
            try:
                # Loop through each configured scorer model to collect labels and fit an estimator
                for i in range(self.num_scorer_models):
                    model_name = self.scorer_model_names[i]
                    print(f"Processing model '{model_name}' for estimator training from visits...")

                    if i >= len(self.feedbacks) or self.feedbacks[i] is None:
                        print(f"  Skipping model '{model_name}': Feedback component not available.")
                        continue
                    if i >= len(self._theta_stars) or self._theta_stars[i] is None:
                        print(f"  Skipping model '{model_name}': Ground truth scorer (_theta_stars[{i}]) not available.")
                        continue
                    
                    print(f"  Collecting labels for model '{model_name}' from visits...")
                    self.feedbacks[i].collect_labels(self.cfg, self.visits, self._theta_stars[i])

                    print(f"  Fitting estimator for model '{model_name}' with collected labels...")
                    self.feedbacks[i].fit_estimator()

                    if i < len(self.estimators):
                        self.estimators[i] = self.feedbacks[i].estimator
                        if self.estimators[i] and getattr(self.estimators[i], 'fitted', False):
                            print(f"  Estimator for model '{model_name}' trained successfully.")
                        else:
                            print(f"  Warning: Estimator for model '{model_name}' not fitted after training attempt.")
                    else:
                        print(f"  Warning: Estimators list too short for model '{model_name}' (index {i}).")
            except Exception as e:
                print(f"Error during estimator training from visits: {e}")
                return False
            
            any_estimator_fitted = any(est and getattr(est, 'fitted', False) for est in self.estimators if est is not None)
            if any_estimator_fitted:
                print("At least one estimator trained successfully from visits.")
                self.test_and_save(current_mode=mode)
                return True
            else:
                print("Error: No estimators were successfully trained from visits.")
                return False

        elif mode == "inspect_visits":
            if self.visits is None:
                print(f"Error: Visits data not loaded, but required for mode '{mode}'.")
                return False
            self.estimators = [None] * self.num_scorer_models
            print("Skipping estimator training/loading in inspection mode.")
            self.test_and_save(current_mode=mode)
            return True

        elif mode == "load_estimator_and_feedback":
            if not estimator_path or self.feedback_data is None:
                print(f"Error: Mode '{mode}' requires both estimator_path and feedback_path.")
                return False
            
            print(f"Loading estimator from: {estimator_path}")
            if not self.load_estimator(estimator_path):
                 print("Error: Failed to load estimator.")
                 return False
            
            # Pass feedback_data down to testing. The environment is NOT modified.
            self.test_and_save(current_mode=mode, feedback_data=self.feedback_data)
            return True

        elif mode == "run_human_feedback":
            if self.visits is None or self.feedback_data is None:
                print(f"Error: Mode '{mode}' requires both visits_path and feedback_path to be provided and loaded.")
                return False

            # Step 1: Train the estimator using the original environment state, excluding benchmark episodes
            print("Processing human feedback to generate training data (using original env state)...")
            benchmark_keys = set(self.feedback_data.get("benchmark_episode_keys", []))
            print(f"Excluding {len(benchmark_keys)} benchmark episodes from training.")
            
            estimation_keys = getattr(self, '_estimation_episode_keys', None)
            comparison_embeddings, labels = self._process_human_feedback(
                self.visits,
                self.feedback_data,
                benchmark_keys_to_exclude=benchmark_keys,
                estimation_keys_to_include=estimation_keys,
            )

            if comparison_embeddings is None or labels is None:
                print("Error: Failed to process human feedback into training data. Skipping estimator fitting.")
                return False
            
            if not self.feedbacks or self.feedbacks[0] is None:
                print("Error: Feedback component (self.feedbacks[0]) not initialized. Cannot fit estimator.")
                return False

            print("Fitting estimator using processed human feedback...")
            try:
                self.feedbacks[0]._collected_data = [(comparison_embeddings, labels)]
                self.feedbacks[0].fit_estimator()
                self.estimators[0] = self.feedbacks[0].estimator
                print("Estimator fitted successfully with human feedback.")
            except Exception as e:
                print(f"Error fitting estimator with human feedback: {e}")
                return False
            
            # Step 2: Proceed to testing, passing feedback_data down so components can use the style description.
            # The environment's base_prompt is NOT modified.
            self.test_and_save(current_mode=mode, feedback_data=self.feedback_data)
            return True
        
        else:
             print(f"Error: Unknown test_only mode '{mode}' received by experiment runner.")
             return False

    def test_and_save(self, current_mode="full_run", feedback_data=None): # Add feedback_data
        """
        Final estimation, testing and saving of results.

        Args:
            current_mode (str): The mode the experiment is running in.
            feedback_data (dict, optional): Loaded data from feedback.json.
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

        # Extract style_description from feedback_data if available (metadata only)
        style_description = None
        if feedback_data:
            candidate_description = feedback_data.get("style_description")
            if isinstance(candidate_description, str) and candidate_description.strip():
                style_description = candidate_description.strip()
            else:
                legacy_prompt = feedback_data.get("user_prompt")
                if isinstance(legacy_prompt, str) and legacy_prompt.strip():
                    style_description = legacy_prompt.strip()
                else:
                    legacy_prompts = feedback_data.get("user_prompts", [])
                    if isinstance(legacy_prompts, list):
                        for prompt in legacy_prompts:
                            if isinstance(prompt, str) and prompt.strip():
                                style_description = prompt.strip()
                                break
            if style_description:
                results.add_metadata('style_description', style_description)


        # --- Pre-run Validation ---
        print("Validating requirements for configured testers and savers...")
        # Check if *at least one* estimator is available in the list
        estimators_available = self.estimators and any(est is not None for est in self.estimators)
        # Check if visits list exists, is not empty, and its first element is not empty
        visits_available = bool(self.visits and isinstance(self.visits, list) and self.visits[0])

        # Import tester/saver classes for isinstance checks
        from components.tester import PreferenceTester, CosineTester, ImageGenerationTester, HumanFeedbackBenchmarkTester
        from components.saver import LearnedEstimatorSaver, VisitsSaver, VisitsImageSaver, ReadableVisitsSaver, ConfSaver

        # --- Mode-Specific Validation ---
        if current_mode == "run_human_feedback":
            # Ensure LearnedEstimatorSaver is present
            # Check if LearnedEstimatorSaver is present (it will be skipped if num_models > 1)
            has_les = any(isinstance(s, LearnedEstimatorSaver) for s in self.savers)
            if not has_les:
                 print(f"Warning: Mode '{current_mode}' typically uses LearnedEstimatorSaver, but it was not found in config.")
            # Ensure at least one estimator was actually fitted
            if not estimators_available:
                 raise ValueError(f"Mode '{current_mode}' completed but no estimator is available. Training likely failed.")
            # Testers are now run in this mode, so we validate them.
            print(f"Validation for mode '{current_mode}': Estimator(s) available.")
        # -----------------------------

        # Validate Testers (only skip for inspect_visits mode)
        if current_mode != "inspect_visits":
            for tester in self.testers:
                tester_name = type(tester).__name__
                # Check for estimator requirement
                if isinstance(tester, (PreferenceTester, CosineTester, HumanFeedbackBenchmarkTester)):
                    if not estimators_available:
                        raise ValueError(f"Tester '{tester_name}' requires an estimator, but none were loaded or available.")
                
                # Check for visits requirement
                if isinstance(tester, HumanFeedbackBenchmarkTester):
                    if not visits_available:
                        raise ValueError(f"Tester '{tester_name}' requires visits data, but it is not available.")

                # Check for ImageGenerationTester requirements
                elif isinstance(tester, ImageGenerationTester):
                     if tester.prompt_ranking_model != 'gt' and not estimators_available:
                         raise ValueError(f"Tester '{tester_name}' is configured with prompt_ranking_model='{tester.prompt_ranking_model}', but no estimator is available.")
                     scorer_models_available = self._scorer_models and any(sm is not None for sm in self._scorer_models)
                     if tester.prompt_ranking_model == 'gt' and not scorer_models_available:
                          raise ValueError(f"Tester '{tester_name}' is configured with prompt_ranking_model='gt', but no ground truth scorer models are available.")
                     if self.embedder is None:
                          raise ValueError(f"Tester '{tester_name}' requires an embedder, but it's not available.")
        else:
            print("Skipping tester validation in 'inspect_visits' mode.")


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
        results.add_metadata('base_prompt', self.env.base_prompt) # Use the original env base prompt
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

        # Skip testers if in inspection mode
        if current_mode == "inspect_visits":
            print("Skipping testers in 'inspect_visits' mode.")
        else:
            # Separate ImageGenerationTesters from others
            image_gen_testers = [t for t in self.testers if isinstance(t, ImageGenerationTester)]
            other_testers = [t for t in self.testers if not isinstance(t, ImageGenerationTester)]

            # Run non-image-gen testers first
            print("\n--- Running Standard Testers ---")
            num_iterations = self.num_scorer_models if self.num_scorer_models > 0 else 1
            for i in range(num_iterations):
                model_name, estimator, theta_star, scorer_model = None, None, None, None
                if self.num_scorer_models > 0:
                    model_name = self.scorer_model_names[i]
                    estimator = self.estimators[i] if self.estimators and i < len(self.estimators) else None
                    theta_star = self._theta_stars[i] if self._theta_stars and i < len(self._theta_stars) else None
                    scorer_model = self._scorer_models[i] if self._scorer_models and i < len(self._scorer_models) else None
                    print(f"\nTesting Model: {model_name}")
                else:
                    model_name = "human_feedback"
                    estimator = self.estimators[0] if self.estimators else None
                    theta_star, scorer_model = None, None
                    print(f"\nTesting Model: {model_name}")

                if self.num_scorer_models > 0 and (theta_star is None or scorer_model is None):
                    print(f"Skipping testing for model '{model_name}': Ground truth components missing.")
                    continue
                if self.num_scorer_models == 0 and estimator is None:
                    print(f"Skipping testing for model '{model_name}': Estimator not available.")
                    continue

                for tester in other_testers:
                    tester_name = type(tester).__name__
                    print(f"  Running tester: {tester_name}")
                    if isinstance(tester, (PreferenceTester, CosineTester, HumanFeedbackBenchmarkTester)) and estimator is None:
                        print(f"  Skipping {tester_name} for model '{model_name}': Estimator not available for this iteration.")
                        continue
                    
                    try:
                        test_args = {
                            'cfg': self.cfg, 'env': self.env, 'estimator': estimator,
                            'theta_star': theta_star, 'scorer_model': scorer_model,
                            'training_words_list': self.training_words, 'testing_words_list': self.testing_words,
                            'visits': self.visits if hasattr(self, 'visits') else None
                        }
                        if isinstance(tester, CosineTester):
                            test_args['feedback'] = self.feedbacks[i] if self.feedbacks and i < len(self.feedbacks) else None
                        
                        tester_results = tester.run_test(**test_args)
                        if tester_results:
                            for key, value in tester_results.items():
                                if isinstance(value, float): print(f"    Model '{model_name}' - {key}: {value:.4f}")
                                else: print(f"    Model '{model_name}' - {key}: {value}")
                                if model_name not in per_model_metrics_collection: per_model_metrics_collection[model_name] = {}
                                per_model_metrics_collection[model_name][key] = value
                    except Exception as e:
                        print(f"  Error running tester {tester_name} for model '{model_name}': {e}")

            # Image generation testers rely on prompts derived from visits; style_description is metadata only.
            print("--------------------------------------\n")

        # If there's only one model (or one set of metrics, like in human feedback mode), 
        # merge its metrics into the main results.metrics so MetricsSaver can pick them up.
        if len(per_model_metrics_collection) == 1:
            # Get the single model name/key and its metrics
            single_model_name = list(per_model_metrics_collection.keys())[0]
            single_model_metrics = per_model_metrics_collection[single_model_name]
            
            if single_model_metrics:
                print(f"Merging metrics from single model/run '{single_model_name}' into main results for saving.")
                for key, value in single_model_metrics.items():
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

        # Now run all savers. ImageGenerationSaver is skipped as it's handled within the tester loop.
        for saver in self.savers:
            saver_name = type(saver).__name__

            if isinstance(saver, ImageGenerationSaver):
                print(f"Skipping saver: {saver_name} (handled within tester loop to ensure correct context)")
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
            current_training_words = [full_list[i] for i in train_idx]
            current_testing_words = [full_list[i] for i in test_idx] # Original line for test set
            
            if not current_testing_words:
                print(f"Warning: LLMExperiment._load_data: Generated empty testing_words_list for vocab file '{path}'. This may cause issues in testers like ImageGenerationTester.")

            training_words_lists.append(current_training_words)
            testing_words_lists.append(current_testing_words)
            # testing_words_lists.append([full_list[i] for i in train_idx]) # TEMPORARY: Use training set for testing
        
        return training_words_lists, testing_words_lists, []
