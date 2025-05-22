import hydra
import torch # Added torch import
from experiment import LLMExperiment
import sys
import os
from omegaconf import OmegaConf, DictConfig

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("\n=== Configuration ===", flush=True)
    print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    print("===================\n", flush=True)

    # --- Print CUDA Availability ---
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}", flush=True)
    if cuda_available:
        print(f"CUDA Device Count: {torch.cuda.device_count()}", flush=True)
        print(f"Current CUDA Device: {torch.cuda.current_device()}", flush=True)
        print(f"CUDA Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}", flush=True)
    print("===================\n", flush=True)
    # -----------------------------

    experiment = LLMExperiment(cfg)

    # --- Handle Special Modes ---
    if cfg.get('test_only', False) and cfg.get('explore_only', False):
        print("Error: Cannot set both test_only and explore_only to true.")
        return 1

    # Check if we're in test-only mode
    if cfg.get('test_only', False):
        print("--- Running in Test-Only Mode ---")
        estimator_path = cfg.get('estimator_path')
        visits_path = cfg.get('visits_path')
        feedback_path = cfg.get('feedback_path')

        # Determine the mode based on provided paths AND inspection_mode flag
        mode = None
        derived_results_dir = None # Initialize derived path as None

        if cfg.get('inspection_mode', False):
            mode = "inspect_visits"
            print("Mode: Inspect Visits (based on inspection_mode=true)")
            # For inspection, the input path is always visits_path
            input_path_for_dir = visits_path
            if not visits_path:
                 print("\nError: inspection_mode=true requires visits_path to be provided.")
                 return 1
        else:
            # Original path-based mode detection for non-inspection test_only runs
            if estimator_path and not visits_path and not feedback_path:
                mode = "load_estimator"
                print("Mode: Load Estimator")
                input_path_for_dir = estimator_path
            elif not estimator_path and visits_path and not feedback_path:
                mode = "estimate_from_visits"
                print("Mode: Estimate from Visits")
                input_path_for_dir = visits_path
            elif estimator_path and not visits_path and feedback_path:
                mode = "load_estimator_and_feedback"
                print("Mode: Load Estimator and Feedback")
                input_path_for_dir = estimator_path # Use estimator path as base
            elif not estimator_path and visits_path and feedback_path:
                 mode = "train_human_feedback"
                 print("Mode: Train Human Feedback")
                 input_path_for_dir = visits_path # Use visits path as base for dir derivation
            else:
                 # Invalid combination if not inspection mode
                 print("\nError: Invalid combination of paths for test_only mode.")
                 print("Valid combinations for llm-test-only:")
                 print("  1. --config-name=config_inference estimator_path=/path/to/estimator.pt")
                 print("  2. --config-name=config_inference visits_path=/path/to/visits.pkl")
                 print("  3. --config-name=config_inference estimator_path=/path/to/estimator.pt feedback_path=/path/to/feedback.json")
                 return 1 # Exit due to invalid combination

        # --- Derive Results Directory if not overridden ---
        if input_path_for_dir and not cfg.get('override_results_dir', False):
            try:
                from hydra.utils import to_absolute_path
                import datetime
                absolute_input_path = to_absolute_path(input_path_for_dir)
                if not os.path.exists(absolute_input_path):
                     print(f"Warning: Input path '{absolute_input_path}' not found. Using default results_dir.")
                else:
                    original_dir = os.path.dirname(absolute_input_path)
                    # Check if the original_dir itself exists
                    if not os.path.isdir(original_dir):
                         print(f"Warning: Parent directory '{original_dir}' of input path not found. Using default results_dir.")
                    else:
                        timestamp = os.environ.get('TIMESTAMP', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
                        # Use mode name in the directory structure
                        # Example: test-load_estimator-..., test-estimate_from_visits-...
                        # --- Try to extract original ID from input path ---
                        original_id_part = None
                        try:
                            # Extract the filename part (e.g., visits-feedback-rand-mult-1.pkl)
                            input_filename = os.path.basename(absolute_input_path)
                            # Remove common prefixes/suffixes to isolate the core ID part
                            # Example: remove "visits-", ".pkl", "estimator-", ".pt"
                            id_core = input_filename.replace("visits-", "").replace("estimator-", "").split('.')[0]
                            # Further refine if needed, e.g., remove timestamp if present
                            # This is heuristic, might need adjustment based on actual filename patterns
                            original_id_part = id_core
                            print(f"Extracted original ID part: {original_id_part}")
                        except Exception as path_e:
                            print(f"Could not extract original ID from path: {path_e}")

                        # Use the extracted ID if found, otherwise fallback to cfg.experiment_id or mode
                        experiment_id_suffix = original_id_part or cfg.experiment_id or mode
                        print(f"Using experiment ID suffix for directory: {experiment_id_suffix}")
                        # ----------------------------------------------------

                        # Determine base directory for 'additional_tests'
                        if os.path.basename(original_dir) == "additional_tests":
                            tests_base_dir = original_dir
                        else:
                            tests_base_dir = os.path.join(original_dir, "additional_tests")

                        # Construct the new results directory path
                        new_results_dir = os.path.join(tests_base_dir, f"{mode}-{experiment_id_suffix}-{timestamp}")

                        # Store the derived path instead of modifying cfg
                        print(f"Derived results directory: {new_results_dir}")
                        derived_results_dir = new_results_dir
                        # cfg.results_dir = new_results_dir # REMOVED
                        # cfg.override_results_dir = True # REMOVED

            except Exception as e:
                print(f"Warning: Error deriving results directory: {e}. Using default results_dir.")
        # ----------------------------------------------------

        # --- Deprecated Modes (Handled by other Makefile targets) ---
        # elif not estimator_path and visits_path and feedback_path:
        #     mode = "train_human" # Now handled by llm-train-human-feedback target
        #     print("Mode: Train Human Feedback (Use llm-train-human-feedback target)")
        # elif not estimator_path and visits_path and not feedback_path:
        #     mode = "inspect_visits" # Now handled by llm-inspect-visits target
        #     print("Mode: Inspect Visits (Use llm-inspect-visits target)")
        # ----------------------------------------------------------
        else:
            print("\nError: Invalid combination of paths for test_only mode.")
            print("Valid combinations for llm-test-only:")
            print("  1. --config-name=config_inference estimator_path=/path/to/estimator.pt")
            print("  2. --config-name=config_inference visits_path=/path/to/visits.pkl")
            print("  3. --config-name=config_inference estimator_path=/path/to/estimator.pt feedback_path=/path/to/feedback.json")
            # print("Use 'llm-train-human-feedback' target for: visits_path + feedback_path")
            # print("Use 'llm-inspect-visits' target for: visits_path only (with config_inspect)")
            return 1 # Exit due to invalid combination

        # Initialize Experiment - Pass derived_results_dir if available
        experiment = LLMExperiment(cfg, derived_results_dir=derived_results_dir)

        # Call the experiment's test-only runner with the determined mode
        success = experiment.run_test_only(
            mode=mode,
            estimator_path=estimator_path,
            visits_path=visits_path,
            feedback_path=feedback_path
        )
        if not success:
            print("Exiting due to error during test_only execution.")
            return 1 # Exit if test_only failed

    # Check if we're in explore-only mode
    elif cfg.get('explore_only', False):
        print("--- Running in Explore-Only Mode ---")
        experiment.run_explore_only()

    # Otherwise, run the full experiment (explore, estimate, test, save)
    else:
        print("--- Running Full Experiment ---")
        experiment.run()
        experiment.test_and_save()

    return 0 # Success

if __name__ == "__main__":
    sys.exit(main())
