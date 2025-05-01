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

        # Determine the mode based on provided paths
        mode = None
        if estimator_path and not visits_path and not feedback_path:
            mode = "load_estimator"
            print("Mode: Load Estimator")
        elif not estimator_path and visits_path and not feedback_path:
            mode = "estimate_from_visits"
            print("Mode: Estimate from Visits")
        elif estimator_path and not visits_path and feedback_path:
            mode = "load_estimator_and_feedback"
            print("Mode: Load Estimator and Feedback")
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
