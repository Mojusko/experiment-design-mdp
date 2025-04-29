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

    # Check if we're in test-only mode (which includes inspection mode)
    if cfg.get('test_only', False):
        print("--- Running in Test-Only / Inspection Mode ---")
        # The logic inside run_test_only now handles checking for estimator_path, visits_path, and feedback_path
        # and setting the correct mode. Pass relevant paths from config.
        success = experiment.run_test_only(
            estimator_path=cfg.get('estimator_path'), # Can be None
            feedback_path=cfg.get('feedback_path')    # Can be None
        )
        if not success:
            # run_test_only returns False if required input paths are missing/invalid or processing fails
            print("Exiting due to error during test_only/inspection/train_human execution.")
            return 1 # Exit if test_only/inspection failed

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
