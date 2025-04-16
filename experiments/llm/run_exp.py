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
        if not cfg.get('estimator_path'):
            print("Error: test_only mode requires estimator_path to be set in the config.")
            return 1
            
        if not os.path.exists(cfg.estimator_path):
            print(f"Error: Estimator file not found at {cfg.estimator_path}")
            return 1
        success = experiment.run_test_only(cfg.estimator_path)
        if not success:
            return 1 # Exit if test_only failed

    # Check for visits-only inspection mode
    elif cfg.get('test_only', False) and cfg.get('visits_path') and not cfg.get('estimator_path'):
        print("--- Running in Visits-Only Inspection Mode ---")
        if not os.path.exists(cfg.visits_path):
             print(f"Error: Visits file not found at {cfg.visits_path}")
             return 1
        success = experiment.run_visits_only(cfg.visits_path)
        if not success:
             return 1 # Exit if visits-only failed

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
