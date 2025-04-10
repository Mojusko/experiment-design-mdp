import hydra
import torch # Added torch import
from experiment import LLMExperiment
import sys
import os
from omegaconf import OmegaConf, DictConfig

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("\n=== Configuration ===")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("===================\n")

    # --- Print CUDA Availability ---
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print("===================\n")
    # -----------------------------

    experiment = LLMExperiment(cfg)
    
    # Check if we're in test-only mode
    if hasattr(cfg, 'test_only') and cfg.test_only:
        if not hasattr(cfg, 'estimator_path') or not cfg.estimator_path:
            print("Error: test_only mode requires estimator_path to be set in config")
            return 1
            
        if not os.path.exists(cfg.estimator_path):
            print(f"Error: Estimator file not found at {cfg.estimator_path}")
            return 1
            
        # Run test-only mode
        success = experiment.run_test_only(cfg.estimator_path)
        if not success:
            return 1
    else:
        # Run full experiment
        experiment.run()
        experiment.test_and_save()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
