import hydra
from experiment import LLMExperiment
import sys
from omegaconf import OmegaConf, DictConfig

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("\n=== Configuration ===")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("===================\n")

    experiment = LLMExperiment(cfg)
    experiment.run()
    experiment.test_and_save()

if __name__ == "__main__":
    main()
