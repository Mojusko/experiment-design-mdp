import hydra
from omegaconf import DictConfig
from experiment import LLMExperiment

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    experiment = LLMExperiment(cfg)
    experiment.run()
    experiment.test_and_save()

if __name__ == "__main__":
    main()
