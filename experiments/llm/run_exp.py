import hydra
from experiment import LLMExperiment
import sys
from omegaconf import OmegaConf, DictConfig

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    import ipdb; ipdb.set_trace()
    experiment = LLMExperiment(cfg)
    experiment.run()
    experiment.test_and_save()

if __name__ == "__main__":
    main()
