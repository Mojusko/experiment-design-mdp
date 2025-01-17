import hydra
from experiment import LLMExperiment
import sys
from omegaconf import OmegaConf, DictConfig



# Early interception of --debug flag
DEBUG_MODE = False
if "--debug" in sys.argv:
    DEBUG_MODE = True
    sys.argv.remove("--debug")  # Remove the flag so Hydra won't see it


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    if DEBUG_MODE:
        debug_cfg = OmegaConf.load("conf/debug.yaml")
        cfg = OmegaConf.merge(cfg, debug_cfg)
        print("Running in debug mode with config:", cfg)

    experiment = LLMExperiment(cfg)
    experiment.run()
    experiment.test_and_save()

if __name__ == "__main__":
    main()
