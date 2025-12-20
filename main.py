import sys

from configs.config_loader import ConfigLoader
from experiments.experiment_runner import ExperimentConfig, ExperimentRunner


def main(cfg_name: str = 'experiments/base_experiment'):

    loader = ConfigLoader(cfg_name)
    cfg = loader.load()
    experiment_config = cfg['experiment_config']

    config = ExperimentConfig(experiment_config)

    runner = ExperimentRunner(
        config=config,
        create_clearml_task=False,
    )

    runner.direct_experiments(
        config_name=config_name
    )


if __name__ == "__main__":
    #config_name = str(sys.argv[1])
    config_name = "experiment/base_experiment.yaml"
    main(config_name)
