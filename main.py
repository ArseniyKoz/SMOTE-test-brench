from configs.config_loader import ConfigLoader
from experiments.experiment_runner import ExperimentConfig, ExperimentRunner


def main():

    config_name = 'experiment/per_dataset.yaml'

    loader = ConfigLoader(config_name)
    cfg = loader.load()

    config = ExperimentConfig()
    config.clearml_project_name = "SMOTE Test Bench"
    config.cv_folds = 10
    config.selected_classifiers = ['LogisticRegression', 'RandomForest', 'SVM']

    runner = ExperimentRunner(
        config=config,
        create_clearml_task=False,
        mode='dataset'
    )

    results = runner.per_dataset_experiment(
        dataset_name=cfg['dataset'],
        config_name=config_name
    )

    runner.close_task()


if __name__ == "__main__":
    main()
