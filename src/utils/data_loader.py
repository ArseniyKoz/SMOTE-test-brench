from typing import Dict

import pandas as pd
from clearml import Dataset

from configs.config_loader import ConfigLoader

config_name = 'data/datasets.yaml'


def get_dataset_config(dataset_name: str) -> Dict[str, str]:
    loader = ConfigLoader(config_name)
    config = loader.load()

    return config[dataset_name]


def fetch_dataset(dataset_name: str, max_workers: int = 8):
    dataset_config = get_dataset_config(dataset_name)
    dataset_id = dataset_config['data_id']

    if not dataset_id:
        raise ValueError(
            f"Dataset ID not found for '{dataset_name}'. "
            f"Check datasets.yaml configuration."
        )

    try:
        dataset = Dataset.get(dataset_id=dataset_id)
        local_path = dataset.get_local_copy(max_workers=max_workers)
        data = pd.read_csv(f"{local_path}/{dataset_name}.csv")

        metadata = dataset.get_metadata()
        return data, metadata

    except Exception as e:
        raise
