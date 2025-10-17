
from clearml import Dataset

"""
   dataset_stats = {
        'total_samples': ,
        'features': ,
        'classes': ,
        'train_samples': ,
        'test_samples': 
    }
"""


def create_dataset(dataset_name: str,
                   dataset_project: str,
                   dataset_path: str,
                   dataset_stats: str,
                   ):
    dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project
    )

    dataset.add_files(path=dataset_path)

    stats = dataset_stats
    dataset.set_metadata(stats)

    dataset.upload()
    dataset.finalize()

