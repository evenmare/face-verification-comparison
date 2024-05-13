from typing import TypeVar

import pandas as pd
from deepface import DeepFace
from tqdm import tqdm

from datasets_adapters import KYCDatasetAdapter


DatasetAdapterVar = TypeVar('DatasetAdapterVar', bound=KYCDatasetAdapter)


class ModelMetricSet:
    """ Class that implements functionality to count model metrics """
    identifier: str

    tp: int
    tn: int
    fp: int
    fn: int

    accuracy: float
    recall: float
    fpr: float
    precision: float
    f_score: float

    def __str__(self) -> str:
        return f'Metric {self.identifier}'

    def __init__(self, identifier: str) -> None:
        """
        :param identifier: metric set identifier
        """
        self.identifier = identifier

        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def calculate(self) -> None:
        """ Metrics set calculation """
        self.accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        self.recall = self.tp / (self.tp + self.fn)
        self.fpr = self.fp / (self.fp + self.tn)
        self.precision = self.tp / (self.tp + self.fp)
        self.f_score = 2 * (self.tp / (2 * self.tp + self.fn + self.fp))

    def to_dataframe(self) -> pd.DataFrame:
        """ Returns metrics in dataset format """
        details = dict(vars(self))
        details.pop('identifier')
        return pd.DataFrame({self.identifier: details})

class ModelVerifier:
    """ Class that implements functionality to verify model """
    model_name: str
    metrics_sets: list[ModelMetricSet]

    def __init__(self, model_name: str) -> None:
        """
        :param model_name: name of model
        """
        self.metrics_sets = []
        self.model_name = model_name

    def execute(self, dataset_adapter: DatasetAdapterVar) -> pd.DataFrame:
        """
        Verification execution
        :param dataset_adapter: adapter used to iterate by dataset
        :return: metrics in dataframe format
        """
        execution_metric = ModelMetricSet(f'{self.model_name}:{str(dataset_adapter)}')

        print(f'Verification {self.model_name} on {str(dataset_adapter)}', end='\n\n')

        with tqdm(total=dataset_adapter.total) as pbar:
            for verification_image_path in dataset_adapter.images_paths_iterator():
                for id_image_path in dataset_adapter.id_images_paths_iterator():
                    try:
                        result = DeepFace.verify(
                            id_image_path,
                            verification_image_path,
                            model_name=self.model_name,
                        )['verified']
                    except ValueError:
                        continue

                    expected = (verification_image_path.name in
                                dataset_adapter.id_to_images_mapping[id_image_path.name])

                    _classify_run(execution_metric, result, expected)

                pbar.update(1)

        execution_metric.calculate()
        self.metrics_sets.append(execution_metric)

        return execution_metric.to_dataframe()


def _classify_run(
        metric: ModelMetricSet,
        _result: bool,
        _expected: bool
) -> None:
    if _result:
        if _expected:
            metric.tp += 1
        else:
            metric.fp += 1
    else:
        if _expected:
            metric.fn += 1
        else:
            metric.tn += 1
