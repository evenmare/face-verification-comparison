import abc
import math
import pathlib
import shutil

from typing import Iterator
from functools import cached_property
from termcolor import colored

import pandas as pd


class BaseDatasetAdapter(abc.ABC):
    """ Base Datasets Adapter """
    dataset_name: str

    sources_path: pathlib.PosixPath
    destination_path: pathlib.PosixPath

    def __str__(self) -> str:
        return self.dataset_name

    def __init__(
            self,
            sources_path: str,
            destination_path: str,
            dataset_name: str | None = None,
    ):
        """
        :param sources_path: dataset images folder path ("{dataset_path}/files")
        :param destination_path: destination data absolute path (after mixing data)
        :param dataset_name: name of dataset (defaults to None, class property set to folder name)
        """
        self.sources_path = pathlib.Path(sources_path)
        self.destination_path = pathlib.Path(destination_path)

        if not self.sources_path.is_dir():
            raise ValueError('sources_path: should be a directory path')

        self.dataset_name = dataset_name or self.sources_path.name

    @abc.abstractmethod
    def prepare_data(self) -> None:
        raise NotImplementedError('prepare_data')

    @property
    @abc.abstractmethod
    def total(self) -> int:
        raise NotImplementedError('total')


class KYCDatasetAdapter(BaseDatasetAdapter):
    """ Universal KYC Datasets Adapter """
    accordance_file_path: pathlib.PosixPath

    image_extension: str = '.jpg'

    def __repr__(self) -> str:
        return f'<KYCDatasetAdapter: {str(self)}>'

    def __init__(
            self,
            *args,
            accordande_file_path: str,
            **kwargs,
    ):
        """
        :param accordance_file_path: accordance file path (usually named as *_dataset.csv)
        """
        super().__init__(*args, **kwargs)

        self.accordance_file_path = pathlib.Path(accordande_file_path)

        if not self.accordance_file_path.is_file():
            raise ValueError('accordance_file_path: should be a file path')
        if self.accordance_file_path.suffix.lower() != '.csv':
            print(self.accordance_file_path.suffix)
            raise ValueError('accordance_file_path: should be a .csv file')

        self.id_images_path = self.destination_path / 'roots'

    @cached_property
    def _accordance_raw(self) -> pd.DataFrame:
        return pd.read_csv(self.accordance_file_path, sep=',')

    @property
    def total(self) -> int:
        return math.prod(self._accordance_raw.loc[:, 'id_2':].shape)

    @cached_property
    def _id_images_mapping_raw(self) -> dict[int, str]:
        return dict(self._accordance_raw['id_1'])

    def _get_target_path(self, filepath: pathlib.Path, image_prefix: str | None = None) -> pathlib.Path:
        image_prefix = image_prefix or ''
        target_filename = f'{image_prefix}{filepath.name}'

        new_filepath = self.destination_path / target_filename
        for id_image_path in self._id_images_mapping_raw.values():
            if f'{filepath.parts[-2]}/{filepath.parts[-1]}' == id_image_path.strip('/'):
                new_filepath = self.id_images_path / target_filename
                break

        return new_filepath

    def prepare_data(self) -> None:
        """ Prepare data for benchmark """
        # delete destination folder if exists
        print(f'Destination folder: {self.destination_path.name}'.ljust(50), end='')
        if self.destination_path.exists():
            print(colored('deleted', 'red'), end='...',)
            shutil.rmtree(self.destination_path)

        # create destination folder
        self.destination_path.mkdir(exist_ok=False, parents=True)
        self.id_images_path.mkdir(exist_ok=False)
        print(colored('created', 'green'))

        # copy files
        for source_child in self.sources_path.iterdir():
            if source_child.is_dir():  # skip external files
                folder = source_child
                image_prefix = f'{folder.name}_'

                print(f'    Processing combination {folder.name}...'.ljust(50), end='')
                for file in folder.iterdir():
                    if file.suffix.lower() == self.image_extension:
                        shutil.copy(file, self._get_target_path(file, image_prefix))
                print(colored('completed', 'green'))

    @cached_property
    def accordance(self) -> pd.DataFrame:
        """
        Accordance
        :returns: dataframe
        """
        df = self._accordance_raw.copy()

        for column_name in df.columns.array[df.columns.get_loc('id_1'):]:
            for y in range(len(df)):
                df.loc[y, column_name] = '_'.join(df[column_name][y].strip('/').split('/'))

        return df

    @cached_property
    def id_to_images_mapping(self) -> dict[int, str]:
        """
        ID images to images set mapping
        :returns: {id image filename: {image path 1, image path 2, ...}}
        """
        return {
            id_image_path: set(self.accordance.loc[i, 'id_2':].tolist())
            for i, id_image_path in enumerate(self.accordance['id_1'])
        }

    def id_images_paths_iterator(self) -> Iterator[str]:
        """
        ID images paths iterator
        :yield: path of id image
        """
        for id_image_file in self.id_images_path.iterdir():
            yield id_image_file.absolute()

    def images_paths_iterator(self) -> Iterator[str]:
        """
        Images iterator
        :yield: path of image
        """
        for image_file in self.destination_path.iterdir():
            if image_file.suffix.lower() == self.image_extension:
                yield image_file.absolute()
