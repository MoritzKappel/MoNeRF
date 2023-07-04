# -- coding: utf-8 --

"""Implementations.py: Dynamically provides access to the implemented datasets, methods and cuda extensions."""

import os
import sys
import importlib
from types import ModuleType
from typing import Type
from pathlib import Path

import Framework
from Logger import Logger

from Methods.Base.Model import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.Base.Trainer import BaseTrainer
from Methods.Base.utils import MethodError

from Datasets.Base import BaseDataset
from Datasets.utils import DatasetError

from CudaExtensions.utils import CudaExtensionError


class Methods:
    """A class containing all implemented methods"""
    path: Path = Path(__file__).resolve().parents[0] / 'Methods'
    options: tuple[str] = tuple([i.name for i in path.iterdir() if i.is_dir() and i.name not in ['Base', '__pycache__']])
    modules: dict[str, ModuleType] = {}

    @staticmethod
    def importMethod(method: str) -> None:
        if method in Methods.options:
            try:
                with setImportPaths():
                    m = importlib.import_module(f'Methods.{method}')
                Methods.modules[method] = m
                return
            except Exception as e:
                raise MethodError(f'failed to import method {method}:\n{e}')
        else:
            raise MethodError(f'requested invalid method type: {method}\navailable methods are: {Methods.options}')

    @staticmethod
    def getMethodClasses(method: str) -> tuple[Type[BaseModel], Type[BaseRenderer], Type[BaseTrainer]]:
        if method not in Methods.modules:
            Methods.importMethod(method)
        module: ModuleType = Methods.modules[method]
        return module.MODEL, module.RENDERER, module.TRAINING_INSTANCE

    @staticmethod
    def getModel(method: str, checkpoint: str = None, name: str = 'Default') -> BaseModel:
        """returns a model of the given type loaded with the provided checkpoint"""
        Logger.logInfo('creating model')
        model_class: Type[BaseModel] = Methods.getMethodClasses(method)[0]
        return model_class.load(checkpoint) if checkpoint is not None else model_class(name).build()

    @staticmethod
    def getRenderer(method: str, model: BaseModel) -> BaseRenderer:
        """returns a renderer for the specified method initialized with the given model instance"""
        Logger.logInfo('creating renderer')
        model_class: Type[BaseRenderer] = Methods.getMethodClasses(method)[1]
        return model_class(model)

    @staticmethod
    def getTrainingInstance(method: str, checkpoint: str = None) -> BaseTrainer:
        """returns a trainer of the given type loaded with the provided checkpoint"""
        Logger.logInfo('creating training instance')
        model_class: Type[BaseTrainer] = Methods.getMethodClasses(method)[2]
        if checkpoint is not None:
            return model_class.load(checkpoint)
        model = Methods.getModel(method=method, name=Framework.config.TRAINING.MODEL_NAME)
        renderer = Methods.getRenderer(method=method, model=model)
        return model_class(model=model, renderer=renderer)


class Datasets:
    """Dynamically loads and provides access to the implemented datasets."""
    path: Path = Path(__file__).resolve().parents[0] / 'Datasets'
    options: tuple[Path] = tuple([i.name.split('.')[0] for i in path.iterdir() if i.is_file() and i.name not in ['Base.py', 'utils.py', 'datasets.md']])
    loaded: dict[str, Type[BaseDataset]] = {}

    @staticmethod
    def importDataset(dataset_type: str) -> None:
        if dataset_type in Datasets.options:
            try:
                with setImportPaths():
                    m = importlib.import_module(f'Datasets.{dataset_type}')
                Datasets.loaded[dataset_type] = m.CustomDataset
            except Exception:
                raise DatasetError(f'failed to import dataset: {dataset_type}')
        else:
            raise DatasetError(f'requested invalid dataset type: {dataset_type}\navailable datasets are: {Datasets.options}')

    @staticmethod
    def getDatasetClass(dataset_type: str) -> Type[BaseDataset]:
        if dataset_type not in Datasets.loaded:
            Datasets.importDataset(dataset_type)
        return Datasets.loaded[dataset_type]

    @staticmethod
    def getDataset(dataset_type: str, path: str) -> BaseDataset:
        """Returns a dataset instance of the given type loaded from the given path."""
        dataset_class: Type[BaseDataset] = Datasets.getDatasetClass(dataset_type)
        return dataset_class(path)


class CudaExtensions:
    """Utility Class providing access to custom CUDA extensions"""
    path: Path = Path(__file__).resolve().parents[0] / 'CudaExtensions'
    options: tuple[str] = tuple([i.name for i in path.iterdir() if i.is_dir() and i.name not in ['__pycache__']])
    loaded: dict[str, ModuleType] = {}

    @staticmethod
    def installExtension(extension: str) -> None:
        if extension in CudaExtensions.options:
            Logger.logInfo(f'installing cuda extension: {extension}')
            command_string = f'cd "{CudaExtensions.path / extension}" && pip install -e .'
            if os.system(command_string) != 0:
                Logger.logError(f'failed to install cuda extenion "{extension}" using: "{command_string}"')
                raise CudaExtensionError(f'failed to install cuda extenion "{extension}"')
        else:
            Logger.logError(
                f'invalid cuda extension name: {extension}'
                f'\n available options are: {CudaExtensions.options}'
            )
            raise CudaExtensionError(f'invalid cuda extension name: {extension}')

    @staticmethod
    def installAll() -> None:
        for option in CudaExtensions.options:
            CudaExtensions.installExtension(option)

    @staticmethod
    def getExtension(extension_name: str) -> ModuleType:
        if extension_name not in CudaExtensions.loaded:
            CudaExtensions.importExtension(extension_name)
        return CudaExtensions.loaded[extension_name]

    @staticmethod
    def importExtension(extension_name: str) -> None:
        if extension_name in CudaExtensions.options:
            try:
                with setImportPaths():
                    m = importlib.import_module(f'CudaExtensions.{extension_name}')
                CudaExtensions.loaded[extension_name] = m
            except Exception:
                raise CudaExtensionError(
                        f'failed to import cuda extension: {extension_name}'
                        f'\n install the extension using: "./scripts/install.py -e {extension_name}"'
                    )
        else:
            raise CudaExtensionError(
                f'invalid cuda extension name: {extension_name}'
                f'\n available options are: {CudaExtensions.options}'
            )


class setImportPaths:
    """helper class adding source code directory to pythonpath during dynamic imports"""

    def __enter__(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

    def __exit__(self, *_):
        sys.path.pop(0)
