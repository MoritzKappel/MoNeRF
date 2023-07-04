# -- coding: utf-8 --
"""Framework.py: Contains all functions that are part of the framework's default setup process."""

__author__ = "Moritz Kappel"
__credits__ = ['Florian Hahlbohm, Marc Kassubeck']
__license__ = "MIT"
__maintainer__ = "Moritz Kappel"
__email__ = "kappel@cg.cs.tu-bs.de"
__status__ = "Development"

import ast
import os
import platform
import random
from argparse import ArgumentParser
from multiprocessing import set_start_method
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from munch import Munch

from Logger import Logger


class FrameworkError(Exception):
    """Raise in case of an exception during the framework's setup process."""
    def __init__(self, msg):
        super().__init__(msg)
        Logger.logError(f'({self.__class__.__name__}) {msg}')


class ConfigParameterList(Munch):

    def recursiveUpdate(self, other: 'ConfigParameterList'):
        # check type
        if not isinstance(other, ConfigParameterList):
            raise TypeError()
        # copy list
        other = other.copy()
        # recursively update sublists
        for key, value in [(i, other[i]) for i in other]:
            if isinstance(value, ConfigParameterList) and hasattr(self, key) and isinstance(self[key], ConfigParameterList):
                self[key].recursiveUpdate(value)
                del other[key]
        # update remaining contents
        self.update(other)


class Configurable:

    _configuration: ConfigParameterList = ConfigParameterList()

    def __init__(self, config_file_data_field: str) -> None:
        # gather default class configuration
        self.config_file_data_field: str = config_file_data_field
        instance_params = self.__class__._configuration.copy()
        # overwrite from config file
        if not hasattr(config, self.config_file_data_field):
            Logger.logWarning(f'data field {self.config_file_data_field} requested by class {self.__class__.__name__} is not available in config file. \
                              Using default config.')
        else:
            instance_params.recursiveUpdate(config[self.config_file_data_field])
        # assign to instance
        for param in instance_params:
            self.__dict__[param] = instance_params[param]

    @classmethod
    def getDefaultParameters(cls):
        return cls._configuration

    @staticmethod
    def configure(**params):
        newParams = ConfigParameterList(params)

        def configDecorator(cls):
            if not issubclass(cls, Configurable):
                raise FrameworkError(f'configure decorator must be applied to subclass of Configurable, but got {cls.__class__}')
            cls._configuration = cls._configuration.copy()
            cls._configuration.recursiveUpdate(newParams)
            return cls
        return configDecorator


def setup(require_custom_config: bool = False, config_path: str = None) -> None:
    """Performs a complete training setup based on the config file provided via comment line (or default)"""
    config_args: dict[str, str] = {}
    unknown_args = None
    if config_path is None:
        # parse arguments to retrieve config file location
        parser: ArgumentParser = ArgumentParser(prog='Framework')
        parser.add_argument(
            '-c', '--config', action='store', dest='config_path', default=None,
            metavar='path/to/config_file.yaml/', required=False, nargs='*',
            help='The yaml config file containing the project configuration.'
        )

        # parse extra args overwriting values in config (for wandb sweeps)
        args, unknown_args = parser.parse_known_args()
        config_path: str = args.config_path[0]
        config_args: dict[str, str] = {}
        for config_arg in args.config_path[1:]:
            try:
                key, value = config_arg.split('=')
                config_args[key] = value
            except ValueError:
                raise FrameworkError(f'invalid config overwrite argument syntax: "{config_arg}" (expected syntax: config_key=config_value).')

    # initialize config and call init methods
    loadConfig(config_path, require_custom_config, config_args)
    checkLibraryVersions()
    setupTorch()
    setRandomSeed()
    # return unused arguments to application
    return unknown_args


def loadConfig(config_path, require_custom_config, config_args) -> None:
    """Loads project configuration from .yaml file."""
    global config
    Logger.setMode(Logger.MODE_VERBOSE)
    if config_path is not None:
        try:
            yaml_dict: dict[str, Any] = yaml.unsafe_load(open(config_path))
            config = ConfigParameterList.fromDict(yaml_dict)
            Logger.setMode(config.GLOBAL.LOG_LEVEL)
            Logger.log(f'configuration file loaded: {config_path}')
            Logger.logDebug(config)
            config._path = os.path.abspath(config_path)
        except IOError:
            raise FrameworkError(f'invalid config file path: "{config_path}"')
    else:
        if require_custom_config:
            raise FrameworkError('missing config file, please provide a config file path using the "-c / --config" argument.')
        config = ConfigParameterList(GLOBAL=getDefaultGlobalConfig())
        Logger.setMode(config.GLOBAL.LOG_LEVEL)
        Logger.log('using default configuration')

    # override single config elements from command line arguments
    for config_arg, value in config_args.items():
        try:
            value = ast.literal_eval(value)
        except ValueError:
            pass  # keep value as string
        elements = config_arg.split('.')
        param_name = elements[-1]
        param_path = elements[:-1]
        target_munch = config
        try:
            for key in param_path:
                target_munch = getattr(target_munch, key)
        except AttributeError:
            raise FrameworkError(f'invalid config file key "{key}" in config overwrite argument "{config_arg}={value}"')
        setattr(target_munch, param_name, value)


def getDefaultGlobalConfig() -> ConfigParameterList:
    """Returns the default values of all global configuration parameters."""
    return ConfigParameterList(
        LOG_LEVEL=Logger.MODE_VERBOSE,
        DEFAULT_TENSOR_TYPE='torch.cuda.FloatTensor',
        EPS=1e-8,
        GPU_INDICES=[0],
        RANDOM_SEED=1618033989,
        ANOMALY_DETECTION=False
    )


def checkLibraryVersions() -> None:
    """Checks if versions of important libraries match the versions used for development and testing."""
    # library versions used during development and testing
    PYTHON_VERSION: str = '3.11.3'
    TORCH_VERSION: str = '2.0.1+cu118'
    CUDA_VERSION: str = '11.8'
    NUMPY_VERSION: str = '1.24.1'
    # compare to current version
    for lib_name, current_version, tested_version in zip(
            ['Python', 'Pytorch', 'CUDA', 'Numpy'],
            [platform.python_version(), torch.__version__, torch.version.cuda, np.__version__],
            [PYTHON_VERSION, TORCH_VERSION, CUDA_VERSION, NUMPY_VERSION]
    ):
        if current_version != tested_version:
            Logger.logWarning(f'current {lib_name} version: {current_version} (tested with {tested_version})')


def setRandomSeed() -> None:
    """Gets or sets the random seed for reproducibility (check https://pytorch.org/docs/stable/notes/randomness.html)"""
    # use a random seed if provided by config file
    if config.GLOBAL.RANDOM_SEED is None:
        config.GLOBAL.RANDOM_SEED = np.random.randint(0, 4294967295)
    Logger.logInfo(f'deterministic mode enabled using random seed: {config.GLOBAL.RANDOM_SEED}')
    torch.manual_seed(config.GLOBAL.RANDOM_SEED)
    random.seed(config.GLOBAL.RANDOM_SEED)
    np.random.seed(config.GLOBAL.RANDOM_SEED)  # legacy
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False  # should be set, but decreases performance


def setupTorch() -> None:
    """Initializes PyTorch by setting the default tensor type, GPU device and misc flags."""
    # set torch hub cache directory
    torch.hub.set_dir(str(Path(__file__).parents[1] / '.cache'))
    Logger.logInfo('initializing pytorch runtime')
    if config.GLOBAL.DEFAULT_TENSOR_TYPE is None:
        config.GLOBAL.DEFAULT_TENSOR_TYPE = 'torch.cuda.HalfTensor'
        Logger.logWarning(f'no default tensor type specified. using: {config.GLOBAL.DEFAULT_TENSOR_TYPE}')
    if 'cuda' in config.GLOBAL.DEFAULT_TENSOR_TYPE:
        if torch.cuda.is_available():
            Logger.logInfo('entering GPU mode')
            if config.GLOBAL.GPU_INDICES is None:
                config.GLOBAL.GPU_INDICES = [0]
                Logger.logWarning('GPU indices not specified. using default device (0)')
            valid_indices: list[int] = [i for i in config.GLOBAL.GPU_INDICES if i in range(torch.cuda.device_count())]
            for item in [i for i in config.GLOBAL.GPU_INDICES if i not in valid_indices]:
                Logger.logWarning(f'requested GPU index {item} not available on this machine!')
            config.GLOBAL.GPU_INDICES = valid_indices
            Logger.logInfo(f'using GPU(s): {str(config.GLOBAL.GPU_INDICES).replace(",", " (main),", 1)}')
            torch.cuda.set_device(config.GLOBAL.GPU_INDICES[0])
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.fastest = True
        else:
            config.GLOBAL.DEFAULT_TENSOR_TYPE = config.GLOBAL.DEFAULT_TENSOR_TYPE.replace('.cuda', '')
            config.GLOBAL.GPU_INDICES = []
            Logger.logWarning(
                f'GPU mode requested but not available on the device '
                f'(switching to CPU: {config.GLOBAL.DEFAULT_TENSOR_TYPE})'
            )
    else:
        Logger.logInfo('entering CPU mode')
    try:
        set_start_method('spawn')
    except RuntimeError:
        Logger.logInfo('multiprocessing start method already set')

    torch.autograd.set_detect_anomaly(config.GLOBAL.ANOMALY_DETECTION)
    # torch.utils.backcompat.broadcast_warning.enabled = True
    torch.set_default_tensor_type(config.GLOBAL.DEFAULT_TENSOR_TYPE)
    return


def setupWandb(project: str, entity: str, name: str) -> bool:
    """Sets up wandb for training visualization."""
    try:
        global wandb
        import wandb
        Logger.logInfo('setting up wandb')
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(project=project, entity=entity, name=name, config=config.toDict())
        Logger.logInfo(f'training logs will be available at: {wandb.run.url}')
    except ModuleNotFoundError:
        Logger.logWarning('wandb module not found: cannot log training information')
        return False
    return True
