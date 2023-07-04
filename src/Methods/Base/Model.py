# -- coding: utf-8 --

"""Base/Model.py: Abstract base class for scene models."""

from abc import ABC, abstractmethod
import datetime
from pathlib import Path
from typing import Callable
import torch

import Framework
from Methods.Base.utils import ModelError
from Logger import Logger

class BaseModel(Framework.Configurable, ABC, torch.nn.Module):
    """Defines the basic PyTorch neural model."""

    def __init__(self, name: str = None) -> None:
        Framework.Configurable.__init__(self, 'MODEL')
        ABC.__init__(self)
        torch.nn.Module.__init__(self)
        self.model_name: str = name if name is not None else 'Default'
        self.creation_date: str = f'{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}'
        self.num_iterations_trained: int = 0

    @abstractmethod
    def build(self) -> 'BaseModel':
        """Implementations build the neural model structure (e.g. assign layers)."""
        return self

    def forward(self) -> None:
        """Invalidates forward passes of model as all models are executed exclusively through renderers."""
        Logger.logError('Model cannot be executed directly. Use a Renderer instead.')

    def __repr__(self) -> str:
        """Returns string representation of the model's metadata."""
        params_string = ''
        additional_parameters = type(self).getDefaultParameters().keys()
        if additional_parameters:
            params_string += '\n\t Additional parameters:'
            for param in additional_parameters:
                params_string += f'\n\t\t{param}: {self.__dict__[param]}'
        return f'<instance of class: {self.__class__.__name__} \n' \
               f'\t model name: {self.model_name} \n' \
               f'\t created on: {self.creation_date} \n' \
               f'\t trained for: {self.num_iterations_trained} iterations{params_string}\n>'

    @classmethod
    def load(cls, checkpoint_name: str | None,
             map_location: Callable = lambda storage, location: storage) -> 'BaseModel':
        """Loads a saved model from '.pt' file."""
        if checkpoint_name is None or checkpoint_name.split('.')[-1] != 'pt':
            raise ModelError(f'Invalid model checkpoint: "{checkpoint_name}"')
        try:
            # load checkpoint
            checkpoint_path = Path(__file__).resolve().parents[3] / 'output'
            checkpoint = torch.load(checkpoint_path / checkpoint_name, map_location=map_location)
            # create new model
            model = cls()
            # load model configuration
            for param in ['model_name', 'creation_date', 'num_iterations_trained'] + list(cls.getDefaultParameters().keys()):
                model.__dict__[param] = checkpoint[param]
            # build the model
            model.build()
            # load model parameters
            model.load_state_dict(checkpoint['model_state_dict'])
            # cast model to current data type
            # model.type(Framework.config.GLOBAL.DEFAULT_TENSOR_TYPE)
        except IOError as e:
            raise ModelError(f'failed to load model from file: "{e}"')
        return model

    def save(self, path: Path) -> None:
        """Saves the current model as '.pt' file."""
        try:
            checkpoint = {'model_state_dict': self.state_dict()}
            for param in ['model_name', 'creation_date', 'num_iterations_trained'] + list(type(self).getDefaultParameters().keys()):
                checkpoint[param] = self.__dict__[param]
            torch.save(checkpoint, path)
        except IOError as e:
            Logger.logWarning(f'failed to save model: "{e}"')

    def exportTorchScript(self, path: Path) -> None:
        """Exports model as torch script module (e.g. for execution in c++)"""
        try:
            script_module = torch.jit.script(self)
            script_module.save(str(path))
        except IOError as e:
            Logger.logWarning(f'failed to generate script module: "{e}"')

    def numModuleParameters(self, trainable_only=False) -> int:
        """Returns the model's number of parameters."""
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))
