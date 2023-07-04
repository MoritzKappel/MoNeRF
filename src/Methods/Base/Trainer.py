# -- coding: utf-8 --

"""Base/Trainer.py: Implementation of the basic trainer for models."""

import os
from pathlib import Path
import git
from operator import attrgetter
from typing import Callable
import datetime
import shutil
import inspect
import torch

import Framework
from Logger import Logger
from Datasets.Base import BaseDataset
from Methods.Base.Model import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.Base.utils import CallbackTimer, CheckpointError, TrainingError, postTrainingCallback, trainingCallback


@Framework.Configurable.configure(
    LOAD_CHECKPOINT=None,
    MODEL_NAME='Default',
    NUM_ITERATIONS=1,
    ACTIVATE_TIMING=False,
    REQUIRE_GIT_COMMIT=False,
    BACKUP=Framework.ConfigParameterList(
        FINAL_CHECKPOINT=True,
        RENDER_TESTSET=True,
        RENDER_TRAINSET=False,
        RENDER_VALSET=False,
        INTERVAL=-1,
        TRAINING_STATE=False
    ),
    LOGGING=Framework.ConfigParameterList(
        ACTIVATE_WANDB=False,
        WANDB_ENTITY=None,
        WANDB_PROJECT=None,
    )
)
class BaseTrainer(Framework.Configurable, torch.nn.Module):
    """Defines the basic trainer used to train a model."""

    def __init__(self, model: BaseModel, renderer: BaseRenderer) -> None:
        Framework.Configurable.__init__(self, 'TRAINING')
        torch.nn.Module.__init__(self)
        self.model: BaseModel = model
        self.renderer: BaseRenderer = renderer
        # check if training code was committed
        self.checkGitCommit()
        # setup training logging
        if self.LOGGING.ACTIVATE_WANDB:
            self.LOGGING.ACTIVATE_WANDB = Framework.setupWandb(project=self.LOGGING.WANDB_PROJECT, entity=self.LOGGING.WANDB_ENTITY, name=self.model.model_name)
        # create output and checkpoint directory
        self.output_directory = Path(__file__).resolve().parents[3] / 'output' / f'{self.model.model_name}({self.model.creation_date})'
        self.checkpoint_directory = self.output_directory / 'checkpoints'
        Logger.logInfo(f'creating output directory: {self.output_directory}')
        os.makedirs(str(self.checkpoint_directory), exist_ok=True)
        shutil.copy2(Framework.config._path, str(self.output_directory / 'training_config.yaml'))

    @classmethod
    def load(cls, checkpoint_name: str) -> 'BaseTrainer':
        """Loads a saved training checkpoint from a '.train' file."""
        if checkpoint_name is None or checkpoint_name.split('.')[-1] != 'train':
            raise CheckpointError(f'Invalid checkpoint name "{checkpoint_name}"')
        try:
            checkpoint_path = Path(__file__).resolve().parents[3] / 'output'
            training_instance = torch.load(checkpoint_path / checkpoint_name)
        except IOError as e:
            raise CheckpointError(f'Failed to load checkpoint "{e}"')
        return training_instance

    def save(self, path: Path) -> None:
        """Saves the current model in a '.train' file at the given path."""
        try:
            torch.save(self, path)
        except IOError as e:
            raise CheckpointError(f'Failed to save checkpoint "{e}"')

    def checkGitCommit(self) -> None:
        """If activated, checks for uncommitted code changes in current git repository."""
        if self.REQUIRE_GIT_COMMIT:
            Logger.logInfo('Checking git status')
            parent_path = Path(__file__).resolve().parents[3]
            try:
                repo = git.Repo(parent_path)
                if repo.is_dirty(untracked_files=True):
                    raise TrainingError('detected uncommitted changes in your git repository. Commit your changes or disable git commit checking in the configuration file.')
            except git.InvalidGitRepositoryError:
                Logger.logInfo(f'parent directory does not contain a git repository: "{parent_path}"')

    def renderDataset(self, dataset: BaseDataset, verbose: bool = True):
        if self.BACKUP.RENDER_TESTSET:
            self.renderer.renderSubset(self.output_directory, dataset.test(), calculate_metrics=True, verbose=verbose)
        if self.BACKUP.RENDER_TRAINSET:
            self.renderer.renderSubset(self.output_directory, dataset.train(), calculate_metrics=False, verbose=verbose)
        if self.BACKUP.RENDER_VALSET:
            self.renderer.renderSubset(self.output_directory, dataset.eval(), calculate_metrics=False, verbose=verbose)

    @trainingCallback(priority=1, start_iteration='BACKUP.INTERVAL', iteration_stride='BACKUP.INTERVAL')
    def saveIntermediateCheckpoint(self, iteration: int, dataset: BaseDataset) -> None:
        """Creates an intermediate checkpoint at the current iteration."""
        self.model.save(self.checkpoint_directory / f'{iteration:07d}.pt')
        if self.BACKUP.TRAINING_STATE:
            self.save(self.checkpoint_directory / f'{iteration:07d}.train')
        self.renderDataset(dataset=dataset, verbose=False)

    @postTrainingCallback(active='BACKUP.FINAL_CHECKPOINT', priority=1)
    def saveFinalCheckpoints(self, _, dataset: BaseDataset) -> None:
        """Creates a final checkpoint before exiting the training loop."""
        Logger.logInfo('creating final model and training checkpoints')
        self.model.save(self.checkpoint_directory / 'final.pt')
        if self.BACKUP.TRAINING_STATE:
            self.save(self.checkpoint_directory / 'final.train')
        self.renderDataset(dataset=dataset)

    def logTiming(self) -> None:
        """Writes runtimes to file if activated."""
        if self.ACTIVATE_TIMING:
            Logger.logInfo('writing timings')
            training_time = 0
            with open(str(self.output_directory / 'timings.txt'), 'w') as f:
                for callback in self.listCallbacks():
                    values = callback.timer.getValues()
                    if callback.callback_type == 0:
                        training_time += datetime.timedelta(seconds=values[0]).total_seconds()
                    f.write(f'{callback.__name__}:\n\t'
                            f'Total execution time: {datetime.timedelta(seconds=round(values[0]))}\n\t'
                            f'Time per iteration [ms]: {values[1] * 1000:.2f}\n\t'
                            f'Number of iterations: {values[2]}\n\n'
                            )
                f.write(f'Time:{training_time}')

    def run(self, dataset: 'BaseDataset') -> None:
        """Trains the model for the specified amount of iterations executing all callbacks along the way."""
        Logger.log(f'starting training for model: {self.model.model_name}')
        # main training loop
        starting_iteration = iteration = self.model.num_iterations_trained
        for callback in self.gatherCallbacks(-1):
            with callback.timer:
                callback(self, starting_iteration, dataset)
        try:
            training_callbacks = self.gatherCallbacks(0)
            for iteration in Logger.logProgressBar(range(starting_iteration, self.NUM_ITERATIONS), desc='training', miniters=10):
                for callback in training_callbacks:
                    if (
                        (callback.start_iteration is not None and iteration < callback.start_iteration) or
                        (callback.end_iteration is not None and iteration >= callback.end_iteration) or
                        (callback.iteration_stride is not None and (iteration - (callback.start_iteration or 0)) % callback.iteration_stride != 0)
                    ):
                        continue
                    with callback.timer:
                        callback(self, iteration, dataset)
                self.model.num_iterations_trained += 1
        except KeyboardInterrupt:
            Logger.logWarning('training manually interrupted')
        for callback in self.gatherCallbacks(1):
            with callback.timer:
                callback(self, iteration + 1, dataset)
        self.logTiming()
        if self.LOGGING.ACTIVATE_WANDB:
            Framework.wandb.finish()
        Logger.log('training finished successfully')

    def gatherCallbacks(self, callback_type: int) -> list[Callable]:
        """Returns a list of all training callback functions of the requested type, and replaces config strings with values"""
        all_callbacks = self.listCallbacks()
        requested_callbacks = []
        for callback in all_callbacks:
            if not callback.callback_type == callback_type:
                continue
            for attr in ['active', 'start_iteration', 'end_iteration', 'iteration_stride']:
                try:
                    attr_value = getattr(callback, attr)
                    if isinstance(attr_value, str):
                        setattr(callback, attr, attrgetter(attr_value)(self))
                except AttributeError:
                    raise TrainingError(f'invalid config parameter for callback function "{callback.__name__}" field "{attr}":  \
                                        Class {self.__class__.__name__} has no config parameter {getattr(callback, attr)}')
            if self.ACTIVATE_TIMING and not isinstance(callback, CallbackTimer):
                callback.timer = CallbackTimer()
            if callback.iteration_stride is not None and callback.iteration_stride <= 0:
                callback.active = False
            if callback.active:
                requested_callbacks.append(callback)
        requested_callbacks.sort(key=lambda c: c.priority, reverse=True)
        return requested_callbacks

    def listCallbacks(self) -> list[Callable]:
        """Returns all registered training callback functions"""
        for member in inspect.getmembers(self.__class__, predicate=inspect.isfunction):
            if hasattr(member[1], 'callback_type'):
                yield member[1]
