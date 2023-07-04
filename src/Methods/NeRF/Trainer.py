# -- coding: utf-8 --

"""NeRF/Trainer.py: Implementation of the trainer for the vanilla (i.e. original) NeRF method."""

from random import randrange

import torch
import torchvision

import Framework
from Cameras.utils import CameraProperties
from Datasets.Base import BaseDataset
from Methods.Base.Trainer import BaseTrainer
from Methods.Base.utils import preTrainingCallback, pseudoColorDepth, trainingCallback
from Methods.NeRF.Loss import NeRFLoss
from Samplers.DatasetSamplers import DatasetSampler, RayPoolSampler
from Samplers.ImageSamplers import RandomImageSampler


@Framework.Configurable.configure(
    NUM_ITERATIONS=500000,
    BATCH_SIZE=1024,
    SAMPLE_SINGLE_IMAGE=True,
    RUN_VALIDATION=False,
    DENSITY_RANDOM_NOISE_STD=0.0,
    ADAM_BETA_1=0.9,
    ADAM_BETA_2=0.999,
    LEARNINGRATE=5.0e-04,
    LEARNINGRATE_DECAY_RATE=0.1,
    LEARNINGRATE_DECAY_STEPS=500000,
    LAMBDA_COLOR_LOSS=1.0,
    LAMBDA_ALPHA_LOSS=0.0,
    LOGGING=Framework.ConfigParameterList(
        INTERVAL=-1,
        INDEX_VALIDATION=-1,
        INDEX_TRAINING=-1,
    ),
)
class NeRFTrainer(BaseTrainer):
    """Defines the trainer for the vanilla (i.e. original) NeRF method."""

    def __init__(self, **kwargs) -> None:
        super(NeRFTrainer, self).__init__(**kwargs)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.LEARNINGRATE, betas=(self.ADAM_BETA_1, self.ADAM_BETA_2),
            eps=Framework.config.GLOBAL.EPS
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self.LRDecayPolicy(self.LEARNINGRATE_DECAY_RATE, self.LEARNINGRATE_DECAY_STEPS),
            last_epoch=self.model.num_iterations_trained - 1
        )
        self.loss = NeRFLoss(self.LAMBDA_COLOR_LOSS, self.LAMBDA_ALPHA_LOSS, self.LOGGING.ACTIVATE_WANDB)

    class LRDecayPolicy(object):
        """Defines a decay policy for the learning rate."""

        def __init__(self, ldr: float, lds: float) -> None:
            self.ldr: float = ldr
            self.lds: float = lds

        def __call__(self, iteration) -> float:
            """Calculates learning rate decay."""
            return self.ldr ** (iteration / self.lds)

    @preTrainingCallback(priority=1000)
    @torch.no_grad()
    def initSampler(self, _, dataset: 'BaseDataset') -> None:
        sampler_cls = DatasetSampler if self.SAMPLE_SINGLE_IMAGE else RayPoolSampler
        self.sampler_train = sampler_cls(dataset=dataset.train(), random=True, img_sampler_cls=RandomImageSampler)
        if self.RUN_VALIDATION:
            self.sampler_val = sampler_cls(dataset=dataset.eval(), random=True, img_sampler_cls=RandomImageSampler)

    @trainingCallback(priority=50)
    def processTrainingSample(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Defines a callback which is executed every iteration to process a training sample."""
        # set modes
        self.model.train()
        self.loss.train()
        dataset.train()
        # sample ray batch
        ray_batch: torch.Tensor = self.sampler_train.get(dataset=dataset, ray_batch_size=self.BATCH_SIZE)['ray_batch']
        # update model
        self.optimizer.zero_grad()
        outputs = self.renderer.renderRays(
            rays=ray_batch,
            camera=dataset.camera,
            return_samples=False,
            randomize_samples=True,
            random_noise_densities=self.DENSITY_RANDOM_NOISE_STD
        )
        loss: torch.Tensor = self.loss(outputs, ray_batch)
        loss.backward()
        self.optimizer.step()
        # update learning rate
        self.lr_scheduler.step()

    @trainingCallback(priority=100, active='RUN_VALIDATION')
    @torch.no_grad()
    def processValidationSample(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Defines a callback which is executed every iteration to process a validation sample."""
        self.model.eval()
        self.loss.eval()
        dataset.eval()
        ray_batch: torch.Tensor = self.sampler_val.get(dataset=dataset, ray_batch_size=self.BATCH_SIZE)['ray_batch']
        outputs = self.renderer.renderRays(
            rays=ray_batch,
            camera=dataset.camera,
            return_samples=False,
            randomize_samples=False,
            random_noise_densities=0.0
        )
        self.loss(outputs, ray_batch)

    @trainingCallback(active='LOGGING.ACTIVATE_WANDB', priority=500, iteration_stride='LOGGING.INTERVAL')
    @torch.no_grad()
    def logTraining(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Logs all losses and visualizes training and validation samples using Weights & Biases."""
        # log losses
        self.loss.log(iteration)
        # reset loss accumulation
        self.loss.reset()

        # visualize samples
        for mode, index, name in zip([dataset.train, dataset.eval],
                                     [self.LOGGING.INDEX_TRAINING, self.LOGGING.INDEX_VALIDATION],
                                     ['training', 'validation']):
            mode()
            if index < 0:
                index: int = randrange(len(dataset))
            sample = dataset[index]
            dataset.camera.setProperties(sample)
            outputs = self.renderer.renderImage(
                camera=dataset.camera,
                to_chw=True
            )
            image_gt: torch.Tensor = sample.rgb.type(Framework.config.GLOBAL.DEFAULT_TENSOR_TYPE) \
                if sample.rgb is not None \
                else dataset.camera.background_color[:, None, None].expand(-1, sample.height, sample.width)
            pseudo_color_depth = pseudoColorDepth(
                color_map='TURBO',
                depth=outputs['depth'],
                near_far=None,  # near_far=(dataset.camera.near_plane, dataset.camera.far_plane)
                alpha=outputs['alpha']
            )
            images = [
                image_gt,
                outputs['rgb'],
                pseudo_color_depth,
                outputs['alpha'].expand(image_gt.shape)
            ]
            image: torch.Tensor = torchvision.utils.make_grid(
                tensor=images,
                nrow=4,
                scale_each=True
            )
            Framework.wandb.log(
                data={name: Framework.wandb.Image(image, caption='GT, rgb, depth, alpha')},
                step=iteration
            )
        # commit current step
        Framework.wandb.log(data={}, commit=True)
