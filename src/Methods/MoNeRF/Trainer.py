# -- coding: utf-8 --
"""MoNeRF/Trainer.py: MoNeRF training code."""

import torch
import torchvision
from random import randrange
from apex.optimizers import FusedAdam

import Framework
from Methods.Base.Trainer import BaseTrainer
from Cameras.NDC import NDCCamera
from Methods.Base.utils import preTrainingCallback, trainingCallback
from Methods.MoNeRF.Loss import MoNeRFLoss
from Datasets.Base import BaseDataset
from Methods.Base.utils import pseudoColorDepth
from Methods.MoNeRF.utils import logOccupancyGrids
from Samplers.DatasetSamplers import DatasetSampler
from Samplers.ImageSamplers import MultinomialImageSampler, RandomImageSampler


@Framework.Configurable.configure(
    RUN_VALIDATION=False,
    NUM_ITERATIONS=30000,
    BATCH_SIZE=8192,
    WARMUP_STEPS=1024,
    RANDOM_BG=True,
    DENSITY_GRID_UPDATE_INTERVAL=16,
    DENSITY_GRID_TEMPORAL_SAMPLES=20,
    LEARNING_RATE_NET=1.0e-3,
    LEARNING_RATE_ENCODING=1.0e-2,
    LEARNING_RATE_DECAY_FACTOR=1 / 30,
    ADAM_EPS=1e-15,
    LAMBDA_BG_ENTROPY=1e-2,
    LAMBDA_FLOW_MAGNITUDE=1e-3,
    LAMBDA_DISTORTION=0.0,
    ALPHA_RAY_SAMPLE_WEIGHT=1.0,
    LOGGING=Framework.ConfigParameterList(
        INTERVAL=-1,
        RENDER_OCCUPANCY_GRIDS=False,
        INDEX_VALIDATION=-1,
        INDEX_TRAINING=-1,
    )
)
class MoNeRFTrainer(BaseTrainer):
    """Defines MoNeRF training schedule"""

    def __init__(self, **kwargs) -> None:
        super(MoNeRFTrainer, self).__init__(**kwargs)
        self.loss: MoNeRFLoss = MoNeRFLoss(self.LAMBDA_BG_ENTROPY, self.LAMBDA_FLOW_MAGNITUDE, self.LAMBDA_DISTORTION, activate_logging=self.LOGGING.ACTIVATE_WANDB)
        self.createOptimizer()

    def createOptimizer(self) -> None:
        """Create optimizer, scheduler and gradient scaler using apex fused adam."""
        param_groups = [
            {'params': self.model.encoding_canonical.parameters(), 'lr': self.LEARNING_RATE_ENCODING},
            {'params': self.model.encoding_direction.parameters(), 'lr': self.LEARNING_RATE_ENCODING},
            {'params': self.model.deformation_net.parameters(), 'lr': self.LEARNING_RATE_NET},
            {'params': self.model.temporal_basis_net_encoding.parameters(), 'lr': self.LEARNING_RATE_ENCODING},
            {'params': self.model.deformation_net_encoding.parameters(), 'lr': self.LEARNING_RATE_ENCODING},
            {'params': self.model.density_net.parameters(), 'lr': self.LEARNING_RATE_NET},
            {'params': self.model.rgb_net.parameters(), 'lr': self.LEARNING_RATE_NET},
        ]
        if self.model.USE_TEMPORAL_BASIS:
            param_groups.append({'params': self.model.temporal_basis_net.parameters(), 'lr': self.LEARNING_RATE_NET}),
        self.optimizer = FusedAdam(param_groups, self.LEARNING_RATE_NET, eps=self.ADAM_EPS, betas=(0.9, 0.99))
        # self.optimizer = torch.optim.AdamW(param_groups, self.LEARNING_RATE_NET, eps=self.ADAM_EPS, betas=(0.9, 0.99), foreach=False, fused=True)  # significantly slower than apex in pt2.01
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.LRDecayPolicy(self.LEARNING_RATE_DECAY_FACTOR, self.NUM_ITERATIONS))
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    class LRDecayPolicy(object):
        """Defines exponential learning rate decay policy."""

        def __init__(self, ldr: float, lds: float) -> None:
            self.ldr: float = ldr
            self.lds: float = max(lds, 1)

        def __call__(self, iteration) -> float:
            """Calculates learning rate decay."""
            return self.ldr ** (iteration / self.lds)

    @preTrainingCallback(priority=100)
    @torch.no_grad()
    def initDensityGrid(self, _, dataset: 'BaseDataset') -> None:
        '''initialize temporal occupancy grid from dataset camera frustums'''
        if not isinstance(dataset.camera, NDCCamera):
            self.renderer.carveDensityGrid(dataset.train(), subtractive=False, use_alpha=False)

    @preTrainingCallback(priority=1000)
    @torch.no_grad()
    def initSampler(self, _, dataset: 'BaseDataset') -> None:
        '''initialize data sampling strategy for training and validation'''
        self.sampler_train = DatasetSampler(dataset=dataset.train(), random=(not self.model.USE_ZERO_CANONICAL),
                                            img_sampler_cls=MultinomialImageSampler if self.ALPHA_RAY_SAMPLE_WEIGHT < 1.0 else RandomImageSampler)
        if self.RUN_VALIDATION:
            self.sampler_val = DatasetSampler(dataset=dataset.eval(), random=True, img_sampler_cls=RandomImageSampler)

    @trainingCallback(priority=1000, start_iteration=0, iteration_stride='DENSITY_GRID_UPDATE_INTERVAL')
    @torch.no_grad()
    def updateDensityGrid(self, iteration: int, _) -> None:
        '''update temporal occupancy grid'''
        self.renderer.updateDensityGrid(self.DENSITY_GRID_TEMPORAL_SAMPLES, warmup=iteration < self.WARMUP_STEPS)

    @trainingCallback(priority=50)
    def processTrainingSample(self, iteration: int, dataset: 'BaseDataset') -> None:
        """performs a single training iteration."""
        # prepare training iteration
        self.model.train()
        self.loss.train()
        dataset.train()
        self.optimizer.zero_grad()
        # get dataset sample
        sample = self.sampler_train.get(dataset=dataset, ray_batch_size=self.BATCH_SIZE)
        ray_batch = sample['ray_batch']
        camera_properties = sample['camera_properties']
        with torch.cuda.amp.autocast(enabled=True):
            # render and calculate loss
            bg_color = torch.rand(3, device=ray_batch.device) if self.RANDOM_BG else None
            output = self.renderer.renderRays(
                rays=ray_batch,
                camera=dataset.camera,
                timestamp=camera_properties.timestamp,
                custom_bg_color=bg_color,
                train_mode=True)
            loss = self.loss(output, ray_batch, bg_color)
        # update model
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scheduler.step()
        self.scaler.update()
        # update sampler
        if self.ALPHA_RAY_SAMPLE_WEIGHT < 1.0:
            sample['image_sampler'].update(ray_ids=sample['ray_ids'], weights=torch.where(output['alpha'] < 1e-2, self.ALPHA_RAY_SAMPLE_WEIGHT, 1.0), constant_addend=1e-3)

    @trainingCallback(active='RUN_VALIDATION', priority=100)
    @torch.no_grad()
    def processValidationSample(self, _, dataset: 'BaseDataset') -> None:
        """performs a single validation iteration."""
        self.model.eval()
        self.loss.eval()
        dataset.eval()
        # get dataset sample
        sample = self.sampler_val.get(dataset=dataset, ray_batch_size=self.BATCH_SIZE)
        ray_batch = sample['ray_batch']
        camera_properties = sample['camera_properties']
        with torch.cuda.amp.autocast(enabled=True):
            # render and calculate loss
            output = self.renderer.renderRays(
                rays=ray_batch,
                camera=dataset.camera,
                timestamp=camera_properties.timestamp,
                train_mode=True)
            self.loss(output, ray_batch, None)

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
            if len(dataset) > 0:
                if index < 0:
                    index: int = randrange(len(dataset))
                sample = dataset[index]
                dataset.camera.setProperties(sample)
                outputs = self.renderer.renderImage(
                    camera=dataset.camera,
                    timestamp=None,
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
        # visualize current occupancy grids as point clouds
        if self.LOGGING.RENDER_OCCUPANCY_GRIDS:
            logOccupancyGrids(self.renderer, iteration, dataset, 'occupancy grid')
        # commit current step
        Framework.wandb.log(data={}, commit=True)
