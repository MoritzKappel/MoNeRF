# -- coding: utf-8 --

"""HierarchicalNeRF/Trainer.py: Implementation of the trainer for the hierarchical NeRF method."""

from random import randrange

import torch
import torchvision

import Framework
from Datasets.Base import BaseDataset
from Methods.Base.utils import pseudoColorDepth, trainingCallback
from Methods.HierarchicalNeRF.Loss import HierarchicalNeRFLoss
from Methods.NeRF.Trainer import NeRFTrainer


class HierarchicalNeRFTrainer(NeRFTrainer):
    """Defines the trainer for the hierarchical NeRF method."""

    def __init__(self, **kwargs) -> None:
        super(HierarchicalNeRFTrainer, self).__init__(**kwargs)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.LEARNINGRATE,
            betas=(self.ADAM_BETA_1, self.ADAM_BETA_2)
            # eps=Framework.config.GLOBAL.EPS
        )
        for param_group in self.optimizer.param_groups:
            param_group['capturable'] = True  # Hacky fix for PT 1.12 bug
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self.LRDecayPolicy(self.LEARNINGRATE_DECAY_RATE, self.LEARNINGRATE_DECAY_STEPS),
            last_epoch=self.model.num_iterations_trained - 1
        )
        self.loss = HierarchicalNeRFLoss(self.LAMBDA_COLOR_LOSS, self.LAMBDA_ALPHA_LOSS, self.LOGGING.ACTIVATE_WANDB)
        self.renderer.RENDER_COARSE = True

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
            pseudo_color_depth_coarse = pseudoColorDepth(
                color_map='TURBO',
                depth=outputs['depth_coarse'],
                near_far=None,  # near_far=(dataset.camera.near_plane, dataset.camera.far_plane)
                alpha=outputs['alpha_coarse']
            )
            pseudo_color_depth = pseudoColorDepth(
                color_map='TURBO',
                depth=outputs['depth'],
                near_far=None,  # near_far=(dataset.camera.near_plane, dataset.camera.far_plane)
                alpha=outputs['alpha']
            )
            images = [
                image_gt,
                outputs['rgb_coarse'],
                pseudo_color_depth_coarse,
                outputs['alpha_coarse'].expand(image_gt.shape),
                outputs['rgb'],
                pseudo_color_depth,
                outputs['alpha'].expand(image_gt.shape)
            ]
            image: torch.Tensor = torchvision.utils.make_grid(
                tensor=images,
                nrow=7,
                scale_each=True
            )
            Framework.wandb.log(
                data={name: Framework.wandb.Image(
                    image,
                    caption='GT, rgb_coarse, depth_coarse, alpha_coarse, rgb, depth, alpha'
                )},
                step=iteration
            )
        # commit current step
        Framework.wandb.log(data={}, commit=True)
