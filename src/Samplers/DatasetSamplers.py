# -- coding: utf-8 --

"""Samplers/DatasetSamplers.py: Samplers returning a batch of rays from a dataset."""
import torch

import Framework
from Cameras.utils import CameraProperties
from Datasets.Base import BaseDataset
from Samplers.ImageSamplers import ImageSampler
from Samplers.utils import SamplerError, RandomSequentialSampler, SequentialSampler


class DatasetSampler:

    def __init__(self,
                 dataset: BaseDataset,
                 random: bool = True,
                 img_sampler_cls: type[ImageSampler] | None = None) -> None:
        self.mode = dataset.mode
        self.id_sampler = RandomSequentialSampler(num_elements=len(dataset)) if random else SequentialSampler(num_elements=len(dataset))
        self.img_samplers = [img_sampler_cls(num_elements=(i.width * i.height)) for i in dataset] if img_sampler_cls else None

    def get(self, dataset: BaseDataset, ray_batch_size: int | None = None) -> dict[str, int | CameraProperties | None]:
        if dataset.mode != self.mode:
            raise SamplerError(f'DatasetSampler initialized for mode "{self.mode}" got dataset with active mode "{dataset.mode}"')
        sample_id = self.id_sampler.get(num_samples=1).item()
        camera_properties = dataset[sample_id]
        image_sampler = ray_ids = ray_batch = None
        if self.img_samplers and ray_batch_size is not None:
            image_sampler = self.img_samplers[sample_id]
            ray_ids = image_sampler.get(ray_batch_size)
            ray_batch = dataset.camera.setProperties(camera_properties).generateRays()[ray_ids].type(Framework.config.GLOBAL.DEFAULT_TENSOR_TYPE)
        return {
            'sample_id': sample_id,
            'camera_properties': camera_properties,
            'image_sampler': image_sampler,
            'ray_ids': ray_ids,
            'ray_batch': ray_batch
        }


class RayPoolSampler:
    def __init__(self,
                 dataset: BaseDataset,
                 img_sampler_cls: type[ImageSampler]) -> None:
        self.mode = dataset.mode
        all_rays = dataset.getAllRays()
        self.image_sampler = img_sampler_cls(num_elements=all_rays.shape[0])

    def get(self, dataset: BaseDataset, ray_batch_size: int) -> dict[str, None | ImageSampler | torch.Tensor]:
        if dataset.mode != self.mode:
            raise SamplerError(f'RayPoolSampler initialized for mode "{self.mode}" got dataset with active mode "{dataset.mode}"')
        sample_id = camera_properties = None
        ray_ids = self.image_sampler.get(ray_batch_size)
        ray_batch = dataset.getAllRays()[ray_ids]
        return {
            'sample_id': sample_id,
            'camera_properties': camera_properties,
            'image_sampler': self.image_sampler,
            'ray_ids': ray_ids,
            'ray_batch': ray_batch
        }
