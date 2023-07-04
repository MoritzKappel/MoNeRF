# -- coding: utf-8 --

"""Samplers/utils.py: Utilities for image and ray sampling routines."""

import torch

import Framework


class SamplerError(Framework.FrameworkError):
    """Raise for errors regarding a ray or image sampler."""


class SequentialSampler:
    def __init__(self, num_elements: int) -> None:
        self.num_elements: int = num_elements
        self.indices: torch.Tensor = torch.arange(self.num_elements)
        self.reset()

    def shuffle(self) -> None:
        pass

    def reset(self) -> None:
        self.current_id: int = 0
        self.shuffle()

    def get(self, num_samples: int) -> torch.Tensor:
        if num_samples > self.num_elements:
            raise SamplerError(f"cannot draw {num_samples} samples from {self.num_elements} elements")
        if self.current_id + num_samples > self.num_elements:
            self.reset()
        samples = self.indices[self.current_id:self.current_id + num_samples]
        self.current_id += num_samples
        return samples


class RandomSequentialSampler(SequentialSampler):

    def shuffle(self) -> None:
        self.indices: torch.Tensor = self.indices[torch.randperm(self.num_elements)]
