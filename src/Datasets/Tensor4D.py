# -- coding: utf-8 --

"""
Datasets/Tensor4D.py: Provides a dataset class for Tensor4D scenes.
Data available at https://github.com/DSaurus/Tensor4D (last accessed 2023-05-25).
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import CameraProperties, CoordinateSystemTransformations
from Datasets.Base import BaseDataset
from Datasets.utils import list_sorted_files, applyBGColor, DatasetError, loadImagesParallel
from Logger import Logger


@Framework.Configurable.configure(
    PATH='dataset/tensor4d/boxing_v12',
    TEST_CAMERA_ID=None,
    MONOCULARIZE=False,
    BACKGROUND_COLOR=[0.0, 0.0, 0.0],
)
class CustomDataset(BaseDataset):
    """Dataset class for Tensor4D scenes."""

    def __init__(self, path: str) -> None:
        super().__init__(
            path,
            PerspectiveCamera(0.1, 2.0),
            CoordinateSystemTransformations.LEFT_HAND
        )

    # loading code adapted from https://github.com/DSaurus/Tensor4D/blob/main/models/dataset.py
    def load(self) -> dict[str, list[CameraProperties]]:
        """Loads the dataset into a dict containing lists of CameraProperties for training, evaluation, and testing."""
        data: dict[str, list[CameraProperties]] = {subset: [] for subset in self.subsets}
        scale_factor_intrinsics = self.IMAGE_SCALE_FACTOR if self.IMAGE_SCALE_FACTOR is not None else 1.0
        metadata_filepath: Path = self.dataset_path / 'cameras_sphere.npz'
        try:
            metadata_file = dict(np.load(str(metadata_filepath)))
        except IOError:
            raise DatasetError(f'Invalid dataset metadata file path "{metadata_filepath}"')
        images_list = list_sorted_files(self.dataset_path / 'image')
        world_mats = [torch.as_tensor(metadata_file[f'world_mat_{idx}'], dtype=torch.float32) for idx in range(len(images_list))]
        scale_mats = [torch.as_tensor(metadata_file[f'scale_mat_{idx}'], dtype=torch.float32) for idx in range(len(images_list))]
        fid_list = [metadata_file[f'fid_{idx}'].item() for idx in range(len(images_list))]
        fid_count = {i: fid_list.count(i) for i in fid_list}
        num_frames = len(fid_count)
        num_cameras = fid_count[0]
        data_ids_train = list(range(len(images_list)))
        data_ids_test = []
        if not all(fid_count[i] == num_cameras for i in fid_count):
            raise DatasetError(f'expected equal amount of frames for all cameras, got: {fid_count}')
        if self.TEST_CAMERA_ID is not None:
            if self.TEST_CAMERA_ID >= num_cameras:
                raise DatasetError(f'TEST_CAMERA_ID out of bounds: requested id {self.TEST_CAMERA_ID} (dataset contains {num_cameras} cameras)')
            data_ids_test = data_ids_train[self.TEST_CAMERA_ID::num_cameras]
            data_ids_train = list(set(data_ids_train) - set(data_ids_test))
            num_cameras -= 1
        if self.MONOCULARIZE:
            sampled_ids = random.choices(list(range(num_cameras)), k=num_frames)
            data_ids_train = [data_ids_train[(i * num_cameras) + sampled_ids[i]] for i in range(len(sampled_ids))]
        # load camera parameters
        intrinsics_all, pose_all = self.loadCameraParams(scale_mats, world_mats)
        # load images
        image_filenames = [str(self.dataset_path / 'image' / image_name) for image_name in images_list]
        mask_filenames = [str(self.dataset_path / 'mask' / image_name) for image_name in images_list]
        rgbs, _ = loadImagesParallel(image_filenames, self.IMAGE_SCALE_FACTOR, num_threads=4, desc='images')
        alphas, _ = loadImagesParallel(mask_filenames, self.IMAGE_SCALE_FACTOR, num_threads=4, desc='masks')
        for split, ids in zip(('train', 'val', 'test'), (data_ids_train, data_ids_test, data_ids_test)):
            # create split CameraProperties objects
            for i in Logger.logProgressBar(ids, desc=f'creating {split} set', leave=False):
                rgb = rgbs[i]
                alpha = alphas[i][:1]
                rgb = applyBGColor(rgb, alpha, self.camera.background_color)
                data[split].append(CameraProperties(
                    width=rgb.shape[2],
                    height=rgb.shape[1],
                    rgb=rgb,
                    alpha=alpha,
                    c2w=pose_all[i].clone(),
                    focal_x=intrinsics_all[i][0, 0] * scale_factor_intrinsics,
                    focal_y=intrinsics_all[i][1, 1] * scale_factor_intrinsics,
                    principal_offset_x=(intrinsics_all[i][0, 2] * scale_factor_intrinsics) - (rgb.shape[2] * 0.5),
                    principal_offset_y=(intrinsics_all[i][1, 2] * scale_factor_intrinsics) - (rgb.shape[1] * 0.5),
                    timestamp=fid_list[i] / max(fid_list)
                ))
        return data

    @staticmethod
    def loadCameraParams(scale_mats: list[torch.Tensor],
                         world_mats: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Converts scale and world matrices to intrinsic and extrinsic camera parameters."""
        intrinsics_all = []
        pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            K, R, t, *_ = cv2.decomposeProjectionMatrix((world_mat @ scale_mat)[:3, :4].cpu().numpy())
            K, R, t = torch.as_tensor(K), torch.as_tensor(R), torch.as_tensor(t)
            K = K / K[2, 2]
            intrinsics = torch.eye(4, dtype=torch.float32)
            intrinsics[:3, :3] = K
            pose = torch.eye(4, dtype=torch.float32)
            pose[:3, :3] = R.t()
            pose[:3, 3] = (t[:3] / t[3])[:, 0]
            intrinsics_all.append(intrinsics)
            pose_all.append(pose)
        return intrinsics_all, pose_all
