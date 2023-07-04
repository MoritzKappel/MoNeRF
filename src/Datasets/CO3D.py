# -- coding: utf-8 --

"""
Datasets/NeRF.py: Provides a dataset class for Common Objects in 3D (CO3D) scenes.
Data available at https://github.com/facebookresearch/co3d (last accessed 2023-05-25).
"""

import gzip
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision import io

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import CameraProperties, CoordinateSystemTransformations
from Datasets.Base import BaseDataset
from Datasets.utils import applyBGColor, DatasetError, loadImagesParallel, getParallelLoadIterator, \
    applyImageScaleFactor
from Logger import Logger


@Framework.Configurable.configure(
    PATH='dataset/co3d',
    IMAGE_SCALE_FACTOR=0.5,
    NORMALIZE_RADIUS=2.0,
    NORMALIZE_RECENTER=True,
    CATEGORY='hydrant',
    SEQUENCE='250_26779_55023',
    SUBSET_NAME='fewview_dev',
    MASK_BACKGROUND_W_ALPHA=False,
    VAL_STEP=0,
)
class CustomDataset(BaseDataset):
    """Dataset class for Common Objects in 3D (CO3D) scenes."""

    def __init__(self, path: str) -> None:
        super().__init__(
            path,
            PerspectiveCamera(0.0, 2.0),
            CoordinateSystemTransformations.PYTORCH3D
        )
        if self.NORMALIZE_RADIUS > 0:
            self.camera.near_plane = 1.0e-2
            self.camera.far_plane = min(self.camera.far_plane, self.NORMALIZE_RADIUS * 2.1)
        else:
            Logger.logWarning(f'using far plane extracted from depth images: {self.camera.far_plane}')

    def load(self) -> dict[str, list[CameraProperties]]:
        """Loads the dataset into a dict containing lists of CameraProperties for training, evaluation, and testing."""
        data: list[CameraProperties | None] = []
        frame_annotations_filepath: Path = self.dataset_path / self.CATEGORY / 'frame_annotations.jgz'
        try:
            with gzip.GzipFile(frame_annotations_filepath, "rb") as f:
                frame_annotations_category: list[dict[str, Any]] = json.loads(f.read().decode("utf8"))
        except IOError:
            raise DatasetError(f'Invalid dataset frame annotations file path "{frame_annotations_filepath}"')
        frame_annotations = sorted([
            frame_annotation for frame_annotation in frame_annotations_category if frame_annotation['sequence_name'] == self.SEQUENCE
        ], key=lambda x: x['frame_number'])
        # load images, masks, and depths
        image_filenames = [f'{self.dataset_path}/{frame_annotation["image"]["path"]}' for frame_annotation in frame_annotations]
        rgbs, alphas = loadImagesParallel(image_filenames, self.IMAGE_SCALE_FACTOR, num_threads=4, desc='images')
        mask_filenames = [f'{self.dataset_path}/{frame_annotation["mask"]["path"]}' for frame_annotation in frame_annotations]
        alphas, _ = loadImagesParallel(mask_filenames, self.IMAGE_SCALE_FACTOR, num_threads=4, desc='masks')
        depths = self.loadCO3DDepthsParallel(frame_annotations, num_threads=4)
        # create CameraProperties objects
        for i, frame_annotation in enumerate(frame_annotations):
            if frame_annotation['mask']['mass'] > 1:
                rgb = rgbs[i]
                alpha = alphas[i]
                depth = depths[i]
                # apply background color to alpha
                if self.MASK_BACKGROUND_W_ALPHA:
                    rgb = applyBGColor(rgb, alpha, self.camera.background_color)
                # load intrinsics
                scale = torch.tensor([rgb.shape[2], rgb.shape[1]], dtype=torch.float32).min() * 0.5
                focal_length = torch.as_tensor(frame_annotation['viewpoint']['focal_length'])
                focal_length_px = focal_length * scale
                principal_point = torch.as_tensor(frame_annotation['viewpoint']['principal_point'])
                principal_point_px = -principal_point * scale
                # load extrinsics
                w2c = torch.hstack((
                    torch.vstack((
                        torch.as_tensor(frame_annotation['viewpoint']['R'][0]),
                        torch.as_tensor(frame_annotation['viewpoint']['R'][1]),
                        torch.as_tensor(frame_annotation['viewpoint']['R'][2]),
                        torch.as_tensor(frame_annotation['viewpoint']['T'])
                    )),
                    torch.tensor([[0, 0, 0, 1]]).t()
                )).t()
                # insert data into target data structure
                data.append(CameraProperties(
                    width=rgb.shape[2],
                    height=rgb.shape[1],
                    rgb=rgb,
                    alpha=alpha,
                    depth=depth,
                    w2c=w2c,
                    focal_x=focal_length_px[0],
                    focal_y=focal_length_px[1],
                    principal_offset_x=principal_point_px[0],
                    principal_offset_y=principal_point_px[1],
                ))
            else:
                data.append(None)
        # find near and far plane
        max_depth: float = 0.0
        for camera_properties in data:
            if camera_properties.depth is not None:
                max_depth = max(max_depth, camera_properties.depth.max())
        self.camera.near_plane = 1.0e-2
        self.camera.far_plane = max_depth * 1.1
        # perform train/test split
        with open(f"{self.dataset_path}/{self.CATEGORY}/set_lists/set_lists_{self.SUBSET_NAME}.json", "r") as f:
            train_indices = [entry[1] - 1 for entry in json.load(f)['train'] if entry[0] == self.SEQUENCE]
        train_data: list[CameraProperties] = []
        test_data: list[CameraProperties] = []
        for i, camera_properties in enumerate(data):
            if i in train_indices:
                train_data.append(camera_properties)
            else:
                test_data.append(camera_properties)
        # remove invalid CameraProperties
        train_data = [camera_properties for camera_properties in train_data if camera_properties is not None]
        test_data = [camera_properties for camera_properties in test_data if camera_properties is not None]
        # perform validation split
        val_data: list[CameraProperties] = []
        if self.VAL_STEP > 0:
            indices = list(range(len(train_data)))
            indices_val = indices[::self.VAL_STEP]
            indices_train = list(set(indices) - set(indices_val))
            val_data = [train_data[i] for i in indices_val]
            train_data = [train_data[i] for i in indices_train]
        # return the dataset
        return {
            'train': train_data,
            'test': test_data,
            'val': val_data
        }

    def loadCO3DDepthsParallel(self, frame_annotations: list[dict[str, Any]], num_threads: int) -> list[torch.Tensor]:
        """Loads multiple depth maps in parallel."""
        depth_fileinfo = [
            f'{self.dataset_path}/{frame_annotation["depth"]["path"]}\n'
            f'{self.dataset_path}/{frame_annotation["depth"]["mask_path"]}\n'
            f'{frame_annotation["depth"]["scale_adjustment"]}'
            for frame_annotation in frame_annotations
        ]
        iterator_depth = getParallelLoadIterator(depth_fileinfo, self.IMAGE_SCALE_FACTOR, num_threads, load_function=self.loadCO3DDepth)
        depths = []
        for _ in Logger.logProgressBar(range(len(frame_annotations)), desc='depth', leave=False):
            depth, depth_mask = next(iterator_depth)
            if self.MASK_BACKGROUND_W_ALPHA:
                depth = torch.where(depth_mask > 0, depth, 0.0)
            depths.append(depth)
        return depths

    @staticmethod
    def loadCO3DDepth(filename: str, scale_factor: float | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Loads a CO3D depth map from the raw depth and the respective mask."""
        filename_depth, filename_depth_mask, scale_adjustment = filename.split('\n')
        # load depth
        try:
            # torchvision.io.read_image fails due to used file extension '*.jpg.geometric.png'
            depth_raw: np.ndarray = cv2.imread(filename_depth, flags=cv2.IMREAD_UNCHANGED)
        except Exception:
            raise DatasetError(f'Failed to load depth file: "{filename_depth}"')
        depth: torch.Tensor = torch.frombuffer(depth_raw, dtype=torch.float16).float().reshape(depth_raw.shape)[None]
        depth[~torch.isfinite(depth)] = 0.0
        depth *= float(scale_adjustment)
        # load depth mask
        try:
            depth_mask: torch.Tensor = io.read_image(filename_depth_mask, io.ImageReadMode.UNCHANGED)
        except Exception:
            raise DatasetError(f'Failed to load depth mask file: "{filename_depth_mask}"')
        depth_mask = (depth_mask > 0.0).float()
        # apply scaling factor
        if scale_factor is not None:
            depth = applyImageScaleFactor(depth, scale_factor)
            depth_mask = applyImageScaleFactor(depth_mask, scale_factor)
        return depth, depth_mask
