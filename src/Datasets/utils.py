# -- coding: utf-8 --

"""Datasets/utils.py: Contains utility functions used for the implementation of the available dataset classes."""

import math
import os
from pathlib import Path
from typing import Iterator, Callable, Any

import natsort
import torch
from torch.multiprocessing import Pool
from torchvision import io

import Framework
from Cameras.utils import CameraProperties
from Logger import Logger


def list_sorted_files(path: Path) -> list[str]:
    """Returns a naturally sorted list of files in the given directory."""
    return natsort.natsorted([i.name for i in path.iterdir() if i.is_file()])


def list_sorted_directories(path: Path) -> list[str]:
    """Returns a naturally sorted list of subdirectories in the given directory."""
    return natsort.natsorted([i.name for i in path.iterdir() if i.is_dir()])


def loadImage(filename: str, scale_factor: float | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Loads an image from the specified file."""
    try:
        image: torch.Tensor = io.read_image(path=filename, mode=io.ImageReadMode.UNCHANGED)
    except Exception:
        raise DatasetError(f'Failed to load image file: "{filename}"')
    # convert image to the format used by the framework
    image = image.float() / 255
    # apply scaling factor to image
    if scale_factor is not None:
        image = applyImageScaleFactor(image, scale_factor)
    # extract alpha channel if available
    rgb, alpha = image.split([3, 1]) if image.shape[0] == 4 else (image, None)
    return rgb, alpha


def parallelLoadFN(args: dict[str, Any]) -> Any:
    """Function executed by each thread when loading in parallel."""
    torch.set_num_threads(1)
    load_function = args['load_function']
    del args['load_function']
    return load_function(**args)


def getParallelLoadIterator(filenames: list[str], scale_factor: float | None, num_threads: int,
                            load_function: Callable) -> Iterator:
    """Returns iterator for parallel image loading."""
    # create thread pool
    if num_threads < 1:
        num_threads = os.cpu_count() - 1
    p = Pool(min(num_threads, len(filenames)))
    # create and return the iterator
    return p.imap(
        func=parallelLoadFN,
        iterable=[{'load_function': load_function, 'filename': filenames[i], 'scale_factor': scale_factor} for i in range(len(filenames))],
        chunksize=1
    )


def loadImagesParallel(filenames: list[str], scale_factor: float | None, num_threads: int, desc="") -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Loads multiple images in parallel."""
    iterator = getParallelLoadIterator(filenames, scale_factor, num_threads, load_function=loadImage)
    rgbs, alphas = [], []
    for _ in Logger.logProgressBar(range(len(filenames)), desc=desc, leave=False):
        rgb, alpha = next(iterator)
        rgbs.append(rgb)
        alphas.append(alpha)
    return rgbs, alphas


def applyImageScaleFactor(image: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Scales the image by the specified factor."""
    return torch.nn.functional.interpolate(
        input=image[None],
        scale_factor=scale_factor,
        mode='area',
    )[0]


def applyBGColor(rgb: torch.Tensor, alpha: torch.Tensor | None, bg_color: torch.Tensor | None) -> torch.Tensor:
    """Applies the given color to the image according to its alpha values."""
    if bg_color is not None and alpha is not None:
        rgb = rgb * alpha + (1 - alpha) * bg_color[:, None, None].to(alpha.device)
    return rgb


def normalizeRay(ray: torch.Tensor) -> torch.Tensor:
    """Normalizes a vector."""
    return ray / torch.linalg.norm(ray)


def getViewMatrix(view_dir: torch.Tensor, up_dir: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Creates a view matrix."""
    v2: torch.Tensor = normalizeRay(view_dir)
    v0: torch.Tensor = normalizeRay(torch.cross(up_dir, v2))
    v1: torch.Tensor = normalizeRay(torch.cross(v0, v2))
    return torch.cat([torch.stack([v0, v1, v2, position], dim=1), torch.tensor([[0, 0, 0, 1]])], dim=0)


def getAveragePose(view_matrices: torch.Tensor) -> torch.Tensor:
    """Creates an average view matrix."""
    avg_position: torch.Tensor = view_matrices[:, :3, -1].mean(0)
    view_dir: torch.Tensor = view_matrices[:, :3, 2].sum(0)
    up_dir: torch.Tensor = view_matrices[:, :3, 1].sum(0)
    return getViewMatrix(view_dir, up_dir, avg_position)


def recenterPoses(view_matrices: torch.Tensor) -> torch.Tensor:
    """Recenters the scene coordinate system."""
    # create inverse average transformation
    center_transform: torch.Tensor = torch.linalg.inv(getAveragePose(view_matrices))
    # apply transformation
    return center_transform @ view_matrices


def createSpiralPath(view_matrices: torch.Tensor, depth_min_max: torch.Tensor, image_shape: tuple[int, int, int],
                     focal_x: float, focal_y: float) -> list[CameraProperties]:
    """Creates test views from spiral path, analogous to the original implementation."""
    average_pose: torch.Tensor = getAveragePose(view_matrices)
    up: torch.Tensor = -normalizeRay(view_matrices[:, :3, 1].sum(0))
    close_depth: float = depth_min_max.min().item() * 0.9
    inf_depth: float = depth_min_max.max().item() * 5.0
    dt: float = 0.75
    focal: float = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)
    # get radii for spiral path
    rads: torch.Tensor = torch.quantile(torch.abs(view_matrices[:, :3, 3]), q=0.9, dim=0)
    n_views: int = 120
    n_rots: int = 2
    view_matrices_test: list[torch.Tensor] = []
    rads: torch.Tensor = torch.tensor(list(rads) + [1.])
    for theta in torch.linspace(0.0, 2.0 * math.pi * n_rots, n_views + 1)[:-1]:
        c: torch.Tensor = torch.mm(
            average_pose[:3, :4],
            (torch.tensor([torch.cos(theta), -torch.sin(theta), -torch.sin(theta * 0.5), 1.]) * rads)[:, None]
        ).squeeze()
        z: torch.Tensor = normalizeRay(c - torch.mm(average_pose[:3, :4], torch.tensor([[0], [0], [-focal], [1.]])).squeeze())
        view_matrices_test.append(getViewMatrix(z, up, c))

    view_matrices_test: torch.Tensor = torch.stack(view_matrices_test).float()
    return [
        CameraProperties(
            width=image_shape[2],
            height=image_shape[1],
            rgb=None,
            alpha=None,
            c2w=c2w,
            focal_x=focal_x,
            focal_y=focal_y
        )
        for c2w in view_matrices_test
    ]


def saveImage(filepath: Path, image: torch.Tensor) -> None:
    """Writes the input image tensor to the file given by filepath."""
    filename, filetype = os.path.splitext(filepath)
    match filetype.lower():
        case '.png' | '':
            io.write_png(
                input=(image * 255).byte().cpu(),
                filename=f'{filename}.png',
                compression_level=6,  # opencv uses 3
            )
        case '.jpg' | '.jpeg':
            io.write_jpeg(
                input=(image * 255).byte().cpu(),
                filename=f'{filename}.jpeg',
                quality=75,  # opencv uses 95
            )
        case _:
            raise DatasetError(f'Invalid file type specified "{filetype}"')


# def loadOpticalFlow(filepath: Path, scale_factor: float | None = None):
#     """ reads optical flow from Middlebury .flo file"""
#     flow = None
#     try:
#         with open(str(filepath), 'rb') as f:
#             magic = np.fromfile(f, np.float32, count=1)
#             if magic != 202021.25:
#                 msg: str = f'invalid .flo file: {filepath} (magic number incorrect)'
#                 Logger.logError(msg)
#                 raise DatasetError(msg)
#             w = np.fromfile(f, np.int32, count=1)[0]
#             h = np.fromfile(f, np.int32, count=1)[0]
#             flow = np.fromfile(f, np.float32, count=2 * w * h)
#             flow = np.resize(flow, (h, w, 2))
#             # apply scaling factor to image
#             if scale_factor is not None:
#                 flow = applyImageScaleFactor(flow, scale_factor)
#                 flow *= scale_factor
#             flow = flow.transpose((2, 0, 1))
#     except FileNotFoundError:
#         msg: str = f'invalid flow filepath: {filepath}'
#         Logger.logError(msg)
#         raise DatasetError(msg)
#     return flow


class DatasetError(Framework.FrameworkError):
    """Raise in case of an exception regarding the dataset."""
