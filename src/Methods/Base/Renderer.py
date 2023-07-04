# -- coding: utf-8 --

"""Base/Renderer.py: Implementation of the basic renderer which processes the results of the models."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import mean, median

import torch
import torchmetrics

import Framework
from Cameras.Base import BaseCamera
from Datasets.Base import BaseDataset
from Datasets.utils import saveImage, loadImagesParallel, list_sorted_files
from Logger import Logger
from Methods.Base.Model import BaseModel
from Methods.Base.utils import RendererError, pseudoColorDepth, VideoWriter, ColorMap


class BaseRenderingComponent(ABC, torch.nn.Module):
    """Basic subcomponent of renderers used to parallelize the rendering procedure of sub-models."""

    def __init__(self) -> None:
        super().__init__()
        super(ABC, self).__init__()

    @classmethod
    def get(cls, *args) -> 'BaseRenderingComponent':
        """Returns an instance of the rendering component that includes support for multi-gpu execution if requested."""
        instance = cls(*args)
        # wrap in DataParallel if multiple GPUs are being used
        if Framework.config.GLOBAL.GPU_INDICES is not None and len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            instance = torch.nn.DataParallel(
                module=instance, device_ids=Framework.config.GLOBAL.GPU_INDICES,
                output_device=Framework.config.GLOBAL.GPU_INDICES[0], dim=0
            )
        return instance

    @abstractmethod
    def forward(self, *args) -> None:
        """Implementations define forward passes."""
        pass


class BaseRenderer(Framework.Configurable, ABC):
    """Defines the basic renderer. Subclasses provide all functionality for their individual implementations."""

    def __init__(self, model: BaseModel, valid_model_types: list[type] = None) -> None:
        Framework.Configurable.__init__(self, 'RENDERER')
        ABC.__init__(self)
        # check if provided model is supported by this renderer
        if valid_model_types is not None and type(model) not in valid_model_types:
            Logger.logError(
                f'provided invalid model for renderer of type: "{type(self)}"'
                f'\n provided model type: "{type(model)}", valid options are: {valid_model_types}'
            )
            raise RendererError(f'provided invalid model for renderer of type: "{type(self)}"')
        # assign model
        self.model = model

    @abstractmethod
    def renderImage(self, camera: 'BaseCamera', to_chw: bool = False) -> dict[str, torch.Tensor | None]:
        """Renders an entire image for a given camera."""
        pass

    @torch.no_grad()
    def calculateImageQualityMetrics(self, results_path: Path, target_path: Path, output_path: Path, file_extension: str = '.png') -> None:
        """Calculate quality metrics (PSNR, SSIM, LPIPS)."""
        Logger.logInfo('calculating image quality metrics')
        result = torch.stack(
            loadImagesParallel([
                str(results_path / name) for name in list_sorted_files(results_path)
                if file_extension in name
            ], scale_factor=None, num_threads=4, desc='loading result')[0]
        ).type(Framework.config.GLOBAL.DEFAULT_TENSOR_TYPE)
        try:
            target = torch.stack(
                loadImagesParallel([
                    str(target_path / name) for name in list_sorted_files(target_path)
                    if file_extension in name
                ], scale_factor=None, num_threads=4, desc='loading gt')[0]
            ).type(Framework.config.GLOBAL.DEFAULT_TENSOR_TYPE)
        except Exception:
            Logger.logWarning('failed to calculate quality metrics: no GT data available.')
            return
        # lpips_func = lpips.LPIPS(net='vgg', verbose=False)
        psnr_metric, psnr_values = torchmetrics.PeakSignalNoiseRatio(data_range=1.0), []
        ssim_metric, ssim_values = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0), []
        lpips_metric, lpips_values = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True), []
        for result, target in Logger.logProgressBar(zip(result, target), total=len(result), desc='calculate metrics', leave=False):
            psnr_values.append(psnr_metric(result[None], target[None]).item())
            ssim_values.append(ssim_metric(result[None], target[None]).item())
            lpips_values.append(lpips_metric(result[None], target[None]).item())
        psnr_all, mean_psnr, median_psnr = psnr_metric.compute(), mean(psnr_values), median(psnr_values)
        ssim_all, mean_ssim, median_ssim = ssim_metric.compute(), mean(ssim_values), median(ssim_values)
        lpips_all, mean_lpips, median_lpips = lpips_metric.compute(), mean(lpips_values), median(lpips_values)
        # write output file
        Logger.logInfo(
            f'results:\n'
            f'PSNR\t{psnr_all:.2f}\n'
            f'SSIM\t{mean_ssim:.3f}\n'
            f'LPIPS\t{mean_lpips:.3f}'
        )
        with open(output_path / 'metrics_8bit.txt', 'w') as f:
            # write summary (raw metrics in first three rows to facilitate parsing)
            f.write(
                f'{self.model.model_name}\n'
                f'Metric\tMean\tMedian\tPixelMean\n'
                f'PSNR\t{mean_psnr:.2f}\t{median_psnr:.2f}\t{psnr_all:.2f}\n'
                f'SSIM\t{mean_ssim:.3f}\t{median_ssim:.3f}\t{ssim_all:.3f}\n'
                f'LPIPS\t{mean_lpips:.3f}\t{median_lpips:.3f}\t{lpips_all:.3f}\n'
                f'\n'
                f'Image\tPSNR\tSSIM\tLPIPS\n'
            )
            for i, (i_psnr, i_ssim, i_lpips) in enumerate(zip(psnr_values, ssim_values, lpips_values)):
                f.write(f'{i}\t{i_psnr:.2f}\t{i_ssim:.3f}\t{i_lpips:.3f}\n')
            f.write(f'\nPSNR:{mean_psnr} SSIM:{mean_ssim} LPIPS:{mean_lpips}')  # for automated parsing

    @torch.no_grad()
    def visualizeError(self, results_path: Path, target_path: Path, output_path: Path, file_extension: str = '.png') -> None:
        """Visualize differences between result and reference images."""
        Logger.logInfo('visualizing errors')
        result = torch.stack(
            loadImagesParallel([
                str(results_path / name) for name in list_sorted_files(results_path)
                if file_extension in name
            ], scale_factor=None, num_threads=4, desc='loading result')[0]
        ).type(Framework.config.GLOBAL.DEFAULT_TENSOR_TYPE)
        try:
            target = torch.stack(
                loadImagesParallel([
                    str(target_path / name) for name in list_sorted_files(target_path)
                    if file_extension in name
                ], scale_factor=None, num_threads=4, desc='loading gt')[0]
            ).type(Framework.config.GLOBAL.DEFAULT_TENSOR_TYPE)
        except Exception:
            Logger.logWarning('failed to visualize errors: no GT data available.')
            return
        # prepare error visualization
        output_directory_error = output_path / 'error'
        os.makedirs(output_directory_error, exist_ok=True)
        video_writer = VideoWriter(output_directory_error / 'error.mp4', width=int(result.shape[3] * 2), height=result.shape[2])
        l1_distances = torch.abs(result - target).clamp(0.0, 1.0)
        l2_distances = torch.sum((result - target) ** 2, dim=1, keepdim=True)
        min_l2, max_l2 = torch.min(l2_distances), torch.max(l2_distances)
        l2_distances = ((l2_distances - min_l2) / (max_l2 - min_l2) * 100).clamp(0.0, 1.0)
        l2_distances = torch.index_select(
            ColorMap.get('VIRIDIS'), dim=0, index=(l2_distances * 255).int().flatten()
        ).reshape(l2_distances.shape[0], *l2_distances.shape[2:], 3).permute(0, 3, 1, 2)

        for index, (l1_distance, l2_distance) in Logger.logProgressBar(
                enumerate(zip(l1_distances, l2_distances)), total=len(result), desc='visualizing errors', leave=False):
            error = torch.cat([l1_distance, l2_distance], dim=-1)
            saveImage(output_directory_error / f'{index:05d}{file_extension}', error)
            video_writer.addFrame(error)
        video_writer.close()

    @torch.no_grad()
    def renderSubset(self, output_directory: Path, dataset: 'BaseDataset',
                     calculate_metrics: bool = False, visualize_errors: bool = False, verbose=True) -> None:
        if not verbose:
            Logger.setMode(Logger.MODE_SILENT)
        if len(dataset) > 0:
            Logger.logInfo(f'rendering {dataset.mode} set images')
            # create output directories
            output_directory_main = output_directory / f'{dataset.mode}_renderings_{self.model.num_iterations_trained}'
            output_directory_rgb = output_directory_main / 'rgb'
            output_directory_alpha = output_directory_main / 'alpha'
            output_directory_depth = output_directory_main / 'depth'
            output_directory_gt = output_directory_main / 'gt'
            os.makedirs(output_directory_rgb, exist_ok=True)
            os.makedirs(output_directory_alpha, exist_ok=True)
            os.makedirs(output_directory_depth, exist_ok=True)
            os.makedirs(output_directory_gt, exist_ok=True)
            used_extension = '.png'
            # initialize video writer
            video_writer = VideoWriter([
                output_directory_rgb / 'rgb.mp4',
                output_directory_alpha / 'alpha.mp4',
                output_directory_depth / 'depth.mp4'
            ], width=dataset[0].width, height=dataset[0].height)

            # loop over subset
            for index, sample in Logger.logProgressBar(enumerate(dataset), total=len(dataset), desc="image", leave=False):
                # render image
                dataset.camera.setProperties(sample)
                outputs = self.renderImage(dataset.camera, to_chw=True)
                rgb = outputs['rgb']
                alpha = outputs['alpha'].expand(rgb.shape)
                depth = pseudoColorDepth(
                    color_map='TURBO',
                    depth=outputs['depth'],
                    near_far=None,  # near_far=(dataset.camera.near_plane, dataset.camera.far_plane)
                    alpha=outputs['alpha']
                )
                # save images
                saveImage(output_directory_rgb / f'{index:05d}{used_extension}', rgb)
                saveImage(output_directory_alpha / f'{index:05d}{used_extension}', alpha)
                saveImage(output_directory_depth / f'{index:05d}{used_extension}', depth)
                if sample.rgb is not None:
                    saveImage(output_directory_gt / f'{index:05d}{used_extension}', sample.rgb)
                video_writer.addFrame([rgb, alpha, depth])
            video_writer.close()

            # calculate quality metrics (PSNR, SSIM, LPIPS), reload saved 8bit images for comparability
            if calculate_metrics:
                self.calculateImageQualityMetrics(output_directory_rgb, output_directory_gt, output_directory_main, used_extension)

            # visualize differences between result and reference images
            if visualize_errors:
                self.visualizeError(output_directory_rgb, output_directory_gt, output_directory_main, used_extension)
        Logger.setMode(Framework.config.GLOBAL.LOG_LEVEL)
