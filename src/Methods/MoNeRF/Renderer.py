# -- coding: utf-8 --

"""MoNeRF/Renderer.py: MoNeRF renderering functionality."""

import torch
from torch import Tensor

from einops import rearrange
from Datasets.Base import BaseDataset
from Methods.Base.Renderer import BaseRenderer, BaseRenderingComponent
from Methods.MoNeRF.Model import MoNeRFModel
from Methods.Base.Model import BaseModel
from Cameras.Base import BaseCamera
from Cameras.utils import RayPropertySlice
import Framework
import Implementations
from Logger import Logger

# Import CUDA extension containing fast rendering routines adapted from kwea123 (https://github.com/kwea123/ngp_pl)
VolumeRenderingCuda = Implementations.CudaExtensions.getExtension('VolumeRenderingV2')


class MoNeRFRayRenderingComponent(BaseRenderingComponent):
    '''dataparallel rendering a batch of rays'''

    def __init__(self, model: MoNeRFModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, rays: Tensor, camera: BaseCamera, timestamp: float, max_samples: int, bg_color: Tensor,
                exponential_steps: bool, train_mode: bool) -> dict[str, Tensor | None]:
        # prepare rays for rendering
        rays_o = rays[:, RayPropertySlice.origin].contiguous() - self.model.center
        rays_d = rays[:, RayPropertySlice.direction].contiguous()
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        _, hits_t, _ = \
            VolumeRenderingCuda.RayAABBIntersector.apply(rays_o, rays_d, torch.zeros((1, 3), device=rays.device), self.model.half_size, 1)
        hits_t[..., 0].clamp_min_(camera.near_plane)
        hits_t[..., 1].clamp_max_(camera.far_plane)
        # render rays
        render_func = self.renderRaysTrain if train_mode else self.renderRaysTest
        kwargs = {}
        if exponential_steps:
            kwargs['exp_step_factor'] = 1 / 256
        results = render_func(rays_o, rays_d, hits_t, max_samples, timestamp, bg_color, **kwargs)
        return results

    @classmethod
    def get(cls, *args) -> 'BaseRenderingComponent':
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.logWarning('MoNeRF rendering should be run in single GPU mode.')
        return super(MoNeRFRayRenderingComponent, cls).get(*args)

    def queryModel(self, x: Tensor, d: Tensor, t: float, **_) -> tuple[Tensor, Tensor, Tensor]:
        '''calculate opacity, color and deformations for a set of spatial samples and a single point in time'''
        x = (x - self.model.xyz_min) / (self.model.xyz_max - self.model.xyz_min)
        x_encoded = torch.cat([self.model.deformation_net_encoding(x), x], dim=-1)
        t_encoded = torch.cat([self.model.temporal_basis_net_encoding(torch.tensor([[(t * 0.7) + 0.15]], device=x.device)), torch.tensor([[t]], device=x.device)], dim=-1)
        if self.model.USE_TEMPORAL_BASIS:
            temporal_basis = self.model.temporal_basis_net(t_encoded)[0]
            delta_x = self.model.deformation_net(x_encoded)
            delta_x = delta_x.view(-1, 3, self.model.TEMPORAL_BASIS_LENGTH) @ temporal_basis
        else:
            delta_x = self.model.deformation_net(torch.cat([x_encoded, t_encoded.expand(x_encoded.shape[0], -1)], dim=-1))
        if t > 0.0 or not self.model.USE_ZERO_CANONICAL:
            x = x + delta_x
        canonical_features = self.model.encoding_canonical(x)
        h = self.model.density_net(canonical_features)
        sigmas = VolumeRenderingCuda.TruncExp.apply(h[..., 0])
        directions_encoded = self.model.encoding_direction((d + 1) / 2.0)
        rgbs = self.model.rgb_net(torch.cat([directions_encoded, h], -1))
        return sigmas, rgbs, delta_x

    @torch.no_grad()
    def renderRaysTest(self, rays_o: Tensor, rays_d: Tensor, hits_t: Tensor, max_samples: int,
                       timestamp: float, bg_color: Tensor, **kwargs) -> dict[str, Tensor | None]:
        """
        Renders large amount of rays using efficient ray marching
        Code adapted from pytorch InstantNGP reimplementation of kwea123 (https://github.com/kwea123/ngp_pl)
        """
        exp_step_factor = kwargs.get('exp_step_factor', 0.)
        results = {}
        # output tensors to be filled
        N_rays = len(rays_o)
        device = rays_o.device
        opacity = torch.zeros(N_rays, device=device)
        depth = torch.zeros(N_rays, device=device)
        rgb = torch.zeros(N_rays, 3, device=device)
        samples = total_samples = 0
        alive_indices = torch.arange(N_rays, device=device)
        # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
        # otherwise, 4 is more efficient empirically
        min_samples = 1 if exp_step_factor == 0 else 4
        while samples < kwargs.get('max_samples', max_samples):
            N_alive = len(alive_indices)
            if N_alive == 0:
                break
            # the number of samples to add on each ray
            N_samples = max(min(N_rays//N_alive, 64), min_samples)
            samples += N_samples
            xyzs, dirs, deltas, ts, N_eff_samples = VolumeRenderingCuda.raymarching_test(
                rays_o, rays_d, hits_t[:, 0], alive_indices,
                self.model.density_bitfield, self.model.cascades,
                self.model.SCALE, exp_step_factor,
                self.model.RESOLUTION, max_samples, N_samples
            )
            total_samples += N_eff_samples.sum()
            xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
            dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
            valid_mask = ~torch.all(dirs == 0, dim=1)
            if valid_mask.sum() == 0:
                break
            sigmas = torch.zeros(len(xyzs), device=device)
            rgbs = torch.zeros(len(xyzs), 3, device=device)
            _sigmas, _rgbs, _ = self.queryModel(xyzs[valid_mask], dirs[valid_mask], timestamp, **kwargs)
            sigmas[valid_mask], rgbs[valid_mask] = _sigmas.float(), _rgbs.float()
            sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
            rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
            VolumeRenderingCuda.composite_test_fw(
                sigmas, rgbs, deltas, ts,
                hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4),
                N_eff_samples, opacity, depth, rgb)
            alive_indices = alive_indices[alive_indices >= 0]  # remove converged rays
        results['alpha'] = opacity
        results['depth'] = depth
        results['rgb'] = rgb
        results['total_samples'] = total_samples  # total samples for all rays
        results['rgb'] += bg_color * rearrange(1 - opacity, 'n -> n 1')
        return results

    @torch.cuda.amp.autocast()
    def renderRaysTrain(self, rays_o: Tensor, rays_d: Tensor, hits_t: Tensor, max_samples: int,
                        timestamp: float, bg_color: Tensor, **kwargs) -> dict[str, Tensor | None]:
        """
        Renders training rays.
        Code adapted from pytorch InstantNGP reimplementation of kwea123 (https://github.com/kwea123/ngp_pl)
        """
        exp_step_factor = kwargs.get('exp_step_factor', 0.)
        results = {}
        rays_a, xyzs, dirs, results['deltas'], results['ts'], results['rm_samples'] = VolumeRenderingCuda.RayMarcher.apply(
            rays_o, rays_d, hits_t[:, 0], self.model.density_bitfield,
            self.model.cascades, self.model.SCALE,
            exp_step_factor, self.model.RESOLUTION, max_samples
        )
        for k, v in kwargs.items():  # supply additional inputs, repeated per ray
            if isinstance(v, Tensor):
                kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)
        sigmas, rgbs, delta_x = self.queryModel(xyzs, dirs, timestamp, **kwargs)
        results['vr_samples'], results['alpha'], results['depth'], results['rgb'], results['ws'] = VolumeRenderingCuda.VolumeRenderer.apply(
            sigmas, rgbs.contiguous(), results['deltas'], results['ts'], rays_a, kwargs.get('T_threshold', 1e-4)
        )
        results['rays_a'] = rays_a
        results['rgb'] = results['rgb'] + bg_color * rearrange(1-results['alpha'], 'n -> n 1')
        results['delta_x'] = delta_x
        return results


@Framework.Configurable.configure(
    MAX_SAMPLES=512,
    EXPONENTIAL_STEPS=False,
    DENSITY_THRESHOLD=0.01
)
class MoNeRFRenderer(BaseRenderer):
    """Rendering routines for MoNeRF."""

    def __init__(self, model: BaseModel) -> None:
        BaseRenderer.__init__(self, model, [MoNeRFModel])
        self.ray_rendering_component = MoNeRFRayRenderingComponent.get(self.model)
        self.density_threshold = self.DENSITY_THRESHOLD * self.MAX_SAMPLES / 3 ** 0.5

    def renderRays(self, rays: Tensor, camera: 'BaseCamera', timestamp: float, train_mode: bool = False,
                   custom_bg_color: Tensor | None = None, **_) -> dict[str, Tensor | None]:
        """Renders the given set of rays using the renderer's rendering component."""
        outputs = self.ray_rendering_component(
            rays=rays,
            camera=camera,
            max_samples=self.MAX_SAMPLES,
            bg_color=custom_bg_color.repeat(len(Framework.config.GLOBAL.GPU_INDICES), 1) if custom_bg_color is not None else camera.background_color.repeat(len(Framework.config.GLOBAL.GPU_INDICES), 1),
            timestamp=timestamp,
            exponential_steps=self.EXPONENTIAL_STEPS,
            train_mode=train_mode
        )
        return outputs

    def renderImage(self, camera: 'BaseCamera', timestamp: float = None, to_chw: bool = False) -> dict[str, Tensor | None]:
        """Renders a complete image for the given camera and optional timestamp."""
        rays: Tensor = camera.generateRays().type(Framework.config.GLOBAL.DEFAULT_TENSOR_TYPE)
        rendered_rays = self.renderRays(
            rays=rays,
            camera=camera,
            timestamp=timestamp if timestamp is not None else camera.properties.timestamp
        )
        # reshape rays to images
        for key in rendered_rays:
            if rendered_rays[key] is not None and key not in ['total_samples', 'delta_x']:
                rendered_rays[key] = rendered_rays[key].reshape(camera.properties.height, camera.properties.width, -1)
                if to_chw:
                    rendered_rays[key] = rendered_rays[key].permute((2, 0, 1))
        return rendered_rays

    @torch.no_grad()
    def getAllCells(self) -> list[tuple[Tensor, Tensor]]:
        '''Returns all cells in the occupancy grid'''
        indices = VolumeRenderingCuda.morton3D(self.model.grid_coords).long()
        cells = [(indices, self.model.grid_coords)] * self.model.cascades
        return cells

    @torch.no_grad()
    def sampleCells(self, M: int, density_threshold: float) -> list[tuple[Tensor, Tensor]]:
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > density_threshold
        """
        cells = []
        for c in range(self.model.cascades):
            # uniform cells
            coords1 = torch.randint(self.model.RESOLUTION, (M, 3), dtype=torch.int32,
                                    device=self.model.density_grid.device)
            indices1 = VolumeRenderingCuda.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.model.density_grid[c] > density_threshold)[:, 0]
            if len(indices2) > 0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.model.density_grid.device)
                indices2 = indices2[rand_idx]
            coords2 = VolumeRenderingCuda.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def carveDensityGrid(self, dataset: 'BaseDataset', subtractive: bool = False, use_alpha: bool = False) -> None:
        '''
        Carves the occupancy grid using the given dataset camera poses.
        subtractive=False -> keep points visible in at least one camera, True-> point must be visible in all cameras
        use_alpha=True -> use images alpha channel to carve the grid, False -> only camera frustum
        '''
        Logger.logInfo(f'carving density grid from camera poses (using alpha masks: {use_alpha})')
        # dataset.train()
        cells = self.getAllCells()
        cell_positions_world = []
        for c in range(self.model.cascades):
            _, coords = cells[c]
            s = min(2 ** (c - 1), self.model.SCALE)
            half_grid_size = s / self.model.RESOLUTION
            xyzs_w = ((coords / (self.model.RESOLUTION - 1) * 2 - 1) * (s - half_grid_size)) + self.model.center
            cell_positions_world.append(xyzs_w)
        remaining_cells = torch.full_like(self.model.density_grid, fill_value=subtractive, dtype=torch.bool, device=self.model.density_grid.device)
        for camera_properties in Logger.logProgressBar(dataset, desc='frame', leave=False):
            dataset.camera.setProperties(camera_properties)
            alpha_img = None
            if use_alpha and camera_properties.alpha is not None:
                alpha_img = torch.nn.functional.conv2d(camera_properties.alpha[None], torch.ones((1, 1, 3, 3), requires_grad=False),
                                                       bias=None, stride=1, padding=1, dilation=1, groups=1)[0] > 0.0
            for c in range(self.model.cascades):
                uv, valid = dataset.camera.projectPoint(cell_positions_world[c])
                if alpha_img is not None:
                    uv = torch.round(uv).long()[valid]
                    alpha_values = alpha_img[:, uv[:, 1], uv[:, 0]] > 0.0
                    valid[valid.clone()] = alpha_values
                remaining_cells[c] = torch.logical_and(remaining_cells[c], valid) if subtractive else torch.logical_or(remaining_cells[c], valid)
        for c in range(self.model.cascades):
            values = torch.where(torch.nn.functional.conv3d(remaining_cells[c].reshape(1, 1, self.model.RESOLUTION, self.model.RESOLUTION, self.model.RESOLUTION).type(Framework.config.GLOBAL.DEFAULT_TENSOR_TYPE),
                                 torch.ones((1, 1, 3, 3, 3), requires_grad=False), bias=None, stride=1, padding=1, dilation=1, groups=1).flatten() > 0.0, 0.0, -1.0)
            self.model.density_grid[c, cells[c][0]] = values

    @torch.no_grad()
    def queryDensityMultitime(self, x: Tensor, t: Tensor, **_) -> Tensor:
        '''calculate maximum opacity for a set of spatial samples over a set of temporal samples'''
        x = (x - self.model.xyz_min) / (self.model.xyz_max - self.model.xyz_min)
        t_encoded = torch.cat([self.model.temporal_basis_net_encoding((t[:, None] * 0.7) + 0.15), t[:, None]], dim=-1)
        x_encoded = torch.cat([self.model.deformation_net_encoding(x), x], dim=-1)
        if self.model.USE_TEMPORAL_BASIS:
            temporal_basis = self.model.temporal_basis_net(t_encoded)
            delta_x = self.model.deformation_net(x_encoded)
            delta_x = (delta_x.view(-1, 3, self.model.TEMPORAL_BASIS_LENGTH)[:, None] * temporal_basis[None, :, None]).sum(-1)
            # delta_x = torch.zeros_like(delta_x)
            x = x[:, None] + delta_x
            canonical_features = self.model.encoding_canonical(x.view(-1, 3))
            h = self.model.density_net(canonical_features)
            sigmas = VolumeRenderingCuda.TruncExp.apply(h[..., 0])
            # sigmas = sigmas.view((x.shape[0], -1)).sum(dim=-1)
            sigmas = sigmas.view((x.shape[0], -1)).max(dim=-1).values
        else:
            sigmas = None
            for i in range(t_encoded.shape[0]):
                delta_x = self.model.deformation_net(torch.cat([x_encoded, t_encoded[i:i+1].expand(x_encoded.shape[0], -1)], dim=-1))
                # delta_x = torch.zeros_like(delta_x)
                canonical_features = self.model.encoding_canonical(x + delta_x)
                h = VolumeRenderingCuda.TruncExp.apply(self.model.density_net(canonical_features)[..., 0])
                # sigmas = h if sigmas is None else (sigmas + h)
                sigmas = h if sigmas is None else torch.maximum(sigmas, h)
        return sigmas

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=True)
    def updateDensityGrid(self, num_temporal_samples: int, warmup: bool = False, decay: float = 0.95) -> None:
        '''Updates the density grid analogous to InstantNGP, but using maximum density over temporal samples'''
        density_grid_tmp = torch.zeros_like(self.model.density_grid)
        if warmup:  # during the first steps
            cells = self.getAllCells()
        else:
            cells = self.sampleCells(self.model.RESOLUTION ** 3 // 4, self.density_threshold)
        t_step = 1.0 / num_temporal_samples
        t_samples = torch.arange(start=0, end=1.0, step=t_step)
        t_samples += torch.rand(t_samples.shape) * t_step
        # infer sigmas
        for c in range(self.model.cascades):
            indices, coords = cells[c]
            s = min(2 ** (c - 1), self.model.SCALE)
            half_grid_size = s / self.model.RESOLUTION
            xyzs_w = ((coords / (self.model.RESOLUTION - 1) * 2 - 1) * (s-half_grid_size))
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w) * 2 - 1) * half_grid_size
            density_grid_tmp[c, indices] += self.queryDensityMultitime(xyzs_w, t_samples)
        self.model.density_grid = \
            torch.where(self.model.density_grid < 0,
                        self.model.density_grid,
                        torch.maximum(self.model.density_grid*decay, density_grid_tmp))

        mean_density = self.model.density_grid[self.model.density_grid > 0].mean().item()

        VolumeRenderingCuda.packbits(self.model.density_grid, min(mean_density, self.density_threshold),
                                     self.model.density_bitfield)
