# -- coding: utf-8 --

"""MoNeRF/Model.py: MoNeRF deformation and canonical model based on InstantNGP/tinycudann."""

import math

import torch
from kornia.utils.grid import create_meshgrid3d

import Framework
from Methods.Base.Model import BaseModel
import Thirdparty.TinyCudaNN as tcnn


@Framework.Configurable.configure(
    USE_ZERO_CANONICAL=False,
    SCALE=0.5,
    RESOLUTION=128,
    CENTER=[0.0, 0.0, 0.0],
    HASHMAP_NUM_LEVELS=16,
    HASHMAP_NUM_FEATURES_PER_LEVEL=2,
    HASHMAP_LOG2_SIZE=19,
    HASHMAP_BASE_RESOLUTION=16,
    HASHMAP_TARGET_RESOLUTION=2048,
    NUM_DENSITY_OUTPUT_FEATURES=16,
    NUM_DENSITY_NEURONS=64,
    NUM_DENSITY_LAYERS=1,
    DIR_SH_ENCODING_DEGREE=4,
    NUM_COLOR_NEURONS=64,
    NUM_COLOR_LAYERS=2,
    NUM_DEFORMATION_NEURONS=128,
    NUM_DEFORMATION_LAYERS=4,
    NUM_TEMPORAL_NEURONS=64,
    NUM_TEMPORAL_LAYERS=2,
    TEMPORAL_ENCODING_LENGTH=4,
    POSITIONAL_ENCODING_LENGTH=10,
    TEMPORAL_BASIS_LENGTH=16,
    USE_TEMPORAL_BASIS=True
)
class MoNeRFModel(BaseModel):
    """Defines MoNeRF data model"""

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    def __del__(self) -> None:
        # super().__del__()
        torch.cuda.empty_cache()
        tcnn.free_temporary_memory()

    def build(self) -> 'MoNeRFModel':
        """Builds the model."""
        # createGrid
        self.register_buffer('center', torch.tensor([self.CENTER]))
        self.register_buffer('xyz_min', -torch.ones(1, 3) * self.SCALE)
        self.register_buffer('xyz_max', torch.ones(1, 3) * self.SCALE)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min) / 2)
        self.cascades = max(1 + int(math.ceil(math.log2(2 * self.SCALE))), 1)
        self.register_buffer('density_grid', torch.zeros(self.cascades, self.RESOLUTION ** 3))
        self.register_buffer('grid_coords', create_meshgrid3d(self.RESOLUTION, self.RESOLUTION, self.RESOLUTION, False, dtype=torch.int32, device=self.density_grid.device).reshape(-1, 3))
        self.register_buffer('density_bitfield', torch.zeros(self.cascades*self.RESOLUTION ** 3 // 8, dtype=torch.uint8))
        # encodings
        self.encoding_canonical = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": self.HASHMAP_NUM_LEVELS,
                    "n_features_per_level": self.HASHMAP_NUM_FEATURES_PER_LEVEL,
                    "log2_hashmap_size": self.HASHMAP_LOG2_SIZE,
                    "base_resolution": self.HASHMAP_BASE_RESOLUTION,
                    "per_level_scale": math.exp(math.log(self.HASHMAP_TARGET_RESOLUTION * self.SCALE / self.HASHMAP_BASE_RESOLUTION) / (self.HASHMAP_NUM_LEVELS - 1)),
                    "interpolation": "Linear"
                },
                seed=Framework.config.GLOBAL.RANDOM_SEED
            )
        self.encoding_direction = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": self.DIR_SH_ENCODING_DEGREE,
            },
            seed=Framework.config.GLOBAL.RANDOM_SEED
        )

        self.deformation_net_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                    "otype": "Frequency",
                    "n_frequencies": self.POSITIONAL_ENCODING_LENGTH
                },
            seed=Framework.config.GLOBAL.RANDOM_SEED
        )

        self.temporal_basis_net_encoding = tcnn.Encoding(
            n_input_dims=1,
            encoding_config={
                "otype": "OneBlob",
                "n_bins": 2 * self.TEMPORAL_ENCODING_LENGTH
            },
            seed=Framework.config.GLOBAL.RANDOM_SEED
        )

        # create networks
        self.deformation_net = tcnn.Network(
            n_input_dims=(self.deformation_net_encoding.n_output_dims + 3) if self.USE_TEMPORAL_BASIS else
            (self.deformation_net_encoding.n_output_dims + self.temporal_basis_net_encoding.n_output_dims + 4),
            n_output_dims=3 * (self.TEMPORAL_BASIS_LENGTH if self.USE_TEMPORAL_BASIS else 1),
            network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.NUM_DEFORMATION_NEURONS,
                    "n_hidden_layers": self.NUM_DEFORMATION_LAYERS + (0 if self.USE_TEMPORAL_BASIS else 1),
                },
            seed=Framework.config.GLOBAL.RANDOM_SEED
        )
        if self.USE_TEMPORAL_BASIS:
            self.temporal_basis_net = tcnn.Network(
                n_input_dims=1 + self.temporal_basis_net_encoding.n_output_dims,
                n_output_dims=self.TEMPORAL_BASIS_LENGTH,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.NUM_TEMPORAL_NEURONS,
                    "n_hidden_layers": self.NUM_TEMPORAL_LAYERS,
                },
                seed=Framework.config.GLOBAL.RANDOM_SEED
            )
        self.density_net = tcnn.Network(
            n_input_dims=self.encoding_canonical.n_output_dims,
            n_output_dims=self.NUM_DENSITY_OUTPUT_FEATURES,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.NUM_DENSITY_NEURONS,
                "n_hidden_layers": self.NUM_DENSITY_LAYERS,
            },
            seed=Framework.config.GLOBAL.RANDOM_SEED
        )
        self.rgb_net = tcnn.Network(
            n_input_dims=self.density_net.n_output_dims + self.encoding_direction.n_output_dims,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": 'Sigmoid',
                "n_neurons": self.NUM_COLOR_NEURONS,
                "n_hidden_layers": self.NUM_COLOR_LAYERS,
            },
            seed=Framework.config.GLOBAL.RANDOM_SEED
        )
        return self
