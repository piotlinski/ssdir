"""$$z_{depth}$$ encoder"""
from itertools import chain
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class DepthEncoder(nn.Module):
    """Module encoding input image features to depth latent distribution params."""

    def __init__(
        self, feature_channels: List[int], z_depth_scale_const: Optional[float] = None
    ):
        super().__init__()
        self.feature_channels = feature_channels
        self.loc_encoders = self._build_depth_encoders()
        self.z_depth_scale_const = z_depth_scale_const
        if self.z_depth_scale_const is None:
            self.scale_encoders = self._build_depth_encoders()
        self.init_encoders()

    def _build_depth_encoders(self) -> nn.ModuleList:
        """Build conv layers list for encoding backbone output."""
        layers = [
            nn.Conv2d(
                in_channels=channels,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            for channels in self.feature_channels
        ]
        return nn.ModuleList(layers)

    def forward(
        self, features: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Takes tuple of tensors (batch_size x grid x grid x features)
        .. and outputs loc and scale tensors
        .. (batch_size x sum_features(grid*grid) x 1)
        """
        locs = []
        scales = []
        batch_size = features[0].shape[0]
        sequence = [features, self.loc_encoders]
        if self.z_depth_scale_const is None:
            sequence.append(self.scale_encoders)
        else:
            sequence.append(len(features) * [self.z_depth_scale_const])
        for feature, loc_enc, scale_enc in zip(*sequence):
            locs.append(
                loc_enc(feature)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, 1)
            )
            if self.z_depth_scale_const is None:
                scales.append(
                    torch.exp(scale_enc(feature))
                    .permute(0, 2, 3, 1)
                    .contiguous()
                    .view(batch_size, -1, 1)
                )
            else:
                scales.append(torch.full_like(locs[-1], fill_value=scale_enc))

        locs = torch.cat(locs, dim=1)
        scales = torch.cat(scales, dim=1)

        return locs, scales

    def init_encoders(self):
        """Initialize model params."""
        modules = [self.loc_encoders.modules()]
        if self.z_depth_scale_const is None:
            modules.append(self.scale_encoders.modules())
        for module in chain(*modules):
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
