"""$$z_{what}$$ encoder and decoder."""
from itertools import chain
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional


class WhatEncoder(nn.Module):
    """Module encoding input image features to what latent distribution params."""

    def __init__(
        self, z_what_size: int, feature_channels: List[int], feature_maps: List[int]
    ):
        super().__init__()
        self.h_size = z_what_size
        self.feature_channels = feature_channels
        self.feature_maps = feature_maps
        self.loc_encoders = self._build_what_encoders()
        self.scale_encoders = self._build_what_encoders()
        self.bg_loc_encoder = self._build_what_bg_encoder()
        self.bg_scale_encoder = self._build_what_bg_encoder()
        self.init_encoders()

    def _build_what_encoders(self) -> nn.ModuleList:
        """Build conv layers list for encoding backbone output."""
        layers = [
            nn.Conv2d(
                in_channels=channels,
                out_channels=self.h_size,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            for channels in self.feature_channels
        ]
        return nn.ModuleList(layers)

    def _build_what_bg_encoder(self) -> nn.Module:
        """Build layer for encoding background what latent from largest feature map."""
        feature_idx = self.feature_maps.index(min(self.feature_maps))
        return nn.Conv2d(
            in_channels=self.feature_channels[feature_idx],
            out_channels=self.h_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(
        self, features: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Takes tuple of tensors (batch_size x grid x grid x features)
        .. and outputs locs and scales tensors
        .. (batch_size x sum_features(grid*grid) x z_what_size)
        """
        locs = []
        scales = []
        batch_size = features[0].shape[0]
        for feature, loc_enc, scale_enc in zip(
            features,
            self.loc_encoders,
            self.scale_encoders,
        ):
            locs.append(
                loc_enc(feature)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, self.h_size)
            )
            scales.append(
                functional.softplus(scale_enc(feature))
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, self.h_size)
            )
        bg_feature_idx = self.feature_maps.index(min(self.feature_maps))
        locs.append(
            self.bg_loc_encoder(features[bg_feature_idx])
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(batch_size, -1, self.h_size)
        )
        scales.append(
            functional.softplus(self.bg_scale_encoder(features[bg_feature_idx]))
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(batch_size, -1, self.h_size)
        )

        locs = torch.cat(locs, dim=1)
        scales = torch.cat(scales, dim=1)

        return locs, scales

    def init_encoders(self):
        """Initialize model params."""
        for module in chain(
            self.loc_encoders.modules(),
            self.scale_encoders.modules(),
            self.bg_loc_encoder.modules(),
            self.bg_scale_encoder.modules(),
        ):
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)


class WhatDecoder(nn.Module):
    """Module decoding latent what code to individual images."""

    def __init__(self, z_what_size: int):
        super().__init__()
        self.h_size = z_what_size
        layers = [
            nn.ConvTranspose2d(self.h_size, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
            nn.Sigmoid(),
        ]
        self.decoder = nn.Sequential(*layers)

    def forward(self, z_what: torch.Tensor) -> torch.Tensor:
        """Takes z_what latent (sum_features(grid*grid) x z_what_size)
        .. and outputs decoded image (sum_features(grid*grid) x 3 x 64 x 64)
        """
        return self.decoder(z_what.view(-1, self.h_size, 1, 1))
