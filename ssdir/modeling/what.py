"""$$z_{what}$$ encoder and decoder."""
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
        self.bg_loc_encoders, self.bg_loc_merger = self._build_what_bg_encoder()
        self.bg_scale_encoders, self.bg_scale_merger = self._build_what_bg_encoder()

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

    def _build_what_bg_encoder(self) -> Tuple[nn.ModuleList, nn.Module]:
        """Build conv layers list for encoding background what latent."""
        conv_layers = [
            nn.Conv2d(
                in_channels=channels,
                out_channels=self.h_size,
                kernel_size=maps,
                stride=3,
                padding=1,
            )
            for channels, maps in zip(self.feature_channels, self.feature_maps)
        ]
        lin_layer = nn.Linear(
            in_features=len(self.feature_channels) * self.h_size,
            out_features=self.h_size,
        )
        return nn.ModuleList(conv_layers), lin_layer

    def forward(
        self, features: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Takes tuple of tensors (batch_size x grid x grid x features)
        .. and outputs locs and scales tensors
        .. (batch_size x sum_features(grid*grid) x z_what_size)
        """
        locs = []
        scales = []
        bg_locs = []
        bg_scales = []
        batch_size = features[0].shape[0]
        for feature, loc_enc, scale_enc, bg_loc_enc, bg_scale_enc in zip(
            features,
            self.loc_encoders,
            self.scale_encoders,
            self.bg_loc_encoders,
            self.bg_scale_encoders,
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
            bg_locs.append(bg_loc_enc(feature).view(batch_size, -1))
            bg_scales.append(bg_scale_enc(feature).view(batch_size, -1))

        locs.append(self.bg_loc_merger(torch.cat(bg_locs, dim=1)).unsqueeze(1))
        scales.append(
            functional.softplus(
                self.bg_scale_merger(torch.cat(bg_scales, dim=1)).unsqueeze(1)
            )
        )

        locs = torch.cat(locs, dim=1)
        scales = torch.cat(scales, dim=1)

        return locs, scales


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
