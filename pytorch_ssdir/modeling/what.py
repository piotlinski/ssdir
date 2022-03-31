"""$$z_{what}$$ encoder and decoder."""
from itertools import chain
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class WhatEncoder(nn.Module):
    """Module encoding input image features to what latent distribution params."""

    def __init__(
        self,
        z_what_size: int,
        n_hidden: int,
        feature_channels: List[int],
        feature_maps: List[int],
        z_what_scale_const: Optional[float] = None,
        background: bool = True,
    ):
        super().__init__()
        self.out_size = z_what_size
        self.feature_channels = feature_channels
        self.feature_maps = feature_maps
        self.n_hidden = n_hidden
        self.background = background
        feature_idx = self.feature_maps.index(min(self.feature_maps))
        self.loc_encoders = self._build_what_encoders()
        self.bg_loc_encoder = (
            self._build_feature_encoder(self.feature_channels[feature_idx])
            if background
            else None
        )
        self.z_what_scale_const = z_what_scale_const
        if z_what_scale_const is None:
            self.scale_encoders = self._build_what_encoders()
            self.bg_scale_encoder = (
                self._build_feature_encoder(self.feature_channels[feature_idx])
                if background
                else None
            )
        self.init_encoders()

    def _build_feature_encoder(self, in_channels: int) -> nn.Module:
        """Prepare single feature encoder."""
        hid_size = 2 * self.out_size
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=hid_size, kernel_size=1)
        ]
        for _ in range(self.n_hidden):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels=hid_size,
                        out_channels=hid_size,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.ReLU(),
                ]
            )
        layers.append(
            nn.Conv2d(in_channels=hid_size, out_channels=self.out_size, kernel_size=1)
        )
        return nn.Sequential(*layers)

    def _build_what_encoders(self) -> nn.ModuleList:
        """Build conv layers list for encoding backbone output."""
        layers = [
            self._build_feature_encoder(channels) for channels in self.feature_channels
        ]
        return nn.ModuleList(layers)

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
        sequence = [features, self.loc_encoders]
        if self.z_what_scale_const is None:
            sequence.append(self.scale_encoders)
        else:
            sequence.append(len(features) * [self.z_what_scale_const])
        for feature, loc_enc, scale_enc in zip(*sequence):
            locs.append(
                loc_enc(feature)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, self.out_size)
            )
            if self.z_what_scale_const is None:
                scales.append(
                    torch.exp(scale_enc(feature))
                    .permute(0, 2, 3, 1)
                    .contiguous()
                    .view(batch_size, -1, self.out_size)
                )
            else:
                scales.append(torch.full_like(locs[-1], fill_value=scale_enc))
        if self.background:
            bg_feature_idx = self.feature_maps.index(min(self.feature_maps))
            locs.append(
                self.bg_loc_encoder(features[bg_feature_idx])
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, self.out_size)
            )
            if self.z_what_scale_const is None:
                scales.append(
                    torch.exp(self.bg_scale_encoder(features[bg_feature_idx]))
                    .permute(0, 2, 3, 1)
                    .contiguous()
                    .view(batch_size, -1, self.out_size)
                )
            else:
                scales.append(
                    torch.full_like(locs[-1], fill_value=self.z_what_scale_const)
                )

        locs = torch.cat(locs, dim=1)
        scales = torch.cat(scales, dim=1)

        return locs, scales

    def init_encoders(self):
        """Initialize model params."""
        modules = [self.loc_encoders.modules()]
        if self.background:
            modules.append(self.bg_loc_encoder.modules())
        if self.z_what_scale_const is None:
            modules.append(self.scale_encoders.modules())
            if self.background:
                modules.append(self.bg_scale_encoder.modules())
        for module in chain(*modules):
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)


class WhatDecoder(nn.Module):
    """Module decoding latent what code to individual images."""

    def __init__(self, z_what_size: int):
        super().__init__()
        self.h_size = z_what_size
        layers = [
            nn.Conv2d(self.h_size, 1024, kernel_size=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid(),
        ]
        self.decoder = nn.Sequential(*layers)
        self.init_decoder()

    def forward(self, z_what: torch.Tensor) -> torch.Tensor:
        """Takes z_what latent (sum_features(grid*grid) x z_what_size)
        .. and outputs decoded image (sum_features(grid*grid) x 3 x 64 x 64)
        """
        return self.decoder(z_what.view(-1, self.h_size, 1, 1))

    def init_decoder(self):
        """Initialize model params."""
        for module in self.decoder.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
