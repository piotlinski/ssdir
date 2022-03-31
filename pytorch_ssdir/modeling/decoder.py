"""SSDIR decoder."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
from pytorch_ssd.modeling.model import SSD

from pytorch_ssdir.modeling.what import WhatDecoder
from pytorch_ssdir.modeling.where import WhereTransformer


class Decoder(nn.Module):
    """Module decoding latent representation.

    .. Pipeline:
       - sort z_depth ascending
       - sort $$z_{what}$$, $$z_{where}$$, $$z_{present}$$ accordingly
       - decode $$z_{what}$$ where $$z_{present} = 1$$
       - transform decoded objects according to $$z_{where}$$
       - merge transformed images based on $$z_{depth}$$
    """

    def __init__(
        self,
        ssd: SSD,
        z_what_size: int = 64,
        drop_empty: bool = True,
        train_what: bool = True,
        background: bool = True,
    ):
        super().__init__()
        self.what_dec = WhatDecoder(z_what_size=z_what_size).requires_grad_(train_what)
        self.where_stn = WhereTransformer(image_size=ssd.image_size[0])
        self.drop = drop_empty
        self.background = background
        if background:
            self.register_buffer("bg_depth", torch.zeros(1))
            self.register_buffer("bg_present", torch.ones(1))
            self.register_buffer("bg_where", torch.tensor([0.5, 0.5, 1.0, 1.0]))
        self.pixel_means = ssd.backbone.PIXEL_MEANS
        self.pixel_stds = ssd.backbone.PIXEL_STDS

    def pad_indices(self, n_present: torch.Tensor) -> torch.Tensor:
        """Using number of objects in chunks create indices
        .. so that every chunk is padded to the same dimension.

        .. Assumes index 0 refers to "starter" (empty) object
        .. Puts background index at the beginning of indices arange

        :param n_present: number of objects in each chunk
        :return: indices for padding tensors
        """
        end_idx = 1
        max_objects = torch.max(n_present)
        indices = []
        for chunk_objects in n_present:
            start_idx = end_idx
            end_idx = end_idx + chunk_objects
            if self.background:
                idx_range = torch.cat(
                    (
                        torch.tensor(
                            [end_idx - 1], dtype=torch.long, device=n_present.device
                        ),
                        torch.arange(
                            start=start_idx,
                            end=end_idx - 1,
                            dtype=torch.long,
                            device=n_present.device,
                        ),
                    ),
                    dim=0,
                )
                start_pad = 0
            else:
                idx_range = torch.arange(
                    start=start_idx,
                    end=end_idx,
                    dtype=torch.long,
                    device=n_present.device,
                )
                start_pad = 1
            indices.append(
                functional.pad(idx_range, pad=[start_pad, max_objects - chunk_objects])
            )
        return torch.cat(indices)

    def pad_reconstructions(
        self,
        transformed_images: torch.Tensor,
        z_depth: torch.Tensor,
        n_present: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad tensors to have identical 1. dim shape
        .. and reshape to (batch_size x n_objects x ...)
        """
        image_starter = transformed_images.new_zeros(
            (1, 3, self.where_stn.image_size, self.where_stn.image_size)
        )
        z_depth_starter = z_depth.new_full((1, 1), fill_value=-float("inf"))
        images = torch.cat((image_starter, transformed_images), dim=0)
        z_depth = torch.cat((z_depth_starter, z_depth), dim=0)
        max_present = torch.max(n_present)
        padded_shape = max_present.item() + (not self.background) * 1
        indices = self.pad_indices(n_present)
        images = images[indices].view(
            -1,
            padded_shape,
            3,
            self.where_stn.image_size,
            self.where_stn.image_size,
        )
        z_depth = z_depth[indices].view(-1, padded_shape)
        return images, z_depth

    def reshape_reconstructions(
        self,
        transformed_images: torch.Tensor,
        z_depth: torch.Tensor,
        z_present: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reshape no-drop reconstructions for merging."""
        batch_size = z_present.shape[0]
        images = transformed_images.view(
            batch_size,
            -1,
            3,
            self.where_stn.image_size,
            self.where_stn.image_size,
        )
        z_depth = z_depth.where(
            z_present == 1.0, z_depth.new_full((1,), fill_value=-float("inf"))
        )
        if self.background:
            images = torch.cat((images[:, [-1]], images[:, :-1]), dim=1)
            z_depth = torch.cat((z_depth[:, [-1]], z_depth[:, :-1]), dim=1)
        return images, z_depth

    def merge_reconstructions(
        self, reconstructions: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Combine decoder images into one by weighted sum."""
        if self.background:
            objects, object_weights = reconstructions[:, 1:], weights[:, 1:]
        else:
            objects, object_weights = reconstructions, weights
        weighted_images = objects * functional.softmax(object_weights, dim=1).view(
            *object_weights.shape[:2], 1, 1, 1
        )
        merged = torch.sum(weighted_images, dim=1)
        if self.background:
            merged = self.fill_background(
                merged=merged, backgrounds=reconstructions[:, 0]
            )
        return merged

    @staticmethod
    def fill_background(
        merged: torch.Tensor, backgrounds: torch.Tensor
    ) -> torch.Tensor:
        """Fill merged images background with background reconstruction."""
        mask = torch.where(merged < 1e-3, 1.0, 0.0)
        return merged + backgrounds * mask

    def handle_latents(
        self,
        z_what: torch.Tensor,
        z_where: torch.Tensor,
        z_present: torch.Tensor,
        z_depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Handle latents according to the model settings."""
        batch_size = z_what.shape[0]
        if self.background:  # append background latents
            z_depth = torch.cat(
                (z_depth, self.bg_depth.expand(batch_size, 1, 1)), dim=1
            )
            z_present = torch.cat(
                (z_present, self.bg_present.expand(batch_size, 1, 1)), dim=1
            )
            z_where = torch.cat(
                (z_where, self.bg_where.expand(batch_size, 1, 4)), dim=1
            )
        if self.drop:
            present_mask = torch.eq(z_present, 1)
            z_what = z_what[present_mask.expand_as(z_what)].view(-1, z_what.shape[-1])
            z_where = z_where[present_mask.expand_as(z_where)].view(
                -1, z_where.shape[-1]
            )
            z_depth = z_depth[present_mask.expand_as(z_depth)].view(
                -1, z_depth.shape[-1]
            )
        return z_what, z_where, z_present, z_depth

    def decode_objects(
        self, z_what: torch.Tensor, z_where: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode z_what to acquire individual objects and their z_where location."""
        z_what_flat = z_what.view(-1, z_what.shape[-1])
        z_where_flat = z_where.view(-1, z_where.shape[-1])
        decoded_images = self.what_dec(z_what_flat)
        return decoded_images, z_where_flat

    def transform_objects(
        self,
        decoded_images: torch.Tensor,
        z_where_flat: torch.Tensor,
        z_present: torch.Tensor,
        z_depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render reconstructions and their depths from batch."""
        n_present = (
            torch.sum(z_present, dim=1, dtype=torch.long).squeeze(-1)
            if self.drop
            else z_present.new_tensor(z_present.shape[0] * [z_present.shape[1]])
        )

        transformed_images = self.where_stn(decoded_images, z_where_flat)
        if self.drop:
            reconstructions, depths = self.pad_reconstructions(
                transformed_images=transformed_images,
                z_depth=z_depth,
                n_present=n_present,
            )
        else:
            reconstructions, depths = self.reshape_reconstructions(
                transformed_images=transformed_images,
                z_depth=z_depth,
                z_present=z_present,
            )
        return reconstructions, depths

    def forward(
        self, latents: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Takes latent variables tensors tuple (z_what, z_where, z_present, z_depth)
        .. and outputs reconstructed images batch
        .. (batch_size x channels x image_size x image_size)
        """
        z_what, z_where, z_present, z_depth = self.handle_latents(*latents)
        decoded_images, z_where_flat = self.decode_objects(z_what, z_where)
        reconstructions, depths = self.transform_objects(
            decoded_images, z_where_flat, z_present, z_depth
        )
        output = self.merge_reconstructions(reconstructions, depths)
        return output
