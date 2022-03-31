"""SSDIR encoder."""
from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from pytorch_ssd.modeling.model import SSD

from pytorch_ssdir.modeling.depth import DepthEncoder
from pytorch_ssdir.modeling.present import PresentEncoder
from pytorch_ssdir.modeling.what import WhatEncoder
from pytorch_ssdir.modeling.where import WhereEncoder


class Encoder(nn.Module):
    """Module encoding input image to latent representation.

    .. latent representation consists of:
       - $$z_{what} ~ N(\\mu^{what}, \\sigma^{what})$$
       - $$z_{where} in R^4$$
       - $$z_{present} ~ Bernoulli(p_{present})$$
       - $$z_{depth} ~ N(\\mu_{depth}, \\sigma_{depth})$$
    """

    def __init__(
        self,
        ssd: SSD,
        z_what_size: int = 64,
        z_what_hidden: int = 2,
        z_what_scale_const: Optional[float] = None,
        z_depth_scale_const: Optional[float] = None,
        z_present_eps: float = 1e-3,
        square_boxes: bool = False,
        train_what: bool = True,
        train_where: bool = True,
        train_present: bool = True,
        train_depth: bool = True,
        train_backbone: bool = True,
        train_backbone_layers: int = -1,
        clone_backbone: bool = False,
        reset_non_present: bool = True,
        background: bool = True,
        normalize_z_present: bool = False,
    ):
        super().__init__()
        self.ssd_backbone = ssd.backbone.requires_grad_(train_backbone)
        self.clone_backbone = clone_backbone
        self.reset = reset_non_present
        self.background = background
        if self.clone_backbone:
            self.ssd_backbone_cloned = deepcopy(self.ssd_backbone).requires_grad_(True)
        if train_backbone_layers >= 0 and train_backbone:
            for module in list(self.ssd_backbone.children())[train_backbone_layers:][
                ::-1
            ]:
                module.requires_grad_(False)
        self.z_present_eps = z_present_eps
        self.what_enc = WhatEncoder(
            z_what_size=z_what_size,
            n_hidden=z_what_hidden,
            z_what_scale_const=z_what_scale_const,
            feature_channels=ssd.backbone.out_channels,
            feature_maps=ssd.backbone.feature_maps,
            background=background,
        ).requires_grad_(train_what)
        self.where_enc = WhereEncoder(
            ssd_box_predictor=ssd.predictor,
            ssd_anchors=ssd.anchors,
            ssd_center_variance=ssd.center_variance,
            ssd_size_variance=ssd.size_variance,
            square_boxes=square_boxes,
        ).requires_grad_(train_where)
        self.present_enc = PresentEncoder(
            ssd_box_predictor=ssd.predictor,
            normalize_probas=normalize_z_present,
        ).requires_grad_(train_present)
        self.depth_enc = DepthEncoder(
            feature_channels=ssd.backbone.out_channels,
            z_depth_scale_const=z_depth_scale_const,
        ).requires_grad_(train_depth)

        self.register_buffer(
            "indices",
            self.latents_indices(
                feature_maps=ssd.backbone.feature_maps,
                boxes_per_loc=ssd.backbone.boxes_per_loc,
            ),
        )
        self.register_buffer("empty_loc", torch.tensor(0.0, dtype=torch.float))
        self.register_buffer("empty_scale", torch.tensor(1.0, dtype=torch.float))

    @staticmethod
    def latents_indices(
        feature_maps: List[int], boxes_per_loc: List[int]
    ) -> torch.Tensor:
        """Get indices for reconstructing images.

        .. Caters for the difference between z_what, z_depth and z_where, z_present.
        """
        indices = []
        idx = 0
        for feature_map, n_boxes in zip(feature_maps, boxes_per_loc):
            for feature_map_idx in range(feature_map ** 2):
                indices.append(
                    torch.full(size=(n_boxes,), fill_value=idx, dtype=torch.float)
                )
                idx += 1
        return torch.cat(indices, dim=0)

    def pad_latents(
        self,
        latents: Tuple[
            Tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """Pad latents according to Encoder's settings."""
        (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        ) = latents
        # repeat rows to match z_where and z_present
        indices = self.indices.long()
        what_indices = indices
        if self.background:
            what_indices = torch.hstack((indices, indices.max() + 1))
        z_what_loc = z_what_loc.index_select(dim=1, index=what_indices)
        z_what_scale = z_what_scale.index_select(dim=1, index=what_indices)
        z_depth_loc = z_depth_loc.index_select(dim=1, index=indices)
        z_depth_scale = z_depth_scale.index_select(dim=1, index=indices)
        return (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        )

    def reset_non_present(
        self,
        latents: Tuple[
            Tuple[torch.Tensor, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """Reset latents, whose z_present is 0.

        .. note: this will set all "non-present" locs to 0. and scales to 1.
        """
        (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        ) = latents
        present_mask = torch.gt(z_present, self.z_present_eps)
        what_present_mask = present_mask
        if self.background:
            what_present_mask = torch.hstack(
                (
                    present_mask,
                    present_mask.new_full((1,), fill_value=True).expand(
                        present_mask.shape[0], 1, 1
                    ),
                )
            )
        z_what_loc = torch.where(what_present_mask, z_what_loc, self.empty_loc)
        z_what_scale = torch.where(what_present_mask, z_what_scale, self.empty_scale)
        z_where = torch.where(present_mask, z_where, self.empty_loc)
        z_depth_loc = torch.where(present_mask, z_depth_loc, self.empty_loc)
        z_depth_scale = torch.where(present_mask, z_depth_scale, self.empty_scale)
        return (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        )

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """Takes images tensors (batch_size x channels x image_size x image_size)
        .. and outputs latent representation tuple
        .. (z_what (loc & scale), z_where, z_present, z_depth (loc & scale))
        """
        where_present_features = self.ssd_backbone(images)
        if self.clone_backbone:
            what_depth_features = self.ssd_backbone_cloned(images)
        else:
            what_depth_features = where_present_features
        z_where = self.where_enc(where_present_features)
        z_present = self.present_enc(where_present_features)
        z_what_loc, z_what_scale = self.what_enc(what_depth_features)
        z_depth_loc, z_depth_scale = self.depth_enc(what_depth_features)
        latents = (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        )
        padded_latents = self.pad_latents(latents)
        if self.reset:
            padded_latents = self.reset_non_present(padded_latents)
        return padded_latents
