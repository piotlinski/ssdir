"""SSDIR encoder, decoder, model and guide declarations."""
from functools import reduce
from operator import mul
from typing import Iterator, Optional, Tuple, Union

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
import torch.nn.functional as functional
from pyssd.config import CfgNode, get_config
from pyssd.modeling.checkpoint import CheckPointer
from pyssd.modeling.model import SSD

from ssdir.modeling.depth import DepthEncoder
from ssdir.modeling.present import PresentEncoder
from ssdir.modeling.what import WhatDecoder, WhatEncoder
from ssdir.modeling.where import WhereEncoder, WhereTransformer


class Encoder(nn.Module):
    """Module encoding input image to latent representation.

    .. latent representation consists of:
       - $$z_{what} ~ N(\\mu^{what}, \\sigma^{what})$$
       - $$z_{where} in R^4$$
       - $$z_{present} ~ Bernoulli(p_{present})$$
       - $$z_{depth} ~ N(\\mu_{depth}, \\sigma_{depth})$$
    """

    def __init__(self, ssd: SSD, z_what_size: int = 64):
        super().__init__()
        self.ssd_backbone = ssd.backbone
        self.what_enc = WhatEncoder(
            z_what_size=z_what_size,
            feature_channels=ssd.backbone.out_channels,
            feature_maps=ssd.predictor.config.DATA.PRIOR.FEATURE_MAPS,
        )
        self.where_enc = WhereEncoder(ssd_box_predictor=ssd.predictor)
        self.present_enc = PresentEncoder(ssd_box_predictor=ssd.predictor)
        self.depth_enc = DepthEncoder(feature_channels=ssd.backbone.out_channels)

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]], ...]:
        """Takes images tensors (batch_size x channels x image_size x image_size)
        .. and outputs latent representation tuple
        .. (z_what (loc & scale), z_where, z_present, z_depth (loc & scale))
        """
        features = self.ssd_backbone(images)
        z_where = self.where_enc(features)
        z_present = self.present_enc(features)
        z_what_loc, z_what_scale = self.what_enc(features)
        z_depth_loc, z_depth_scale = self.depth_enc(features)
        return (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        )


class Decoder(nn.Module):
    """Module decoding latent representation.

    .. Pipeline:
       - sort z_depth ascending
       - sort $$z_{what}$$, $$z_{where}$$, $$z_{present}$$ accordingly
       - decode $$z_{what}$$ where $$z_{present} = 1$$
       - transform decoded objects according to $$z_{where}$$
       - merge transformed images based on $$z_{depth}$$
    """

    def __init__(self, ssd: SSD, z_what_size: int = 64, drop_empty: bool = True):
        super().__init__()
        ssd_config = ssd.predictor.config
        self.what_dec = WhatDecoder(z_what_size=z_what_size)
        self.where_stn = WhereTransformer(image_size=ssd_config.DATA.SHAPE[0])
        self.indices = nn.Parameter(
            self.reconstruction_indices(ssd_config), requires_grad=False
        )
        self.drop = drop_empty
        self.empty_obj_const = nn.Parameter(torch.tensor(-1000.0), requires_grad=False)

    @staticmethod
    def reconstruction_indices(ssd_config: CfgNode) -> torch.Tensor:
        """Get indices for reconstructing images.

        .. Caters for the difference between z_what, z_depth and z_where, z_present.
        """
        indices = []
        img_idx = last_img_idx = 0
        for feature_map, boxes_per_loc in zip(
            ssd_config.DATA.PRIOR.FEATURE_MAPS, ssd_config.DATA.PRIOR.BOXES_PER_LOC
        ):
            for feature_map_idx in range(feature_map ** 2):
                img_idx = last_img_idx + feature_map_idx
                indices.append(
                    torch.full(
                        size=(boxes_per_loc,), fill_value=img_idx, dtype=torch.float
                    )
                )
            last_img_idx = img_idx + 1
        indices.append(
            torch.full(size=(1,), fill_value=last_img_idx, dtype=torch.float)
        )
        return torch.cat(indices, dim=0)

    @staticmethod
    def pad_indices(n_present: torch.Tensor) -> torch.Tensor:
        """Using number of objects in chunks create indices
        .. so that every chunk is padded to the same dimension.

        .. Assumes index 0 refers to "starter" (empty) object, added to every chunk.

        :param n_present: number of objects in each chunk
        :return: indices for padding tensors
        """
        end_idx = 1
        max_objects = torch.max(n_present)
        indices = []
        for chunk_objects in n_present:
            start_idx = end_idx
            end_idx = end_idx + chunk_objects
            idx_range = torch.arange(
                start=start_idx, end=end_idx, dtype=torch.long, device=n_present.device
            )
            indices.append(
                functional.pad(idx_range, pad=[1, max_objects - chunk_objects])
            )
        return torch.cat(indices)

    def _pad_reconstructions(
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
        padded_shape = max_present.item() + 1
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

    @staticmethod
    def merge_reconstructions(
        reconstructions: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Combine decoded images into one by weighted sum."""
        weighted_images = reconstructions * weights.view(*weights.shape[:2], 1, 1, 1)
        return torch.sum(weighted_images, dim=1)

    def reconstruct_objects(
        self,
        z_what: torch.Tensor,
        z_where: torch.Tensor,
        z_present: torch.Tensor,
        z_depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render reconstructions and their depths from batch."""
        z_what_shape = z_what.shape
        z_where_shape = z_where.shape
        z_depth_shape = z_depth.shape
        if self.drop:
            present_mask = z_present == 1
            n_present = torch.sum(present_mask, dim=1).squeeze(-1)
            z_what = z_what[present_mask.expand_as(z_what)].view(-1, z_what_shape[-1])
            z_where = z_where[present_mask.expand_as(z_where)].view(
                -1, z_where_shape[-1]
            )
            z_depth = z_depth[present_mask.expand_as(z_depth)].view(
                -1, z_depth_shape[-1]
            )
        z_what_flat = z_what.view(-1, z_what.shape[-1])
        z_where_flat = z_where.view(-1, z_where.shape[-1])
        decoded_images = self.what_dec(z_what_flat)
        transformed_images = self.where_stn(decoded_images, z_where_flat)
        if self.drop:
            reconstructions, depths = self._pad_reconstructions(
                transformed_images=transformed_images,
                z_depth=z_depth,
                n_present=n_present,
            )
        else:
            reconstructions = transformed_images.view(
                -1,
                z_what_shape[1],
                3,
                self.where_stn.image_size,
                self.where_stn.image_size,
            )
            depths = z_depth.where(z_present == 1.0, self.empty_obj_const)
        return reconstructions, depths

    def pad_latents(
        self, latents: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad latents according to Decoder's settings."""
        z_what, z_where, z_present, z_depth = latents
        # repeat rows to match z_where and z_present
        z_what = z_what.index_select(dim=1, index=self.indices.long())
        z_depth = z_depth.index_select(dim=1, index=self.indices.long())
        return z_what, z_where, z_present, z_depth

    def forward(
        self, latents: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Takes latent variables tensors tuple (z_what, z_where, z_present, z_depth)
        .. and outputs reconstructed images batch
        .. (batch_size x channels x image_size x image_size)
        """
        z_what, z_where, z_present, z_depth = self.pad_latents(latents)
        # render reconstructions
        reconstructions, depths = self.reconstruct_objects(
            z_what, z_where, z_present, z_depth
        )
        # merge reconstructions
        return self.merge_reconstructions(
            reconstructions=reconstructions, weights=functional.softmax(depths, dim=1)
        )


class SSDIR(nn.Module):
    """Single-Shot Detect, Infer, Repeat."""

    def __init__(
        self,
        z_what_size: int = 64,
        ssd_config: Optional[CfgNode] = None,
        ssd_model_file: str = ("checkpoint.pth"),
        z_where_scale_eps: float = 1e-5,
        z_present_p_prior: float = 0.01,
        z_where_prior: float = 0.5,
        drop_empty: bool = True,
    ):
        super().__init__()
        if ssd_config is None:
            ssd_config = get_config()
        ssd_model = SSD(config=ssd_config)
        ssd_checkpointer = CheckPointer(config=ssd_config, model=ssd_model)
        ssd_checkpointer.load(filename=ssd_model_file)

        self.z_what_size = z_what_size
        self.n_objects = sum(
            features ** 2 for features in ssd_config.DATA.PRIOR.FEATURE_MAPS
        )
        self.n_ssd_features = sum(
            boxes * features ** 2
            for features, boxes in zip(
                ssd_config.DATA.PRIOR.FEATURE_MAPS, ssd_config.DATA.PRIOR.BOXES_PER_LOC
            )
        )
        self.z_where_scale_eps = z_where_scale_eps
        self.z_present_p_prior = z_present_p_prior
        self.z_where_prior = z_where_prior

        self.encoder = Encoder(ssd=ssd_model, z_what_size=z_what_size)
        self.decoder = Decoder(
            ssd=ssd_model, z_what_size=z_what_size, drop_empty=drop_empty
        )

    def encoder_forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform forward pass through encoder network."""
        (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        ) = self.encoder(inputs)
        z_what = dist.Normal(z_what_loc, z_what_scale).sample()
        z_present = dist.Bernoulli(z_present).sample()
        z_depth = dist.Normal(z_depth_loc, z_depth_scale).sample()
        return z_what, z_where, z_present, z_depth

    def decoder_forward(
        self, latents: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Perform forward pass through decoder network."""
        outputs = self.decoder(latents)
        return outputs

    def model(self, x: torch.Tensor):
        """Pyro model; $$P(x|z)P(z)$$."""
        with poutine.scale(scale=1 / reduce(mul, x.shape[1:])):
            pyro.module("decoder", self.decoder)
            batch_size = x.shape[0]

            z_what_loc = x.new_zeros(batch_size, self.n_objects, self.z_what_size)
            z_what_scale = torch.ones_like(z_what_loc)

            z_where_loc = x.new_full(
                (batch_size, self.n_ssd_features, 4), fill_value=self.z_where_prior
            )
            z_where_scale = torch.full_like(
                z_where_loc, fill_value=self.z_where_scale_eps
            )

            z_present_p = x.new_full(
                (batch_size, self.n_ssd_features, 1),
                fill_value=self.z_present_p_prior,
            )

            z_depth_loc = x.new_zeros((batch_size, self.n_objects, 1))
            z_depth_scale = torch.ones_like(z_depth_loc)

            with pyro.plate("data"):
                z_what = pyro.sample(
                    "z_what", dist.Normal(z_what_loc, z_what_scale).to_event(2)
                )
                z_where = pyro.sample(
                    "z_where", dist.Normal(z_where_loc, z_where_scale).to_event(2)
                )
                z_present = pyro.sample(
                    "z_present", dist.Bernoulli(z_present_p).to_event(2)
                )
                z_depth = pyro.sample(
                    "z_depth", dist.Normal(z_depth_loc, z_depth_scale).to_event(2)
                )

                output = self.decoder((z_what, z_where, z_present, z_depth))

                pyro.sample(
                    "obs",
                    dist.Bernoulli(output.view(batch_size, -1)).to_event(1),
                    obs=x.view(batch_size, -1),
                )

    def guide(self, x: torch.Tensor):
        """Pyro guide; $$q(z|x)$$."""
        with poutine.scale(scale=1 / reduce(mul, x.shape[1:])):
            pyro.module("encoder", self.encoder)
            with pyro.plate("data"):
                (
                    (z_what_loc, z_what_scale),
                    z_where_loc,
                    z_present_p,
                    (z_depth_loc, z_depth_scale),
                ) = self.encoder(x)
                z_where_scale = torch.full_like(
                    z_where_loc, fill_value=self.z_where_scale_eps
                )

                pyro.sample("z_what", dist.Normal(z_what_loc, z_what_scale).to_event(2))
                pyro.sample(
                    "z_where", dist.Normal(z_where_loc, z_where_scale).to_event(2)
                )
                pyro.sample("z_present", dist.Bernoulli(z_present_p).to_event(2))
                pyro.sample(
                    "z_depth", dist.Normal(z_depth_loc, z_depth_scale).to_event(2)
                )

    def filtered_parameters(
        self,
        include: Optional[str] = None,
        exclude: Optional[str] = None,
        recurse: bool = True,
    ) -> Iterator[nn.Parameter]:
        """Get filtered SSDIR parameters for the optimizer.
        :param include: parameter name part to include
        :param exclude: parameter name part to exclude
        :param recurse: iterate recursively through model parameters
        :return: iterator of filtered parameters
        """
        for name, param in self.named_parameters(recurse=recurse):
            if include is not None and include not in name:
                continue
            if exclude is not None and exclude in name:
                continue
            yield param
