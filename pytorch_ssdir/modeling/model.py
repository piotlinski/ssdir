"""SSDIR model and guide declarations."""
import warnings
from argparse import ArgumentParser
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import PIL.Image as PILImage
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from pyro.infer import Trace_ELBO
from pytorch_ssd.args import str2bool
from pytorch_ssd.data.datasets import datasets
from pytorch_ssd.data.transforms import DataTransform, TrainDataTransform
from pytorch_ssd.modeling.model import SSD
from pytorch_ssd.modeling.visualize import denormalize
from torch.utils.data.dataloader import DataLoader

from pytorch_ssdir.args import parse_kwargs
from pytorch_ssdir.modeling.decoder import Decoder
from pytorch_ssdir.modeling.encoder import Encoder
from pytorch_ssdir.modeling.where import WhereTransformer
from pytorch_ssdir.run.loss import per_site_loss
from pytorch_ssdir.run.transforms import corner_to_center_target_transform

warnings.filterwarnings(
    "ignore",
    message="^.* was not registered in the param store because requires_grad=False",
)

optimizers = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
lr_schedulers = {
    # name: optimizer, interval, var to monitor
    "StepLR": (torch.optim.lr_scheduler.StepLR, "step", None),
    "MultiStepLR": (torch.optim.lr_scheduler.MultiStepLR, "step", None),
    "ExponentialLR": (torch.optim.lr_scheduler.ExponentialLR, "epoch", None),
    "CosineAnnealingLR": (torch.optim.lr_scheduler.CosineAnnealingLR, "step", None),
    "ReduceLROnPlateau": (
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        "epoch",
        "val_loss",
    ),
    "CyclicLR": (torch.optim.lr_scheduler.CyclicLR, "step", None),
    "CosineAnnealingWarmRestarts": (
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "step",
        None,
    ),
}


class SSDIR(pl.LightningModule):
    """Single-Shot Detect, Infer, Repeat."""

    def __init__(
        self,
        ssd_model: SSD,
        optimizer: str = "Adam",
        optimizer_kwargs: Optional[List[Tuple[str, Any]]] = None,
        learning_rate: float = 1e-3,
        ssd_lr_multiplier: float = 1.0,
        lr_scheduler: str = "",
        lr_scheduler_kwargs: Optional[List[Tuple[str, Any]]] = None,
        auto_lr_find: bool = False,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        z_what_size: int = 64,
        z_what_hidden: int = 2,
        z_present_p_prior: float = 0.01,
        drop: bool = True,
        drop_too_small: bool = True,
        strong_crop: bool = False,
        square_boxes: bool = False,
        background: bool = True,
        normalize_z_present: bool = False,
        z_what_scale_const: Optional[float] = None,
        z_depth_scale_const: Optional[float] = None,
        normalize_elbo: bool = False,
        what_coef: float = 1.0,
        present_coef: float = 1.0,
        depth_coef: float = 1.0,
        rec_coef: float = 1.0,
        obs_coef: float = 1.0,
        score_boxes_only: bool = False,
        normalize_reconstructions: bool = True,
        train_what_encoder: bool = True,
        train_what_decoder: bool = True,
        train_where: bool = True,
        train_present: bool = True,
        train_depth: bool = True,
        train_backbone: bool = True,
        train_backbone_layers: int = -1,
        clone_backbone: bool = False,
        reset_non_present: bool = True,
        visualize_inference: bool = True,
        visualize_inference_freq: int = 500,
        n_visualize_objects: int = 10,
        visualize_latents: bool = True,
        visualize_latents_freq: int = 5,
        **_kwargs,
    ):
        """
        :param ssd_model: trained SSD to use as backbone
        :param optimizer: optimizer name
        :param learning_rate: learning rate
        :param optimizer_kwargs: optimizer argumnets dictionary
        :param ssd_lr_multiplier: ssd learning rate multiplier (learning rate * mult)
        :param lr_scheduler: LR scheduler name
        :param lr_scheduler_kwargs: LR scheduler arguments dictionary
        :param auto_lr_find: perform auto lr finding
        :param batch_size: mini-batch size for training
        :param num_workers: number of workers for dataloader
        :param pin_memory: pin memory for training
        :param z_what_size: latent what size
        :param z_what_hidden: number of extra hidden layers for what encoder
        :param z_present_p_prior: present prob prior
        :param drop: drop empty objects' latents
        :param drop_too_small: remove objects that are smaller than 4% of image
        :param strong_crop: crop input image additionally (for small objects datasets)
        :param square_boxes: use square boxes instead of rectangular
        :param background: learn background latents
        :param normalize_z_present: normalize z_present probabilities
        :param z_what_scale_const: fixed z_what scale (if None - use NN to model)
        :param z_depth_scale_const: fixed z_depth scale (if None - use NN to model)
        :param normalize_elbo: normalize elbo components by tenors' numels
        :param what_coef: z_what loss component coefficient
        :param present_coef: z_present loss component coefficient
        :param depth_coef: z_depth loss component coefficient
        :param rec_coef: reconstruction error component coefficient (per-object)
        :param obs_coef: reconstruction error component coefficient (entire image)
        :param score_boxes_only: score reconstructions only inside bounding boxes
        :param normalize_reconstructions: normalize reconstructions before scoring
        :param train_what_encoder: train what encoder
        :param train_what_decoder: train what decoder
        :param train_where: train where encoder
        :param train_present: train present encoder
        :param train_depth: train depth encoder
        :param train_backbone: train ssd backbone
        :param train_backbone_layers: n layers to train in the backbone (neg for all)
        :param clone_backbone: clone backbone for depth and what encoders
        :param reset_non_present: set non-present latents to some ordinary ones
        :param visualize_inference: visualize inference
        :param visualize_inference_freq: how often to visualize inference
        :param n_visualize_objects: number of objects to visualize
        :param visualize_latents: visualize model latents
        :param visualize_latents_freq: how often to visualize latents
        """
        super().__init__()

        self.encoder = Encoder(
            ssd=ssd_model,
            z_what_size=z_what_size,
            z_what_hidden=z_what_hidden,
            z_what_scale_const=z_what_scale_const,
            z_depth_scale_const=z_depth_scale_const,
            square_boxes=square_boxes,
            train_what=train_what_encoder,
            train_where=train_where,
            train_present=train_present,
            train_depth=train_depth,
            train_backbone=train_backbone,
            train_backbone_layers=train_backbone_layers,
            clone_backbone=clone_backbone,
            reset_non_present=reset_non_present,
            background=background,
            normalize_z_present=normalize_z_present,
        )
        self.decoder = Decoder(
            ssd=ssd_model,
            z_what_size=z_what_size,
            drop_empty=drop,
            train_what=train_what_decoder,
            background=background,
        )

        self.optimizer = optimizers[optimizer]
        if optimizer_kwargs is None:
            optimizer_kwargs = []
        self.optimizer_kwargs = dict(optimizer_kwargs)
        self.lr = learning_rate
        self.ssd_lr_multiplier = ssd_lr_multiplier
        self.lr_scheduler: Optional[object]
        self.lr_freq: Optional[str]
        self.lr_metric: Optional[str]
        self.lr_scheduler, self.lr_freq, self.lr_metric = lr_schedulers.get(
            lr_scheduler, (None, None, None)
        )
        if lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = []
        self.lr_scheduler_kwargs = dict(lr_scheduler_kwargs)
        self.auto_lr_find = auto_lr_find
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pixel_means = ssd_model.backbone.PIXEL_MEANS
        self.pixel_stds = ssd_model.backbone.PIXEL_STDS
        self._mse_train = pl.metrics.MeanSquaredError()
        self._mse_val = pl.metrics.MeanSquaredError()
        self.mse = {"train": self._mse_train, "val": self._mse_val}

        self.image_size = ssd_model.image_size
        self.flip_train = ssd_model.flip_train
        self.augment_colors_train = ssd_model.augment_colors_train
        self.strong_crop = strong_crop
        self.drop_too_small = drop_too_small
        self.dataset = ssd_model.dataset
        self.data_dir = ssd_model.data_dir

        self.z_what_size = z_what_size
        self.z_present_p_prior = z_present_p_prior
        self.drop = drop
        self.background = background

        self.normalize_elbo = normalize_elbo
        self._what_coef = what_coef
        self._present_coef = present_coef
        self._depth_coef = depth_coef
        self._rec_coef = rec_coef
        self._obs_coef = obs_coef
        self.score_boxes_only = score_boxes_only
        self.normalize_reconstructions = normalize_reconstructions
        self.rec_stn = WhereTransformer(image_size=64, inverse=True)

        self.reset_non_present = reset_non_present
        self.visualize_inference = visualize_inference
        self.visualize_inference_freq = visualize_inference_freq
        self.n_visualize_objects = n_visualize_objects
        self.visualize_latents = visualize_latents
        self.visualize_latents_freq = visualize_latents_freq

        self.n_ssd_features = sum(
            boxes * features ** 2
            for features, boxes in zip(
                ssd_model.backbone.feature_maps, ssd_model.backbone.boxes_per_loc
            )
        )

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        """Add SSDIR args to parent argument parser."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--dataset_name",
            type=str,
            default="MNIST",
            help=f"Used dataset name. Available: {list(datasets.keys())}",
        )
        parser.add_argument(
            "--data_dir", type=str, default="data", help="Dataset files directory"
        )
        parser.add_argument(
            "--optimizer",
            type=str,
            default="Adam",
            help=f"Used optimizer. Available: {list(optimizers.keys())}",
        )
        parser.add_argument(
            "--optimizer_kwargs",
            type=parse_kwargs,
            default=[],
            nargs="*",
            help="Optimizer kwargs in the form of key=value separated by spaces",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-3,
            help="Learning rate used for training the model",
        )
        parser.add_argument(
            "--ssd_lr_multiplier",
            type=float,
            default=1.0,
            help="Learning rate multiplier for training SSD backbone",
        )
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            default="None",
            help=(
                "Used LR scheduler. "
                f"Available: {list(lr_schedulers.keys())}; default: None"
            ),
        )
        parser.add_argument(
            "--lr_scheduler_kwargs",
            type=parse_kwargs,
            default=[],
            nargs="*",
            help="LR scheduler kwargs in the form of key=value separated by spaces",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="Mini-batch size used for training the model",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="Number of workers used to load the dataset",
        )
        parser.add_argument(
            "--pin_memory",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Pin data in memory while training",
        )
        parser.add_argument(
            "--z_what_size", type=int, default=64, help="z_what latent size"
        )
        parser.add_argument(
            "--z_what_hidden",
            type=int,
            default=2,
            help="Number of what encoder hidden layers",
        )
        parser.add_argument(
            "--z_present_p_prior",
            type=float,
            default=0.01,
            help="z_present probability prior",
        )
        parser.add_argument(
            "--drop",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Drop empty objects' latents",
        )
        parser.add_argument(
            "--drop_too_small",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Remove objects that are smaller than 4% of image",
        )
        parser.add_argument(
            "--strong_crop",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Crop input image additionally (for small objects datasets)",
        )
        parser.add_argument(
            "--square_boxes",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Use square boxes only",
        )
        parser.add_argument(
            "--background",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Learn background latents",
        )
        parser.add_argument(
            "--normalize_z_present",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Normalize z_present probabilities",
        )
        parser.add_argument(
            "--z_what_scale_const",
            type=float,
            default=None,
            help="constant z_what scale",
        )
        parser.add_argument(
            "--z_depth_scale_const",
            type=float,
            default=None,
            help="constant z_depth scale",
        )
        parser.add_argument(
            "--normalize_elbo",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Normalize elbo components by tenors' numels",
        )
        parser.add_argument(
            "--what_coef",
            type=float,
            default=1.0,
            help="z_what loss component coefficient",
        )
        parser.add_argument(
            "--present_coef",
            type=float,
            default=1.0,
            help="z_present loss component coefficient",
        )
        parser.add_argument(
            "--depth_coef",
            type=float,
            default=1.0,
            help="z_depth loss component coefficient",
        )
        parser.add_argument(
            "--rec_coef",
            type=float,
            default=1.0,
            help="Reconstruction error component coefficient (per object)",
        )
        parser.add_argument(
            "--obs_coef",
            type=float,
            default=1.0,
            help="Reconstruction error component coefficient (entire image)",
        )
        parser.add_argument(
            "--score_boxes_only",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Score reconstructions only inside bounding boxes",
        )
        parser.add_argument(
            "--normalize_reconstructions",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Normalize reconstructions before scoring",
        )
        parser.add_argument(
            "--train_what_encoder",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Train what encoder",
        )
        parser.add_argument(
            "--train_what_decoder",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Train what decoder",
        )
        parser.add_argument(
            "--train_where",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Train where encoder",
        )
        parser.add_argument(
            "--train_present",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Train present encoder",
        )
        parser.add_argument(
            "--train_depth",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Train depth encoder",
        )
        parser.add_argument(
            "--train_backbone",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Train SSD backbone",
        )
        parser.add_argument(
            "--train_backbone_layers",
            type=int,
            default=-1,
            help="Number of final layers to train in the backbone (negative for all)",
        )
        parser.add_argument(
            "--clone_backbone",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Clone SSD backbone for what and depth encoders",
        )
        parser.add_argument(
            "--reset_non_present",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Reset non-present objects' latents to more ordinary",
        )
        parser.add_argument(
            "--flip_train",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Flip train images during training",
        )
        parser.add_argument(
            "--augment_colors_train",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Perform random colors augmentation during training",
        )
        parser.add_argument(
            "--visualize_inference",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Log visualizations of model predictions",
        )
        parser.add_argument(
            "--visualize_inference_freq",
            type=int,
            default=500,
            help="How often to perform inference visualization.",
        )
        parser.add_argument(
            "--n_visualize_objects",
            type=int,
            default=5,
            help="Number of objects to visualize",
        )
        parser.add_argument(
            "--visualize_latents",
            type=str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Log visualizations of model latents",
        )
        parser.add_argument(
            "--visualize_latents_freq",
            type=int,
            default=10,
            help="How often to perform latents visualization.",
        )
        return parser

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

    @staticmethod
    def normalize_output(reconstructions: torch.tensor) -> torch.tensor:
        """Normalize output to fit range 0-1."""
        batch_size = reconstructions.shape[0]
        max_values, _ = reconstructions.view(batch_size, -1).max(dim=1, keepdim=True)
        max_values = max_values.unsqueeze(-1).unsqueeze(-1).expand_as(reconstructions)
        return reconstructions / max_values

    def decoder_forward(
        self, latents: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Perform forward pass through decoder network."""
        outputs = self.decoder(latents)
        return self.normalize_output(outputs)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Pass data through the model."""
        latents = self.encoder_forward(images)
        return self.decoder_forward(latents)

    @property
    def what_coef(self) -> float:
        """Calculate what sampling elbo coefficient."""
        coef = self._what_coef
        if self.normalize_elbo:
            coef /= (
                self.batch_size
                * (self.n_ssd_features + self.background * 1)
                * self.z_what_size
            )
        return coef

    @property
    def present_coef(self) -> float:
        """Calculate present sampling elbo coefficient."""
        coef = self._present_coef
        if self.normalize_elbo:
            coef /= self.batch_size * (self.n_ssd_features + self.background * 1)
        return coef

    @property
    def depth_coef(self) -> float:
        """Calculate depth sampling elbo coefficient."""
        coef = self._depth_coef
        if self.normalize_elbo:
            coef /= self.batch_size * (self.n_ssd_features + self.background * 1)
        return coef

    @property
    def rec_coef(self) -> float:
        """Calculate reconstruction sampling elbo coefficient (per-object).

        .. note: needs to be divided by the number of reconstructed objects.
        """
        coef = self._rec_coef
        if self.normalize_elbo:
            coef /= self.batch_size * 3 * 64 * 64
        return coef

    @property
    def obs_coef(self) -> float:
        """Calculate reconstruction sampling elbo coefficient (entire image)."""
        coef = self._obs_coef
        if self.normalize_elbo:
            coef /= self.batch_size * 3 * self.image_size[0] * self.image_size[1]
        return coef

    def model(self, x: torch.Tensor):
        """Pyro model; $$P(x|z)P(z)$$."""
        pyro.module("decoder", self.decoder)
        batch_size = x.shape[0]
        _, z_where, *_ = self.encoder(x)

        with pyro.plate("data", batch_size):
            z_what_loc = x.new_zeros(
                batch_size, self.n_ssd_features + self.background * 1, self.z_what_size
            )
            z_what_scale = torch.ones_like(z_what_loc)

            z_present_p = x.new_full(
                (batch_size, self.n_ssd_features, 1),
                fill_value=self.z_present_p_prior,
            )

            z_depth_loc = x.new_zeros((batch_size, self.n_ssd_features, 1))
            z_depth_scale = torch.ones_like(z_depth_loc)

            with poutine.scale(scale=self.what_coef):
                z_what = pyro.sample(
                    "z_what", dist.Normal(z_what_loc, z_what_scale).to_event(2)
                )

            with poutine.scale(scale=self.present_coef):
                z_present = pyro.sample(
                    "z_present", dist.Bernoulli(z_present_p).to_event(2)
                )

            with poutine.scale(scale=self.depth_coef):
                z_depth = pyro.sample(
                    "z_depth", dist.Normal(z_depth_loc, z_depth_scale).to_event(2)
                )

            obs = denormalize(
                x.permute(0, 2, 3, 1),
                pixel_mean=self.pixel_means,
                pixel_std=self.pixel_stds,
            )

            # all stages from decoder.forward
            z_what, z_where, z_present, z_depth = self.decoder.handle_latents(
                z_what, z_where, z_present, z_depth
            )
            if torch.sum(z_present) == 0:
                raise ValueError("No object present in batch")

            decoded_images, z_where_flat = self.decoder.decode_objects(z_what, z_where)
            reconstructions, depths = self.decoder.transform_objects(
                decoded_images, z_where_flat, z_present, z_depth
            )
            output = self.decoder.merge_reconstructions(
                reconstructions, depths
            ).permute(0, 2, 3, 1)

            if self.score_boxes_only:
                mask = output != 0
                output = torch.where(mask, output, output.new_tensor(0.0))
                obs = torch.where(mask, obs, obs.new_tensor(0.0))
            if self.normalize_reconstructions:
                output = self.normalize_output(output)
            with poutine.scale(scale=self.obs_coef):
                pyro.sample("obs", dist.Bernoulli(output).to_event(3), obs=obs)

        with pyro.plate("reconstructions"):
            if self.rec_coef:
                n_present = (
                    torch.sum(z_present, dim=1, dtype=torch.long).squeeze(-1)
                    if self.drop
                    else z_present.new_tensor(z_present.shape[0] * [z_present.shape[1]])
                )
                obs_idx = torch.repeat_interleave(
                    torch.arange(
                        n_present.numel(),
                        dtype=n_present.dtype,
                        device=n_present.device,
                    ),
                    n_present,
                )
                transformed_obs = self.rec_stn(
                    obs[obs_idx].permute(0, 3, 1, 2), z_where_flat
                )
                with poutine.scale(scale=self.rec_coef / n_present.sum()):
                    pyro.sample(
                        "rec",
                        dist.Bernoulli(decoded_images).to_event(3),
                        obs=transformed_obs,
                    )

    def guide(self, x: torch.Tensor):
        """Pyro guide; $$q(z|x)$$."""
        pyro.module("encoder", self.encoder)
        batch_size = x.shape[0]

        with pyro.plate("data", batch_size):
            (
                (z_what_loc, z_what_scale),
                z_where,
                z_present_p,
                (z_depth_loc, z_depth_scale),
            ) = self.encoder(x)

            with poutine.scale(scale=self.what_coef):
                pyro.sample("z_what", dist.Normal(z_what_loc, z_what_scale).to_event(2))

            with poutine.scale(scale=self.present_coef):
                pyro.sample("z_present", dist.Bernoulli(z_present_p).to_event(2))

            with poutine.scale(scale=self.depth_coef):
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

    def get_inference_visualization(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        reconstruction: torch.Tensor,
        z_where: torch.Tensor,
        objects: torch.Tensor,
    ) -> Tuple[PILImage.Image, Dict[str, Any]]:
        """Create model inference visualization."""
        denormalized_image = denormalize(
            image.permute(1, 2, 0),
            pixel_mean=self.pixel_means,
            pixel_std=self.pixel_stds,
        )
        vis_image = PILImage.fromarray(
            (denormalized_image.cpu().numpy() * 255).astype(np.uint8)
        )
        vis_reconstruction = PILImage.fromarray(
            (reconstruction.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        vis_objects = objects[: self.n_visualize_objects].squeeze(1)
        inference_image = PILImage.new(
            "RGB",
            (
                vis_image.width
                + vis_objects.shape[0] * vis_objects.shape[-1]
                + vis_reconstruction.width,
                max(vis_image.height, vis_reconstruction.height, vis_objects.shape[-2]),
            ),
            "white",
        )
        inference_image.paste(vis_image, (0, 0))

        output = vis_objects.new_zeros(vis_objects.shape[1:])
        for idx, obj in enumerate(vis_objects):
            filtered_obj = obj * torch.where(output == 0, 1.0, 0.3)
            output += filtered_obj
            obj_image = PILImage.fromarray(
                (filtered_obj.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
            inference_image.paste(
                obj_image, (vis_image.width + idx * obj_image.width, 0)
            )

        inference_image.paste(
            vis_reconstruction,
            (
                vis_image.width + vis_objects.shape[0] * vis_objects.shape[-1],
                0,
            ),
        )
        wandb_inference_boxes = {
            "gt": {
                "box_data": [
                    {
                        "position": {
                            "middle": (
                                box[0].int().item(),
                                box[1].int().item(),
                            ),
                            "width": box[2].int().item(),
                            "height": box[3].int().item(),
                        },
                        "domain": "pixel",
                        "box_caption": "gt_object",
                        "class_id": 1,
                    }
                    for box in boxes * self.image_size[-1]
                ],
                "class_labels": {1: "object"},
            },
            "where": {
                "box_data": [
                    {
                        "position": {
                            "middle": (
                                box[0].int().item()
                                + vis_image.width
                                + vis_objects.shape[0] * vis_objects.shape[-1],
                                box[1].int().item(),
                            ),
                            "width": box[2].int().item(),
                            "height": box[3].int().item(),
                        },
                        "domain": "pixel",
                        "box_caption": "object",
                        "class_id": 1,
                    }
                    for box in z_where * self.image_size[-1]
                ],
                "class_labels": {1: "object"},
            },
        }
        return inference_image, wandb_inference_boxes

    def common_run_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_nb: int,
        stage: str,
    ):
        """Common model running step for training and validation."""
        criterion = Trace_ELBO().differentiable_loss

        images, boxes, _ = batch
        try:
            loss = criterion(self.model, self.guide, images)
            self.log(f"{stage}_loss", loss, prog_bar=False, logger=True)
            for site, site_loss in per_site_loss(
                self.model, self.guide, images
            ).items():
                self.log(f"{stage}_loss_{site}", site_loss, prog_bar=False, logger=True)
        except ValueError as ex:
            loss = torch.tensor(float("NaN"))

        if torch.isnan(loss):
            print("Skipping training with this batch due to NaN loss.")
            return None

        vis_images = images.detach()
        vis_boxes = boxes.detach()

        if (
            self.global_step % self.visualize_inference_freq == 0
            or self.global_step % self.trainer.log_every_n_steps == 0
            or batch_nb == 0
        ):
            with torch.no_grad():
                latents = self.encoder_forward(vis_images)
                reconstructions = self.decoder_forward(latents)

                self.logger.experiment.log(
                    {
                        f"{stage}_mse": self.mse[stage](
                            reconstructions.permute(0, 2, 3, 1),
                            denormalize(
                                vis_images.permute(0, 2, 3, 1),
                                pixel_mean=self.pixel_means,
                                pixel_std=self.pixel_stds,
                            ),
                        )
                    },
                    step=self.global_step,
                )

                if self.visualize_latents:
                    z_what, z_where, z_present, z_depth = latents
                    z_what, z_where, z_present, z_depth = self.decoder.handle_latents(
                        z_what[0].unsqueeze(0),
                        z_where[0].unsqueeze(0),
                        z_present[0].unsqueeze(0),
                        z_depth[0].unsqueeze(0),
                    )
                    decoded_image, z_where_flat = self.decoder.decode_objects(
                        z_what, z_where
                    )
                    objects, depths = self.decoder.transform_objects(
                        decoded_image,
                        z_where_flat,
                        z_present,
                        z_depth,
                    )
                    _, sort_index = torch.sort(depths, dim=1, descending=True)
                    sorted_objects = objects.gather(
                        dim=1,
                        index=sort_index.view(1, -1, 1, 1, 1).expand_as(objects),
                    )

                    (
                        inference_image,
                        wandb_inference_boxes,
                    ) = self.get_inference_visualization(
                        image=vis_images[0],
                        boxes=vis_boxes[0],
                        reconstruction=reconstructions[0],
                        z_where=z_where,
                        objects=sorted_objects[0],
                    )
                    self.logger.experiment.log(
                        {
                            f"{stage}_inference_image": wandb.Image(
                                inference_image,
                                boxes=wandb_inference_boxes,
                                caption="model inference",
                            )
                        },
                        step=self.global_step,
                    )

        if self.visualize_latents and (
            self.global_step % self.visualize_latents_freq == 0 or batch_nb == 0
        ):
            with torch.no_grad():
                (
                    (z_what_loc, z_what_scale),
                    z_where,
                    z_present_p,
                    (z_depth_loc, z_depth_scale),
                ) = self.encoder(vis_images)
            if self.reset_non_present:
                present_mask = torch.gt(z_present_p, 1e-3)
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
                z_what_loc = z_what_loc[what_present_mask.expand_as(z_what_loc)].view(
                    -1, z_what_loc.shape[-1]
                )
                z_what_scale = z_what_scale[
                    what_present_mask.expand_as(z_what_scale)
                ].view(-1, z_what_scale.shape[-1])
                z_where = z_where[present_mask.expand_as(z_where)].view(
                    -1, z_where.shape[-1]
                )
                z_depth_loc = z_depth_loc[present_mask].view(-1, z_depth_loc.shape[-1])
                z_depth_scale = z_depth_scale[present_mask].view(
                    -1, z_depth_scale.shape[-1]
                )
            latents_dict = {
                "z_what_loc": z_what_loc,
                "z_what_scale": z_what_scale,
                "z_where": z_where,
                "z_present_p": z_present_p,
                "z_depth_loc": z_depth_loc,
                "z_depth_scale": z_depth_scale,
            }
            for latent_name, latent in latents_dict.items():
                self.logger.experiment.log(
                    {f"{stage}_{latent_name}": wandb.Histogram(latent.cpu())},
                    step=self.global_step,
                )

        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_nb: int
    ):
        """Step for training."""
        return self.common_run_step(batch, batch_nb, stage="train")

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_nb: int
    ):
        """Step for validation."""
        return self.common_run_step(batch, batch_nb, stage="val")

    def configure_optimizers(self):
        """Configure training optimizer."""
        if self.ssd_lr_multiplier != 1:
            optimizer_params = [
                {"params": self.filtered_parameters(exclude="ssd")},
                {
                    "params": self.filtered_parameters(include="ssd"),
                    "lr": self.lr * self.ssd_lr_multiplier,
                },
            ]
        else:
            optimizer_params = self.parameters()

        optimizer = self.optimizer(
            optimizer_params, lr=self.lr, **self.optimizer_kwargs
        )
        configuration = {"optimizer": optimizer}
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(
                optimizer=optimizer, **self.lr_scheduler_kwargs
            )
            configuration["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                "interval": self.lr_freq,
            }
            if self.lr_metric is not None:
                configuration["lr_scheduler"]["monitor"] = self.lr_metric
        return configuration

    def train_dataloader(self) -> DataLoader:
        """Prepare train dataloader."""
        data_transform = TrainDataTransform(
            image_size=self.image_size,
            pixel_mean=self.pixel_means,
            pixel_std=self.pixel_stds,
            flip=self.flip_train,
            augment_colors=self.augment_colors_train,
            strong_crop=self.strong_crop,
        )
        dataset = self.dataset(
            self.data_dir,
            data_transform=data_transform,
            target_transform=partial(
                corner_to_center_target_transform, drop_too_small=self.drop_too_small
            ),
            subset="train",
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Prepare validation dataloader."""
        data_transform = DataTransform(
            image_size=self.image_size,
            pixel_mean=self.pixel_means,
            pixel_std=self.pixel_stds,
        )
        dataset = self.dataset(
            self.data_dir,
            data_transform=data_transform,
            target_transform=partial(
                corner_to_center_target_transform, drop_too_small=self.drop_too_small
            ),
            subset="test",
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
