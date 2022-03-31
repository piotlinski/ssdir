"""Main function for SSDIR training."""
from argparse import ArgumentParser

import pyro
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_ssd.args import str2bool
from pytorch_ssd.modeling.model import SSD

from pytorch_ssdir.modeling import SSDIR


def main(hparams):
    """Main function that creates and trains SSDIR model."""
    if hparams.seed is not None:
        seed_everything(hparams.seed)
        pyro.set_rng_seed(hparams.seed)

    kwargs = vars(hparams)
    if hparams.ssd_checkpoint is not None:
        ssd = SSD.load_from_checkpoint(checkpoint_path=hparams.ssd_checkpoint, **kwargs)
    else:
        ssd = SSD(**kwargs)
    if hparams.ssdir_checkpoint is not None:
        model = SSDIR.load_from_checkpoint(
            checkpoint_path=hparams.ssdir_checkpoint, ssd_model=ssd, **kwargs
        )
    else:
        model = SSDIR(ssd_model=ssd, **kwargs)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="ckpt-{epoch:02d}-{val_loss:.2f}",
        save_top_k=hparams.n_checkpoints,
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=hparams.early_stopping_patience
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor_callback]
    if hparams.early_stopping:
        callbacks.append(early_stopping_callback)

    logger = WandbLogger(
        name=(
            f"{hparams.dataset_name}-"
            f"SSDIR-{model.encoder.ssd_backbone.__class__.__name__}-"
            f"seed{hparams.seed}"
        ),
        save_dir=hparams.default_root_dir,
        project="ssdir",
    )
    logger.watch(model, log=hparams.watch, log_freq=hparams.watch_freq)

    trainer = Trainer.from_argparse_args(hparams, logger=logger, callbacks=callbacks)
    trainer.tune(model)
    trainer.fit(model)


def cli():
    """SSDIR CLI with argparse."""
    parser = ArgumentParser(conflict_handler="resolve")
    parser.add_argument(
        "-s", "--seed", type=int, default=None, help="Random seed for training"
    )
    parser.add_argument(
        "-c",
        "--ssdir_checkpoint",
        type=str,
        default=None,
        help="Checkpoint to start training from",
    )
    parser.add_argument(
        "--ssd_checkpoint", type=str, default=None, help="SSD checkpoint file"
    )
    parser = SSDIR.add_model_specific_args(parser)
    parser.add_argument(
        "--n_checkpoints", type=int, default=3, help="Number of top checkpoints to save"
    )
    parser.add_argument(
        "--early_stopping",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable early stopping",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Number of epochs with no improvements before stopping early",
    )
    parser.add_argument(
        "--watch",
        type=str,
        default=None,
        help="Log model topology as well as optionally gradients and weights. "
        "Available options: None, gradients, parameters, all",
    )
    parser.add_argument(
        "--watch_freq",
        type=int,
        default=100,
        help="How often to perform model watch.",
    )
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
