"""Test SSDIR model."""
import pytest
import torch

from pytorch_ssdir.modeling.model import SSDIR


@pytest.mark.parametrize("z_what_size", [2, 4])
@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("background", [True, False])
def test_ssdir_encoder_forward(
    z_what_size, batch_size, background, ssd_model, n_ssd_features
):
    """Verify SSDIR encoder_forward output dimensions and dtypes."""
    model = SSDIR(
        ssd_model=ssd_model,
        dataset_name="MNIST",
        data_dir="test",
        z_what_size=z_what_size,
        batch_size=batch_size,
        background=background,
    )

    data_shape = (3, *ssd_model.image_size)
    inputs = torch.rand(batch_size, *data_shape)

    latents = model.encoder_forward(inputs)
    z_what, z_where, z_present, z_depth = latents

    assert z_what.shape == (batch_size, n_ssd_features + background * 1, z_what_size)
    assert z_what.dtype == torch.float
    assert z_where.shape == (batch_size, n_ssd_features, 4)
    assert z_where.dtype == torch.float
    assert z_present.shape == (batch_size, n_ssd_features, 1)
    assert z_present.dtype == torch.float
    assert z_depth.shape == (batch_size, n_ssd_features, 1)
    assert z_depth.dtype == torch.float


def test_ssdir_normalize_output():
    """Verify normalizing output fits 0-1."""
    outputs = torch.rand(5, 3, 8, 8)
    normalized = SSDIR.normalize_output(outputs)
    assert (torch.max(normalized.view(5, -1), dim=1)[0] == 1).all()


@pytest.mark.parametrize("z_what_size", [2, 4])
@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize("background", [True, False])
def test_ssdir_decoder_forward(
    z_what_size, batch_size, drop, background, ssd_model, n_ssd_features
):
    """Verify SSDIR decoder_forward output dimensions and dtypes."""
    model = SSDIR(
        ssd_model=ssd_model,
        dataset_name="MNIST",
        data_dir="test",
        z_what_size=z_what_size,
        batch_size=batch_size,
        background=background,
    )

    z_what = torch.rand(batch_size, n_ssd_features + background * 1, z_what_size)
    z_where = torch.rand(batch_size, n_ssd_features, 4)
    z_present = torch.randint(0, 1, (batch_size, n_ssd_features, 1))
    z_depth = torch.rand(batch_size, n_ssd_features, 1)
    latents = (z_what, z_where, z_present, z_depth)
    outputs = model.decoder_forward(latents)

    data_shape = (3, *ssd_model.image_size)
    assert outputs.shape == (batch_size, *data_shape)
    assert outputs.dtype == torch.float


@pytest.mark.parametrize("z_what_size", [2, 4])
@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("background", [True, False])
def test_ssdir_forward(z_what_size, batch_size, background, ssd_model, n_ssd_features):
    model = SSDIR(
        ssd_model=ssd_model,
        dataset_name="MNIST",
        data_dir="test",
        z_what_size=z_what_size,
        batch_size=batch_size,
        background=background,
    )

    data_shape = (3, *ssd_model.image_size)
    inputs = torch.rand(batch_size, *data_shape)

    outputs = model(inputs)

    assert outputs.shape == inputs.shape
