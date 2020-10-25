"""Test what modules."""
import pytest
import torch

from ssdir.modeling.what import WhatDecoder, WhatEncoder


@pytest.mark.parametrize("z_what_size", [8, 10, 13])
@pytest.mark.parametrize("feature_channels", [[5], [3, 7], [2, 4, 8]])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("grid_size", [3, 5, 7])
def test_what_encoder_dimensions(z_what_size, feature_channels, batch_size, grid_size):
    """Verify what encoder z dimensions."""
    inputs = [
        torch.rand(batch_size, feature_channel, grid_size, grid_size)
        for feature_channel in feature_channels
    ]
    encoder = WhatEncoder(
        z_what_size=z_what_size,
        feature_channels=feature_channels,
        feature_maps=[grid_size] * len(feature_channels),
    )
    locs, scales = encoder(inputs)
    assert (
        locs.shape
        == scales.shape
        == (batch_size, len(feature_channels) * grid_size ** 2 + 1, z_what_size)
    )


def test_what_encoder_dtype():
    """Verify what encoder output dtype."""
    inputs = [torch.rand(3, 4, 5, 5)]
    encoder = WhatEncoder(z_what_size=7, feature_channels=[4], feature_maps=[5])
    locs, scales = encoder(inputs)
    assert locs.dtype == torch.float
    assert scales.dtype == torch.float
    assert (scales > 0).all()


@pytest.mark.parametrize("z_what_size", [2, 4, 5])
@pytest.mark.parametrize("n_objects", [1, 4, 9])
def test_what_decoder_dimensions(z_what_size, n_objects):
    """Verify if what decoder output dimensions."""
    z_whats = torch.rand(n_objects, z_what_size)
    decoder = WhatDecoder(z_what_size=z_what_size)
    outputs = decoder(z_whats)
    assert outputs.shape == (n_objects, 3, 64, 64)


def test_what_decoder_dtype():
    """Verify what decoder output dtype."""
    z_whats = torch.rand(3, 4, 5)
    decoder = WhatDecoder(z_what_size=5)
    outputs = decoder(z_whats)
    assert outputs.dtype == torch.float
    assert (outputs >= 0).all()
    assert (outputs <= 1).all()
