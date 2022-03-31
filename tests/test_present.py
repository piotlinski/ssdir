"""Test present modules."""
import torch

from pytorch_ssdir.modeling.present import PresentEncoder


def test_present_encoder_dimensions(ssd_model, ssd_features, n_ssd_features):
    """Verify PresentEncoder output dimensions."""
    encoder = PresentEncoder(ssd_box_predictor=ssd_model.predictor)
    outputs = encoder(ssd_features)
    assert outputs.shape == (4, n_ssd_features, 1)


def test_present_encoder_dtype(ssd_model, ssd_features, n_ssd_features):
    """Verify PresentEncoder output types."""
    encoder = PresentEncoder(ssd_box_predictor=ssd_model.predictor)
    outputs = encoder(ssd_features)
    assert outputs.dtype == torch.float
    assert (outputs >= 0).all()
    assert (outputs <= 1).all()
