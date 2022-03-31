"""Common tests tools."""
import pytest
import torch
from pytorch_ssd.modeling.model import SSD


@pytest.fixture
def ssd_model():
    """Default SSD model."""
    return SSD(dataset_name="MNIST", data_dir="test", backbone_name="VGGLite")


@pytest.fixture
def ssd_features(ssd_model):
    """Sample ssd features tensors tuple."""
    features = [
        torch.rand(4, channels, feature_map, feature_map)
        for feature_map, channels in zip(
            ssd_model.backbone.feature_maps, ssd_model.backbone.out_channels
        )
    ]
    return tuple(features)


@pytest.fixture
def n_ssd_features(ssd_model):
    """Total number of ssd features."""
    return sum(
        boxes * features ** 2
        for features, boxes in zip(
            ssd_model.backbone.feature_maps, ssd_model.backbone.boxes_per_loc
        )
    )
