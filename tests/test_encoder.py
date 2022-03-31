"""Test SSDIR encoder"""
import pytest
import torch

from pytorch_ssdir.modeling.encoder import Encoder


@pytest.mark.parametrize("z_what_size", [2, 4])
@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("background", [True, False])
def test_encoder_dimensions(
    z_what_size, batch_size, background, ssd_model, n_ssd_features
):
    """Verify encoder output dimensions."""
    inputs = torch.rand(batch_size, 3, 300, 300)
    encoder = Encoder(ssd=ssd_model, z_what_size=z_what_size, background=background)
    (
        (z_what_loc, z_what_scale),
        z_where,
        z_present,
        (z_depth_loc, z_depth_scale),
    ) = encoder(inputs)
    assert (
        z_what_loc.shape
        == z_what_scale.shape
        == (batch_size, n_ssd_features + background * 1, z_what_size)
    )
    assert z_where.shape == (batch_size, n_ssd_features, 4)
    assert z_present.shape == (batch_size, n_ssd_features, 1)
    assert z_depth_loc.shape == z_depth_scale.shape == (batch_size, n_ssd_features, 1)


@pytest.mark.parametrize(
    "modules_enabled",
    [
        [True, True, True, True, True],
        [True, False, False, False, False],
        [True, True, True, True, False],
        [False, False, False, False, False],
    ],
)
def test_disabling_encoder_modules(modules_enabled, ssd_model):
    """Verify if disabling encoder modules influences requires_grad attribute."""
    kwargs_keys = [
        "train_what",
        "train_where",
        "train_present",
        "train_depth",
        "train_backbone",
    ]
    module_names = ["what_enc", "where_enc", "present_enc", "depth_enc", "ssd_backbone"]
    encoder = Encoder(ssd=ssd_model, **dict(zip(kwargs_keys, modules_enabled)))
    for name, requires_grad in zip(module_names, modules_enabled):
        assert all(
            param.requires_grad == requires_grad
            for param in getattr(encoder, name).parameters()
        )


@pytest.mark.parametrize("n_trained", [0, 1, 2, 5])
def test_disabling_backbone_layers(n_trained, ssd_model):
    """Verify if disabling encoder backbone layers disables it effectively."""
    encoder = Encoder(
        ssd=ssd_model, train_backbone=True, train_backbone_layers=n_trained
    )
    for idx, module in enumerate(encoder.ssd_backbone.children()):
        if idx < n_trained:
            assert all(param.requires_grad is True for param in module.parameters())
        else:
            assert all(param.requires_grad is False for param in module.parameters())


def test_cloning_backbone(ssd_model):
    """Verify if disabling encoder backbone layers disables it effectively."""
    encoder = Encoder(ssd=ssd_model, clone_backbone=True)
    assert len(list(encoder.ssd_backbone_cloned.children())) <= len(
        list(encoder.ssd_backbone.children())
    )


@pytest.mark.parametrize("train_backbone", [False, True])
def test_cloning_grads(train_backbone, ssd_model):
    """Verify if train_backbone is used appropriately for backbone and cloned."""
    encoder = Encoder(
        ssd=ssd_model,
        train_backbone=train_backbone,
        clone_backbone=True,
    )
    assert all(
        param.requires_grad is train_backbone
        for param in encoder.ssd_backbone.parameters()
    )
    assert all(
        param.requires_grad is True
        for param in encoder.ssd_backbone_cloned.parameters()
    )
    for backbone_child, cloned_child in zip(
        list(encoder.ssd_backbone.children()),
        list(encoder.ssd_backbone_cloned.children()),
    ):
        assert backbone_child is not cloned_child
        for backbone_param, cloned_param in zip(
            backbone_child.parameters(), cloned_child.parameters()
        ):
            assert backbone_param is not cloned_param


def test_latents_indices(ssd_model, n_ssd_features):
    """Verify latents indices calculation."""
    indices = Encoder.latents_indices(
        feature_maps=ssd_model.backbone.feature_maps,
        boxes_per_loc=ssd_model.backbone.boxes_per_loc,
    )
    assert indices.shape == (n_ssd_features,)
    assert indices.unique().numel() == sum(
        features ** 2 for features in ssd_model.backbone.feature_maps
    )
    assert (torch.sort(indices)[0] == indices).all()


@pytest.mark.parametrize("background", [True, False])
def test_pad_latents(background, ssd_model, n_ssd_features):
    """Verify if latents are padded appropriately."""
    n_features = sum(features ** 2 for features in ssd_model.backbone.feature_maps)
    encoder = Encoder(ssd=ssd_model, background=background)
    z_what_loc = (
        torch.arange(n_features + background * 1, dtype=torch.float)
        .view(1, -1, 1)
        .expand(1, n_features + background * 1, 4)
    )
    z_what_scale = (
        torch.arange(
            n_features + background * 1,
            2 * (n_features + background * 1),
            dtype=torch.float,
        )
        .view(1, -1, 1)
        .expand(1, n_features + background * 1, 4)
    )
    z_where = torch.zeros(1, n_ssd_features, dtype=torch.float)
    z_present = torch.zeros(1, n_ssd_features, dtype=torch.float)
    z_depth_loc = torch.arange(2 * n_features, 3 * n_features, dtype=torch.float).view(
        1, -1, 1
    )
    z_depth_scale = torch.arange(
        3 * n_features, 4 * n_features, dtype=torch.float
    ).view(1, -1, 1)
    (
        (new_z_what_loc, new_z_what_scale),
        _,
        _,
        (new_z_depth_loc, new_z_depth_scale),
    ) = encoder.pad_latents(
        ((z_what_loc, z_what_scale), z_where, z_present, (z_depth_loc, z_depth_scale))
    )
    assert (
        new_z_what_loc.shape
        == new_z_what_scale.shape
        == (1, n_ssd_features + background * 1, 4)
    )
    assert torch.equal(new_z_what_loc[0][0], new_z_what_loc[0][1])
    assert torch.equal(new_z_what_scale[0][2], new_z_what_scale[0][3])
    assert torch.equal(new_z_what_loc[0][8], new_z_what_loc[0][9])
    assert torch.equal(new_z_what_scale[0][16], new_z_what_scale[0][17])
    assert torch.equal(new_z_what_loc[0][400], new_z_what_loc[0][401])
    assert torch.equal(new_z_what_scale[0][562], new_z_what_scale[0][563])
    assert new_z_depth_loc.shape == new_z_depth_scale.shape == (1, n_ssd_features, 1)
    assert torch.equal(new_z_depth_loc[0][204], new_z_depth_loc[0][205])
    assert torch.equal(new_z_depth_scale[0][368], new_z_depth_scale[0][369])
    assert torch.equal(new_z_depth_loc[0][604], new_z_depth_loc[0][605])
    assert torch.equal(new_z_depth_scale[0][628], new_z_depth_scale[0][629])
    assert torch.equal(new_z_depth_loc[0][702], new_z_depth_loc[0][703])
    assert torch.equal(new_z_depth_scale[0][850], new_z_depth_scale[0][851])


@pytest.mark.parametrize("background", [True, False])
def test_reset_non_present(background, ssd_model):
    """Verify if appropriate latents are reset in encoder."""
    encoder = Encoder(ssd=ssd_model, background=background)
    z_what_loc = (
        torch.arange(1, 5 + background * 1, dtype=torch.float)
        .view(1, -1, 1)
        .expand(1, 4 + background * 1, 3)
    )
    z_what_scale = (
        torch.arange(5 + background * 1, 9 + background * 2, dtype=torch.float)
        .view(1, -1, 1)
        .expand(1, 4 + background * 1, 3)
    )
    z_where = torch.arange(1, 5, dtype=torch.float).view(1, -1, 1).expand(1, 4, 4)
    z_present = torch.tensor([1, 0, 0, 1], dtype=torch.float).view(1, -1, 1)
    z_depth_loc = torch.arange(5, 9, dtype=torch.float).view(1, -1, 1)
    z_depth_scale = torch.arange(9, 13, dtype=torch.float).view(1, -1, 1)
    (
        (reset_z_what_loc, reset_z_what_scale),
        reset_z_where,
        reset_z_present,
        (reset_z_depth_loc, reset_z_depth_scale),
    ) = encoder.reset_non_present(
        (
            (z_what_loc, z_what_scale),
            z_where,
            z_present,
            (z_depth_loc, z_depth_scale),
        )
    )
    assert torch.equal(reset_z_what_loc[0][0], z_what_loc[0][0])
    assert torch.equal(reset_z_what_loc[0][3], z_what_loc[0][3])
    assert torch.equal(reset_z_what_scale[0][0], z_what_scale[0][0])
    assert torch.equal(reset_z_what_scale[0][3], z_what_scale[0][3])
    assert torch.equal(reset_z_where[0][0], z_where[0][0])
    assert torch.equal(reset_z_where[0][3], z_where[0][3])
    assert torch.equal(reset_z_depth_loc[0][0], z_depth_loc[0][0])
    assert torch.equal(reset_z_depth_loc[0][3], z_depth_loc[0][3])
    assert torch.equal(reset_z_depth_scale[0][0], z_depth_scale[0][0])
    assert torch.equal(reset_z_depth_scale[0][3], z_depth_scale[0][3])
    assert (reset_z_what_loc[0][1] == reset_z_what_loc[0][2]).all()
    assert (reset_z_what_loc[0][1] == encoder.empty_loc).all()
    assert (reset_z_what_scale[0][1] == reset_z_what_scale[0][2]).all()
    assert (reset_z_what_scale[0][1] == encoder.empty_scale).all()
    assert (reset_z_where[0][1] == reset_z_where[0][2]).all()
    assert (reset_z_where[0][1] == encoder.empty_loc).all()
    assert (reset_z_depth_loc[0][1] == reset_z_depth_loc[0][2]).all()
    assert (reset_z_depth_loc[0][1] == encoder.empty_loc).all()
    assert (reset_z_depth_scale[0][1] == reset_z_depth_scale[0][2]).all()
    assert (reset_z_depth_scale[0][1] == encoder.empty_scale).all()
