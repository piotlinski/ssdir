"""Test SSDIR decoder."""
import pytest
import torch

from pytorch_ssdir.modeling.decoder import Decoder


@pytest.mark.parametrize(
    "n_present, expected",
    [
        (torch.tensor([1, 3, 2]), torch.tensor([1, 0, 0, 4, 2, 3, 6, 5, 0])),
        (torch.tensor([1, 2, 2]), torch.tensor([1, 0, 3, 2, 5, 4])),
        (torch.tensor([3, 1, 1]), torch.tensor([3, 1, 2, 4, 0, 0, 5, 0, 0])),
    ],
)
def test_pad_indices(n_present, expected, ssd_model):
    """Verify pad indices calculation."""
    decoder = Decoder(ssd=ssd_model, z_what_size=4, background=True)
    indices = decoder.pad_indices(n_present)
    assert indices.shape == (n_present.shape[0] * (torch.max(n_present)),)
    assert torch.max(indices) == torch.sum(n_present)
    assert torch.equal(indices, expected)


@pytest.mark.parametrize(
    "n_present, expected",
    [
        (torch.tensor([1, 3, 2]), torch.tensor([0, 1, 0, 0, 0, 2, 3, 4, 0, 5, 6, 0])),
        (torch.tensor([1, 2, 2]), torch.tensor([0, 1, 0, 0, 2, 3, 0, 4, 5])),
        (torch.tensor([3, 1, 1]), torch.tensor([0, 1, 2, 3, 0, 4, 0, 0, 0, 5, 0, 0])),
    ],
)
def test_pad_indices_no_background(n_present, expected, ssd_model):
    """Verify pad indices calculation."""
    decoder = Decoder(ssd=ssd_model, z_what_size=4, background=False)
    indices = decoder.pad_indices(n_present)
    assert indices.shape == (n_present.shape[0] * (torch.max(n_present) + 1),)
    assert torch.max(indices) == torch.sum(n_present)
    assert torch.equal(indices, expected)


def test_pad_reconstructions(ssd_model):
    """Verify padding reconstructions in Decoder."""
    decoder = Decoder(ssd=ssd_model, z_what_size=4)
    images = (
        torch.arange(1, 5, dtype=torch.float)
        .view(-1, 1, 1, 1)
        .expand(4, 3, decoder.where_stn.image_size, decoder.where_stn.image_size)
    )
    z_depth = torch.arange(5, 9, dtype=torch.float).view(-1, 1)
    n_present = torch.tensor([1, 3])
    padded_images, padded_z_depth = decoder.pad_reconstructions(
        transformed_images=images, z_depth=z_depth, n_present=n_present
    )
    assert padded_images.shape == (2, 3, 3, 300, 300)
    assert padded_z_depth.shape == (2, 3)
    assert torch.equal(padded_images[0][0], images[0])
    assert torch.equal(padded_images[0][1], padded_images[0][2])
    assert torch.equal(padded_images[1][0], images[3])
    assert torch.equal(padded_images[1][1], images[1])
    assert torch.equal(padded_images[1][2], images[2])
    assert padded_z_depth[0][0] == z_depth[0]
    assert padded_z_depth[0][1] == padded_z_depth[0][2] == -float("inf")
    assert padded_z_depth[1][0] == z_depth[3]
    assert padded_z_depth[1][1] == z_depth[1]
    assert padded_z_depth[1][2] == z_depth[2]


def test_reshape_reconstructions(ssd_model):
    """Verify reshaping reconstructions in no-drop Decoder."""
    decoder = Decoder(ssd=ssd_model, z_what_size=4)
    images = (
        torch.arange(1, 5, dtype=torch.float)
        .view(-1, 1, 1, 1)
        .expand(4, 3, decoder.where_stn.image_size, decoder.where_stn.image_size)
    )
    z_depth = torch.tensor([[1, 2], [3, 4]], dtype=torch.float).unsqueeze(-1)
    z_present = torch.tensor([[True, True], [False, True]]).unsqueeze(-1)
    reshaped_images, reshaped_z_depth = decoder.reshape_reconstructions(
        transformed_images=images, z_depth=z_depth, z_present=z_present
    )
    assert reshaped_images.shape == (2, 2, 3, 300, 300)
    assert reshaped_z_depth.shape == (2, 2, 1)
    assert torch.equal(reshaped_images[0][0], images[1])
    assert torch.equal(reshaped_images[0][1], images[0])
    assert torch.equal(reshaped_images[1][0], images[3])
    assert torch.equal(reshaped_images[1][1], images[2])
    assert reshaped_z_depth[0][0] == z_depth[0][1]
    assert reshaped_z_depth[0][1] == z_depth[0][0]
    assert reshaped_z_depth[1][0] == z_depth[1][1]
    assert reshaped_z_depth[1][1] == -float("inf")


@pytest.mark.parametrize(
    "inputs, weights, expected",
    [
        (
            torch.ones(1, 2, 3, 5, 5),
            torch.tensor([[0.3, 0.2]]),
            torch.ones(1, 3, 5, 5),
        ),
        (
            torch.ones(2, 3, 3, 2, 2),
            torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.4, 0.5]]),
            torch.ones(2, 3, 2, 2),
        ),
    ],
)
def test_merge_reconstructions(inputs, weights, expected, ssd_model):
    """Verify reconstructions merging."""
    decoder = Decoder(ssd=ssd_model, z_what_size=4)
    merged = decoder.merge_reconstructions(inputs, weights=weights)
    assert merged.shape == (inputs.shape[0], *inputs.shape[2:])
    assert torch.all(torch.le(torch.abs(merged - expected), 1e-3))


def test_fill_background():
    """Verify adding background to merged reconstructions."""
    background_1 = torch.full((1, 3, 1, 2), fill_value=0.3)
    background_2 = torch.full((1, 3, 1, 2), fill_value=0.7)
    backgrounds = torch.cat((background_1, background_2), dim=0)
    merged = torch.zeros((2, 3, 1, 2))
    merged[0, 1, 0, 1] = 0.4
    merged[1, 2, 0, 1] = 0.5
    merged[0, 2, 0, 0] = 0.9
    merged[1, 0, 0, 1] = 1.0
    filled = Decoder.fill_background(merged, backgrounds)
    assert filled[0, 1, 0, 1] == 0.4
    assert filled[1, 2, 0, 1] == 0.5
    assert filled[0, 2, 0, 0] == 0.9
    assert filled[1, 0, 0, 1] == 1.0
    assert (
        filled[0, 0, 0, 0]
        == filled[0, 0, 0, 1]
        == filled[0, 1, 0, 0]
        == filled[0, 2, 0, 1]
        == 0.3
    )
    assert (
        filled[1, 0, 0, 0]
        == filled[1, 1, 0, 0]
        == filled[1, 1, 0, 1]
        == filled[1, 2, 0, 0]
        == 0.7
    )


@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize("background", [True, False])
def test_handle_latents(background, drop, ssd_model):
    """Test latents are modified according to settings."""
    batch_size = 2
    n_objects = 4
    z_what_size = 3
    z_what = torch.rand(batch_size, n_objects + background * 1, z_what_size)
    z_where = torch.rand(batch_size, n_objects, 4)
    z_present = torch.randint(0, 1, (batch_size, n_objects, 1))
    z_depth = torch.rand(batch_size, n_objects, 1)
    decoder = Decoder(
        ssd=ssd_model, z_what_size=z_what_size, background=background, drop_empty=drop
    )
    new_z_what, new_z_where, new_z_present, new_z_depth = decoder.handle_latents(
        z_what, z_where, z_present, z_depth
    )
    if drop:
        shape = (torch.sum(z_present, dtype=torch.long) + batch_size * background,)
    else:
        shape = (batch_size, n_objects + background * 1)
    assert new_z_what.shape == (*shape, z_what_size)
    assert new_z_where.shape == (*shape, 4)
    assert new_z_present.shape == (batch_size, n_objects + background * 1, 1)
    assert new_z_depth.shape == (*shape, 1)


@pytest.mark.parametrize("z_what_size", [2, 3, 4])
@pytest.mark.parametrize("n_objects", [1, 2])
@pytest.mark.parametrize("batch_size", [2, 3])
def test_decode_objects(batch_size, n_objects, z_what_size, ssd_model):
    """Verify dimension of decoded objects."""
    decoder = Decoder(ssd=ssd_model, z_what_size=z_what_size)
    z_what = torch.rand(batch_size, n_objects, z_what_size)
    z_where = torch.rand(batch_size, n_objects, 4)
    decoded_images, z_where_flat = decoder.decode_objects(z_what, z_where)
    assert decoded_images.shape == (batch_size * n_objects, 3, 64, 64)
    assert z_where_flat.shape == (batch_size * n_objects, 4)


@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize("background", [True, False])
def test_transform_objects(background, drop, ssd_model):
    """Verify dimension of transformed objects."""
    decoder = Decoder(
        ssd=ssd_model, z_what_size=2, background=background, drop_empty=drop
    )
    decoded_images = torch.rand(6, 3, 64, 64)
    z_where_flat = torch.rand(6, 4)
    z_present = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.long)
    z_depth = torch.rand(6, 1) if drop else torch.rand(2, 3)
    reconstructions, depths = decoder.transform_objects(
        decoded_images, z_where_flat, z_present, z_depth
    )
    assert reconstructions.shape[2:] == (3, 300, 300)
    assert depths.shape[0] == reconstructions.shape[0]
    assert depths.shape[1] == reconstructions.shape[1]


@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize("background", [True, False])
def test_decoder_dimensions(batch_size, background, ssd_model, n_ssd_features):
    """Verify decoder output dimensions."""
    z_what_size = 3
    z_what = torch.rand(batch_size, n_ssd_features + background * 1, z_what_size)
    z_where = torch.rand(batch_size, n_ssd_features, 4)
    z_present = torch.randint(0, 1, (batch_size, n_ssd_features, 1))
    z_depth = torch.rand(batch_size, n_ssd_features, 1)
    inputs = (z_what, z_where, z_present, z_depth)
    decoder = Decoder(ssd=ssd_model, z_what_size=z_what_size, background=background)
    outputs = decoder(inputs)
    assert outputs.shape == (batch_size, 3, *ssd_model.image_size)


@pytest.mark.parametrize("train_what", [True, False])
def test_disabling_decoder_modules(train_what, ssd_model):
    """Verify if disabling encoder modules influences requires_grad attribute."""
    decoder = Decoder(ssd=ssd_model, train_what=train_what)
    assert all(
        param.requires_grad == train_what for param in decoder.what_dec.parameters()
    )
