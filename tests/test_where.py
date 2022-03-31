"""Test where modules."""
import pytest
import torch

from pytorch_ssdir.modeling.where import WhereEncoder, WhereTransformer


def test_where_encoder_dimensions(ssd_model, ssd_features, n_ssd_features):
    """Verify WhereEncoder output dimensions."""
    encoder = WhereEncoder(
        ssd_box_predictor=ssd_model.predictor,
        ssd_anchors=ssd_model.anchors,
        ssd_center_variance=ssd_model.center_variance,
        ssd_size_variance=ssd_model.size_variance,
        square_boxes=False,
    )
    outputs = encoder(ssd_features)
    assert outputs.shape == (ssd_features[0].shape[0], n_ssd_features, 4)


def test_where_encoder_dtype(ssd_model, ssd_features):
    """Verify WhereEncoder output dtype."""
    encoder = WhereEncoder(
        ssd_box_predictor=ssd_model.predictor,
        ssd_anchors=ssd_model.anchors,
        ssd_center_variance=ssd_model.center_variance,
        ssd_size_variance=ssd_model.size_variance,
        square_boxes=False,
    )
    outputs = encoder(ssd_features)
    assert outputs.dtype == torch.float


@pytest.mark.parametrize(
    "rectangular, square",
    [
        (
            torch.tensor([[[1.0, 2.0, 5.0, 10.0], [2.0, 5.0, 9.0, 4.0]]]),
            torch.tensor([[[1.0, 2.0, 10.0, 10.0], [2.0, 5.0, 9.0, 9.0]]]),
        ),
        (
            torch.tensor([[[12.3, 34.2, 56.2, 13.4]], [[12.5, 65.3, 66.3, 66.3]]]),
            torch.tensor([[[12.3, 34.2, 56.2, 56.2]], [[12.5, 65.3, 66.3, 66.3]]]),
        ),
    ],
)
def test_convert_to_square(rectangular, square):
    """Check if rectangular bounding boxes are converted to max squares."""
    assert torch.equal(WhereEncoder.convert_to_square(rectangular), square)


@pytest.mark.parametrize("decoded_size", [2, 3])
@pytest.mark.parametrize("image_size", [7, 8])
@pytest.mark.parametrize("hidden_size", [5, 6])
def test_where_transformer_dimensions(decoded_size, image_size, hidden_size):
    """Verify WhereTransformer output dimensions."""
    decoded_images = torch.rand(hidden_size, 3, decoded_size, decoded_size)
    z_where = torch.rand(hidden_size, 4)
    transformer = WhereTransformer(image_size=image_size)
    outputs = transformer(decoded_images, z_where)
    assert outputs.shape == (hidden_size, 3, image_size, image_size)


@pytest.mark.parametrize(
    "theta, expected",
    [
        (
            torch.tensor([[[2.0, 0.0, 1.0], [0.0, 4.0, 2.0]]]),
            torch.tensor([[[0.5, 0.0, -0.5], [0.0, 0.25, -0.5]]]),
        ),
        (
            torch.tensor(
                [[[4.0, 0.0, 1.0], [0.0, 4.0, 0.5]], [[2.0, 0.0, 1.2], [0.0, 0.8, 0.4]]]
            ),
            torch.tensor(
                [
                    [[0.25, 0.0, -0.25], [0.0, 0.25, -0.125]],
                    [[0.5, 0.0, -0.6], [0.0, 1.25, -0.5]],
                ]
            ),
        ),
    ],
)
def test_get_inverse_theta(theta, expected):
    """Verify getting inverse transformation matrix."""
    inverted = WhereTransformer.get_inverse_theta(theta)
    assert torch.equal(inverted, expected)


def test_where_transformer_when_empty():
    """Verify WhereTransformer when given empty input."""
    decoded_size = 64
    image_size = 300
    decoded_images = torch.empty((0, 3, decoded_size, decoded_size))
    z_where = torch.empty(0, 4)
    transformer = WhereTransformer(image_size=image_size)
    output = transformer(decoded_images, z_where)
    assert output.shape == (0, 3, image_size, image_size)


@pytest.mark.parametrize(
    "boxes, expected",
    [
        (torch.tensor([1.0, 3.0, 2.0, 4.0]), torch.tensor([-0.5, -1.25, 0.5, 0.25])),
        (
            torch.tensor([[10.0, 12.0, 8.0, 10.0], [6.0, 4.0, 2.0, 5.0]]),
            torch.tensor([[-2.375, -2.3, 0.125, 0.1], [-5.5, -1.4, 0.5, 0.2]]),
        ),
        (
            torch.tensor([[[0.5, 0.2, 0.4, 0.5]]]),
            torch.tensor([[[0.0, 1.2, 2.5, 2.0]]]),
        ),
    ],
)
def test_scale_boxes(boxes, expected):
    """Verify converting xywh boxes to sxy."""
    transformer = WhereTransformer(image_size=1)
    scaled_boxes = transformer.scale_boxes(boxes)
    assert torch.equal(scaled_boxes, expected)


@pytest.mark.parametrize(
    "boxes, expected",
    [
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[[3.0, 0.0, 1.0], [0.0, 4.0, 2.0]]]),
        ),
        (
            torch.tensor([[5.0, 6.0, 7.0, 8.0], [2.0, 3.0, 9.0, 0.0]]),
            torch.tensor(
                [[[7.0, 0.0, 5.0], [0.0, 8.0, 6.0]], [[9.0, 0.0, 2.0], [0.0, 0.0, 3.0]]]
            ),
        ),
    ],
)
def test_expand_convert_boxes_to_theta(boxes, expected):
    """Verify expanding boxes to transformation matrix."""
    transformation_mtx = WhereTransformer.convert_boxes_to_theta(boxes)
    assert torch.equal(transformation_mtx, expected)
