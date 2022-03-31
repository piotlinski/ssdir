"""$$z_{where}$$ encoder and decoder."""
import warnings
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
from pytorch_ssd.data.bboxes import convert_locations_to_boxes
from pytorch_ssd.modeling.box_predictors import SSDBoxPredictor

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)


class WhereEncoder(nn.Module):
    """Module encoding input image features to where latent params.

    .. converts regressional location results into boxes (center_x center_y, w, h)
       $$hat{center} * center_variance = \\frac {center - center_prior} {hw_prior}$$
       $$exp(hat{hw} * size_variance) = \\frac {hw} {hw_prior}$$
    """

    def __init__(
        self,
        ssd_box_predictor: SSDBoxPredictor,
        ssd_anchors: torch.Tensor,
        ssd_center_variance: float,
        ssd_size_variance: float,
        square_boxes: bool,
    ):
        super().__init__()
        self.ssd_loc_reg_headers = ssd_box_predictor.reg_headers
        self.register_buffer("anchors", ssd_anchors)
        self.center_variance = ssd_center_variance
        self.size_variance = ssd_size_variance
        self.square_boxes = square_boxes

    @staticmethod
    def convert_to_square(boxes: torch.Tensor) -> torch.Tensor:
        """Convert rectangular boxes to squares by taking max(height, width)."""
        wh = (
            (torch.argmax(boxes[..., 2:], dim=-1) + 2)
            .unsqueeze(-1)
            .expand(*boxes.shape[:-1], 2)
        )
        xy = wh.new_tensor([0, 1]).expand_as(wh)
        index = torch.cat((xy, wh), dim=-1)
        return torch.gather(boxes, dim=-1, index=index)

    def forward(self, features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Takes tuple of tensors (batch_size x grid x grid x features)
        .. and outputs bounding box parameters x_center, y_center, w, h tensor
        .. (batch_size x sum_features(grid*grid*n_boxes) x 4)
        """
        where = []
        batch_size = features[0].shape[0]
        for feature, reg_header in zip(features, self.ssd_loc_reg_headers):
            where.append(
                reg_header(feature)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, 4)
            )

        where_locations = torch.cat(where, dim=1)
        where_boxes = convert_locations_to_boxes(
            locations=where_locations,
            priors=self.anchors,
            center_variance=self.center_variance,
            size_variance=self.size_variance,
        )

        if self.square_boxes:
            where_boxes = self.convert_to_square(where_boxes)

        return where_boxes


class WhereTransformer(nn.Module):
    """Transforms WhereDecoder output image using where box."""

    def __init__(self, image_size: int, inverse: bool = False):
        super().__init__()
        self.image_size = image_size
        self.inverse = inverse

    @staticmethod
    def scale_boxes(where_boxes: torch.Tensor) -> torch.Tensor:
        """Adjust scaled XYWH boxes to STN format.

        .. t_{XY} = (1 - 2 * {XY}) * s_{WH}
           s_{WH} = 1 / {WH}

        :param where_boxes: latent - detection box
        :return: scaled box
        """
        xy = where_boxes[..., :2]
        wh = where_boxes[..., 2:]
        scaled_wh = 1 / wh
        scaled_xy = (1 - 2 * xy) * scaled_wh
        return torch.cat((scaled_xy, scaled_wh), dim=-1)

    @staticmethod
    def convert_boxes_to_theta(where_boxes: torch.Tensor) -> torch.Tensor:
        """Convert where latents to transformation matrix.

        .. [ w_scale    0    x_translation ]
           [    0    h_scale y_translation ]

        :param where_boxes: latent - detection box
        :return: transformation matrix for transposing and scaling
        """
        n_boxes = where_boxes.shape[0]
        transformation_mtx = torch.cat(
            (torch.zeros((n_boxes, 1), device=where_boxes.device), where_boxes), dim=1
        )
        return transformation_mtx.index_select(
            dim=1,
            index=torch.tensor([3, 0, 1, 0, 4, 2], device=where_boxes.device),
        ).view(n_boxes, 2, 3)

    @staticmethod
    def get_inverse_theta(theta: torch.Tensor) -> torch.Tensor:
        """Get inverse transformation matrix.

        :param theta: transformation matrix for transposing and scaling
        :return: inverted transformation matrix
        """
        last_row = theta.new_tensor([0.0, 0.0, 1.0]).expand(theta.shape[0], 1, 3)
        transformation_mtx = torch.cat((theta, last_row), dim=1)
        return transformation_mtx.inverse()[:, :-1]

    def forward(
        self, decoded_images: torch.Tensor, where_boxes: torch.Tensor
    ) -> torch.Tensor:
        """Takes decoded images (sum_features(grid*grid) x 3 x 64 x 64)
        .. and bounding box parameters x_center, y_center, w, h tensor
        .. (sum_features(grid*grid*n_boxes) x 4)
        .. and outputs transformed images
        .. (sum_features(grid*grid*n_boxes) x 3 x image_size x image_size)
        """
        n_objects = decoded_images.shape[0]
        channels = decoded_images.shape[1]
        if where_boxes.numel():
            scaled_boxes = self.scale_boxes(where_boxes)
            theta = self.convert_boxes_to_theta(where_boxes=scaled_boxes)
            if self.inverse:
                theta = self.get_inverse_theta(theta)
            grid = functional.affine_grid(
                theta=theta,
                size=[n_objects, channels, self.image_size, self.image_size],
            )
            transformed_images = functional.grid_sample(
                input=decoded_images,
                grid=grid,
            )
        else:
            transformed_images = decoded_images.view(
                -1, channels, self.image_size, self.image_size
            )
        return transformed_images
