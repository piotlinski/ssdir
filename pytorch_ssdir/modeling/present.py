"""$$z_{present}$$ encoder."""
from typing import Tuple

import torch
import torch.nn as nn
from pytorch_ssd.modeling.box_predictors import SSDBoxPredictor


class PresentEncoder(nn.Module):
    """Module encoding input image features to present latent param."""

    def __init__(
        self, ssd_box_predictor: SSDBoxPredictor, normalize_probas: bool = False
    ):
        super().__init__()
        self.normalize_probas = normalize_probas
        self.ssd_cls_headers = ssd_box_predictor.cls_headers
        self.n_classes = ssd_box_predictor.n_classes

    def forward(self, features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Takes tuple of tensors (batch_size x grid x grid x features)
        .. and outputs probabilities tensor
        .. (batch_size x sum_features(grid*grid*n_boxes) x 1)
        """
        presents = []
        batch_size = features[0].shape[0]
        for feature, cls_header in zip(features, self.ssd_cls_headers):
            logits = torch.sigmoid(
                cls_header(feature)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, self.n_classes)
            )
            max_values, max_indices = torch.max(logits, dim=-1, keepdim=True)
            present = torch.zeros_like(max_values)
            presents.append(present.where(max_indices == 0, max_values))

        presents = torch.cat(presents, dim=1)

        if self.normalize_probas:
            maxes = torch.max(presents, dim=1)[0]
            maxes[maxes == 0.0] = 1.0
            presents = presents / maxes.unsqueeze(-1).expand_as(presents)

        return presents
