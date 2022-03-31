"""Data transforms."""
from typing import Union

import numpy as np
import torch
import torch.nn.functional as functional
from pytorch_ssd.data.bboxes import corner_bbox_to_center_bbox


def corner_to_center_target_transform(
    boxes: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    drop_too_small: bool = False,
):
    """Convert ground truth boxes from corner to center form."""
    if drop_too_small and boxes.numel():
        boxes_mask_w = boxes[:, 2] - boxes[:, 0] > 0.04
        boxes_mask_h = boxes[:, 3] - boxes[:, 1] > 0.04
        boxes_mask = torch.logical_and(boxes_mask_w, boxes_mask_h)
        boxes = boxes[boxes_mask.unsqueeze(-1).expand_as(boxes)].view(-1, 4)
        labels = labels[boxes_mask]

    if boxes.numel() == 0:
        return torch.tensor([]), torch.tensor([])

    n_objs = boxes.shape[0]
    pad_size = 200

    return (
        functional.pad(
            corner_bbox_to_center_bbox(boxes).view(-1, 4), [0, 0, 0, pad_size - n_objs]
        ),
        functional.pad(labels, [0, pad_size - n_objs]),
    )
