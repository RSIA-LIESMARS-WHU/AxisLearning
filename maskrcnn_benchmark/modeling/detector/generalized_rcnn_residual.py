# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn_residual import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """
 
    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        # resnet fpn
        self.backbone_fpn = build_backbone(cfg)
        self.backbone = self.backbone_fpn.body
        self.fpn = self.backbone_fpn.fpn

        # rpn fcos
        self.rpn = build_rpn(cfg, self.backbone.out_channels, self.backbone_fpn.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        # R-50-FPN-RETINANET FPN
        backbone_features = self.backbone(images.tensors)
        # for f in backbone_features:
        #     print(f.shape)
        # exit(0)
        fpn_features = self.fpn(backbone_features)
        proposals, proposal_losses = self.rpn(images, backbone_features[1:], fpn_features , targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            #FOCS 代替rpn
            x = fpn_features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
