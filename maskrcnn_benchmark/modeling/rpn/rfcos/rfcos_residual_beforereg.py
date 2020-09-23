import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
# from .inference_reorg import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator
# from .loss_adap_range import make_fcos_loss_evaluator

from maskrcnn_benchmark.layers import Scale


class RFCOSHead(torch.nn.Module):
    def __init__(self, cfg, backbone_out_channels, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(RFCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 5, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        
        # residual
        residual_conv = []
        for feature_layer in range(3):
            residual_conv.append(nn.Conv2d(
                                    backbone_out_channels*2**feature_layer,
                                    in_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1
                                ))
        self.residual_conv = nn.ModuleList(residual_conv)

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness, 
                        self.residual_conv[0], self.residual_conv[1], self.residual_conv[2]]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        # self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, backbone_features, fpn_features):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(fpn_features):

            # 对FPN的结果进行卷积
            cls_tower = self.cls_tower(feature)
            bboxtower = self.bbox_tower(feature)

            if l<3:
                residual = self.residual_conv[l](backbone_features[l])
                bbox_reg.append(self.bbox_pred(bboxtower+residual))
                logits.append(self.cls_logits(cls_tower+residual))
                # print(logits[l].shape)

                centerness.append(self.centerness(cls_tower+residual))
            else:
                logits.append(self.cls_logits(cls_tower))
                # print(logits[l].shape)

                centerness.append(self.centerness(cls_tower))
            
                bbox_reg.append(self.bbox_pred(bboxtower))
            # bbox_reg.append(self.bbox_pred(self.bbox_tower(feature)))
        return logits, bbox_reg, centerness

 
class RFCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, backbone_outchannels, in_channels):
        super(RFCOSModule, self).__init__()

        head = RFCOSHead(cfg, backbone_outchannels, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images, backbone_features, fpn_features ,targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(backbone_features, fpn_features)
        locations = self.compute_locations(fpn_features)
 
        if self.training:
            return self._forward_train(
                locations, box_cls, 
                box_regression, 
                centerness, targets
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression, 
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        '''
        locations       不同分辨率 每个点中心xy坐标
        box_cls         logits
        box_regression  对logits使用torch.exp 
        '''
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        '''locations 长度为5'''
        locations = []
        # P3 - P7 
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        '''为不同分辨率卷积结果 都计算locations'''
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_rfcos(cfg, backbone_outchannels, in_channels):
    return RFCOSModule(cfg, backbone_outchannels, in_channels)
