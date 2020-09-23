import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from maskrcnn_benchmark.layers import Scale


class RFCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(RFCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        # self.cls_logits_list = []
        # self.cls_tower_list  = []
        # self.bbox_tower_list = []
        # self.bbox_pred_list  = []
        # self.centerness_list  = []

        for feature_layer in range(2):
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


            cls_tower_  = nn.Sequential(*cls_tower)
            bbox_tower_ = nn.Sequential(*bbox_tower)

            cls_logits = nn.Conv2d(
                in_channels, num_classes, kernel_size=3, stride=1,
                padding=1
            )
            bbox_pred = nn.Conv2d(
                in_channels, 5, kernel_size=3, stride=1,
                padding=1
            )
            centerness = nn.Conv2d(
                in_channels, 1, kernel_size=3, stride=1,
                padding=1
            )

            # initialization
            for modules in [cls_tower_, bbox_tower_,
                            cls_logits, bbox_pred,
                            centerness]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)

            self.add_module('cls_tower_{}'.format(feature_layer),  cls_tower_)
            self.add_module('bbox_tower_{}'.format(feature_layer), bbox_tower_)
            self.add_module('cls_logits_{}'.format(feature_layer), cls_logits)
            self.add_module('bbox_pred_{}'.format(feature_layer), bbox_pred)
            self.add_module('centerness_{}'.format(feature_layer), centerness)

            # initialize the bias for focal loss
            prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(cls_logits.bias, bias_value)

            # self.cls_logits_list.append(cls_logits)
            # self.cls_tower_list.append(cls_tower_)
            # self.bbox_tower_list.append(bbox_tower_)
            # self.bbox_pred_list.append(bbox_pred)
            # self.centerness_list.append(centerness)

        # self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        # for l, feature in enumerate(x):
        # 对FPN的结果进行卷积
        # p3
        cls_tower =       self.cls_tower_0 (x[0])
        logits.append(    self.cls_logits_0(cls_tower))
        centerness.append(self.centerness_0(cls_tower))
        bbox_reg.append(self.bbox_pred_0(self.bbox_tower_0(x[0])))
        # p4
        cls_tower =       self.cls_tower_0 (x[1])
        logits.append(    self.cls_logits_0(cls_tower))
        centerness.append(self.centerness_0(cls_tower))
        bbox_reg.append(self.bbox_pred_0(self.bbox_tower_0(x[1])))
        # p5
        cls_tower =       self.cls_tower_1 (x[2])
        logits.append(    self.cls_logits_1(cls_tower))
        centerness.append(self.centerness_1(cls_tower))
        bbox_reg.append(self.bbox_pred_1(self.bbox_tower_1(x[2])))
        # p6
        cls_tower =       self.cls_tower_1 (x[3])
        logits.append(    self.cls_logits_1(cls_tower))
        centerness.append(self.centerness_1(cls_tower))
        bbox_reg.append(self.bbox_pred_1(self.bbox_tower_1(x[3])))
        # p7
        cls_tower =       self.cls_tower_1 (x[4])
        logits.append(    self.cls_logits_1(cls_tower))
        centerness.append(self.centerness_1(cls_tower))
        bbox_reg.append(self.bbox_pred_1(self.bbox_tower_1(x[4])))

        # ori
        # for l, feature in enumerate(x):
        #     # 对FPN的结果进行卷积
        #     cls_tower = self.cls_tower(feature)
        #     logits.append(self.cls_logits(cls_tower))
        #     centerness.append(self.centerness(cls_tower))

        #     # 每一层赋予一个权重
        #     # bbox_reg.append(torch.exp(self.scales[l](
        #     #     self.bbox_pred(self.bbox_tower(feature))
        #     # )))
        #     bbox_reg.append(self.scales[l](
        #         self.bbox_pred(self.bbox_tower(feature))
        #     ))
        return logits, bbox_reg, centerness

 
class RFCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(RFCOSModule, self).__init__()

        head = RFCOSHead(cfg, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images, features, targets=None):
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
        box_cls, box_regression, centerness = self.head(features)
        locations = self.compute_locations(features)
 
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

def build_rfcos(cfg, in_channels):
    return RFCOSModule(cfg, in_channels)
