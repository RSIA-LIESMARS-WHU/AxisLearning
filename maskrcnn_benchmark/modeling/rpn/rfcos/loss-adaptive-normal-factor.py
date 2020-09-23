"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn

from ..utils import concat_box_prediction_layers
from maskrcnn_benchmark.layers import IOULoss
from maskrcnn_benchmark.layers import SigmoidFocalLoss, smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.rboxlist_ops import targets_for_locations
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
import time

INF = 100000000



class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        # self.cls_loss_func = nn.CrossEntropyLoss()

        self.cfg = cfg
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        # self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.num_pts = cfg.MODEL.FCOS.NUM_PTS

    # 2
    def prepare_targets(self, points, targets, normal_factor):
        # FoveaBox
        # strides=[8, 16, 32, 64, 128],
        # base_edge_list=[16, 32, 64, 128, 256],
        # scale_ranges=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
        if self.cfg.MODEL.FCOS.SELECT_FEATURE_METHOD == "fcos":
            # FCOS
            object_sizes_of_interest = [
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, INF],
            ]
        elif self.cfg.MODEL.FCOS.SELECT_FEATURE_METHOD == "foveabox":
            object_sizes_of_interest = [
                [-1, 64],
                [32, 128],
                [64, 256],
                [128, 512],
                [256, INF],
            ]
        elif self.cfg.MODEL.FCOS.SELECT_FEATURE_METHOD == "all":
            object_sizes_of_interest = [
                [-1, 64],
                [-1, 128],
                [-1, 256],
                [-1, 512],
                [-1, INF]]
        elif self.cfg.MODEL.FCOS.SELECT_FEATURE_METHOD == "neighbor":
                object_sizes_of_interest = [
                [-1, 64],
                [-1, 128],
                [64, 256],
                [128, 512],
                [256, INF]]

        # normal_factor = [16, 32, 64, 128, 256]
        # normal_factor = [64, 128, 256, 512, 800]
        print("normal_factor {} {} {} {} {}".format(normal_factor[0], normal_factor[1], normal_factor[2], normal_factor[3], normal_factor[4]) )



        expanded_object_sizes_of_interest = []
        expanded_normal_factor_list=[]
        # p3 - p7
        for l, points_per_level in enumerate(points):
            # 2
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            # 1 2 -> len(points_per_level) 2
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )
            # normal_factor_per_level = points_per_level.new_tensor(normal_factor[l])
            # print("points_per_level.shape", points_per_level.shape)
            # normal_factor_per_level = normal_factor[l].repeat(points_per_level.shape)


            # 1 2 -> len(points_per_level) 2
            # expanded_normal_factor_list.append(
            #     normal_factor_per_level.expand(len(points_per_level))
            # )
            expanded_normal_factor_list.append(
                normal_factor[l].expand(len(points_per_level))
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        expanded_normal_factor = torch.cat(expanded_normal_factor_list, dim=0)

        
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        # batch len(locations) 1   batch len(locations) 6 
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest, expanded_normal_factor
        )

        # 对每一张图片进行处理
        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            reg_targets_level_first.append(
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
            )

        return labels_level_first, reg_targets_level_first

    # def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
    #     labels = []
    #     reg_targets = []
    #     xs, ys = locations[:, 0], locations[:, 1]

    #     for im_i in range(len(targets)):
    #         # 第i张图片
    #         targets_per_im = targets[im_i]
    #         assert targets_per_im.mode == "xyxy"
    #         bboxes = targets_per_im.bbox
    #         labels_per_im = targets_per_im.get_field("labels")
    #         area = targets_per_im.area()

    #         l = xs[:, None] - bboxes[:, 0][None]
    #         t = ys[:, None] - bboxes[:, 1][None]
    #         r = bboxes[:, 2][None] - xs[:, None]
    #         b = bboxes[:, 3][None] - ys[:, None]
    #         reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

    #         is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

    #         max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]

    #         # limit the regression range for each location 上下左右都要在感兴趣范围之内
    #         is_cared_in_the_level = \
    #             (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
    #             (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

                

    #         locations_to_gt_area = area[None].repeat(len(locations), 1)
    #         locations_to_gt_area[is_in_boxes == 0] = INF
    #         locations_to_gt_area[is_cared_in_the_level == 0] = INF

    #         # if there are still more than one objects for a location,
    #         # we choose the one with minimal area
    #         locations_to_min_aera, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

    #         reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
    #         labels_per_im = labels_per_im[locations_to_gt_inds]
    #         labels_per_im[locations_to_min_aera == INF] = 0

    #         labels.append(labels_per_im)
    #         reg_targets.append(reg_targets_per_im)

    #     return labels, reg_targets 

    # 3
    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest, normal_factor):
        labels = []
        reg_targets = []
        # xs, ys = locations[:, 0], locations[:, 1]
        
        for im_i in range(len(targets)):
            # 第i张图片
            targets_per_im = targets[im_i]
            # assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")#.cpu()
            # print(labels_per_im)

            # area = targets_per_im.area()

            # len(locations), len(bboxes)
            # l = xs[:, None] - bboxes[:, 0][None]
            # t = ys[:, None] - bboxes[:, 1][None]
            # r = bboxes[:, 2][None] - xs[:, None]
            # b = bboxes[:, 3][None] - ys[:, None]
            
            # len(locations), len(bboxes) 4
            # reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            # len(locations) 8(target_idx target area detax1 detay1 detax2 detay2 min(w h) weight)
            # 8 k
            # torch.cuda.empty_cache()
            # print(locations.shape)
            # reg_targets_per_im = None
            reg_targets_per_im = targets_for_locations(bboxes, locations)#.cpu()
            # print(reg_targets_per_im)
            # reg_targets_per_im = reg_targets_per_im

            torch.cuda.empty_cache()
            # time.sleep(0.5)
            # exit(0)

            # # len(locations), len(bboxes)
            # max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            
            max_reg_targets_per_im = torch.abs(reg_targets_per_im[:,2:6]).max(dim=1)[0]
            # distance
            
            dist_1 = torch.sqrt(torch.pow(reg_targets_per_im[:,2],2) + torch.pow(reg_targets_per_im[:,3],2))
            dist_2 = torch.sqrt(torch.pow(reg_targets_per_im[:,4],2) + torch.pow(reg_targets_per_im[:,5],2))
            target_h = reg_targets_per_im[:,6] 
            max_reg_targets_per_im = torch.stack([dist_1, dist_2, target_h], dim=1).max(dim=1)[0]

            # limit the regression range for each location 上下左右都要在感兴趣范围之内

            # len(locations)  len(locations), 1
            object_sizes_of_interest= object_sizes_of_interest#.cpu()
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, 0]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, 1])
            
            # print("labels_per_im", len(labels_per_im), len(bboxes), torch.min(reg_targets_per_im[:, 0].long()), torch.max(reg_targets_per_im[:, 0].long()), reg_targets_per_im[:, 0].sum())
            labels_per_im = labels_per_im[reg_targets_per_im[:, 0].long()]

            # 落在目标框外面label为0
            labels_per_im[reg_targets_per_im[:, 1] < 0.5 ] = 0#bg
            # 感受野不够
            labels_per_im[is_cared_in_the_level == 0] = 0#bg
            
            # labels_per_im[reg_targets_per_im[:,-1] < 0.3] = -1#bg
            # labels_per_im[reg_targets_per_im[:,-1] < 0.2] = -1#bg

            # detax1 detay1  detax2 detay2 h
            ones = torch.ones_like(reg_targets_per_im[:,2:7])
            one_minusone = torch.where(reg_targets_per_im[:,2:7]>=0, ones, -ones)#.cpu()
            
            reg_targets_per_im_col_2_7 = reg_targets_per_im[:,2:7]/normal_factor[:,None]#[1][:,None]
            # reg_targets_per_im_col_2_7 = one_minusone*torch.pow(torch.abs(reg_targets_per_im[:,2:7])/normal_factor[:,None], 1/3)#.cpu()#.cpu()
            reg_targets_per_im_col_2_8 = torch.cat((reg_targets_per_im_col_2_7, reg_targets_per_im[:,-1:]), dim=1)
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im_col_2_8)

        return labels, reg_targets 

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    # 1
    def __call__(self, locations, box_cls, box_regression, centerness, targets, normal_factor):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        # 0 fpn 第一层
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)#//self.num_pts

        # level first
        labels, reg_targets = self.prepare_targets(locations, targets, normal_factor)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        # for level
        for l in range(len(labels)):
            # batch*num_pos num_classes
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 5))
            # layer_h, layer_w = box_cls[l].size(2), box_cls[l].size(3)
            # box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(N, layer_h, layer_w, self.num_pts, num_classes).permute(0, 3, 1, 2, 4).reshape(-1,num_classes))
            # box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(N, layer_h, layer_w, self.num_pts, 5).permute(0, 3, 1, 2, 4).reshape(-1,5))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 6))
            centerness_flatten.append(centerness[l].reshape(-1))
        # level batch*num_pos num_classes
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        # valid_inds = torch.nonzero(labels_flatten > -1).squeeze(1)
        


        # wrong
        # cls_weight=torch.where(centerness_flatten==0, torch.ones_like(centerness_flatten), centerness_flatten).unsqueeze(-1)
        # cls_loss = self.cls_loss_func(
        #     box_cls_flatten,#.cpu()
        #     labels_flatten.int(),#,#.cpu()
        #     weight = cls_weight
        # ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero

        # true
        all_centerness_targets = reg_targets_flatten[:, -1]
        # torch.sqrt(
        
        # cls_weight=torch.where(all_centerness_targets==0, torch.ones_like(all_centerness_targets), 2*all_centerness_targets).unsqueeze(-1)
        
        '''
        加权是为了减弱边缘特征点的影像
        实验１发现扩大centerness范围可以增加飞机和船的map
        实验2发现减小背景样本权重达到同样的效果
        '''
        cls_weight=torch.where(all_centerness_targets==0, torch.full_like(all_centerness_targets, 0.8), all_centerness_targets).unsqueeze(-1)

        # print(cls_weight)
        # print((all_centerness_targets==0).sum(), (cls_weight==0).sum())
        
        # 并不是所有点都是正样本
        cls_loss = self.cls_loss_func(
            box_cls_flatten,#.cpu()
            labels_flatten.int(),#.cpu()
            weight = cls_weight
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero

        box_regression_flatten  = box_regression_flatten[pos_inds]
        reg_targets_flatten_pos = reg_targets_flatten[pos_inds]
        # centerness_flatten = centerness_flatten[pos_inds]

        if pos_inds.numel() > 0:
            # centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            # centerness_targets = reg_targets_flatten_pos[:, -1]

            #这里是不是要和cls loss 保持一致 
            reg_loss = smooth_l1_loss(
                box_regression_flatten,#.cpu()
                reg_targets_flatten_pos[:, :-1],#.cpu()
                weight =reg_targets_flatten_pos[:, -1].unsqueeze(-1)#cls_weight #
            )

            # 一定要回归center ness
            # reg_loss = torch.tensor(0)
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,#.cpu()
                reg_targets_flatten[:, -1]#.cpu()
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()
            # .cuda()
        return cls_loss, reg_loss, centerness_loss#*0


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
