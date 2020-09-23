import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import cat
# from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.rotation_box import RBoxList
# from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
# from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
# from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes
from maskrcnn_benchmark.structures.rboxlist_ops import cat_boxlist

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class RFCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(RFCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes

        self.id =0
        
    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes, normal_factor, layer):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape
        # C=C//5
        # put in the same format as locations
        box_cls = box_cls.permute(0, 2, 3, 1)#.view(N, C, H, W)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()#softmax(dim=2)#

        # N -1
        max_cls, max_cls_ind = box_cls.max(dim=2)


        # box_cls = torch.index_select(box_cls, 2, max_cls_ind)
        # print(box_cls)
        # box_cls = box_cls.reshape(N, -1, C).softmax(dim=2)

        # detax1 detay1 detax2 detay2 h
        box_regression = box_regression.permute(0, 2, 3, 1)#.view(N, -1, H, W)
        # box_regression = box_regression.reshape(N, -1, 10)[...,:5]#.chunk(2,2)
        box_regression = box_regression.reshape(N, -1, 5)#.chunk(2,2)
        centerness = centerness.permute(0, 2, 3, 1)#.view(N, 1, H, W)
        centerness = centerness.reshape(N, -1).sigmoid()

        # N h*w C 
        candidate_inds = max_cls* centerness  > 0.2#self.pre_nms_thresh

        # candidate_inds = (box_cls>0.1)  * (centerness[:, :, None]>0.1)> self.pre_nms_thresh
        # N h*w*C -> N 1
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        # N h*w*C (1000)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        # N h*w C 
        # 现在的置信度可以直接算出来
        # box_cls = ((box_cls>0.1)  * (centerness[:, :, None]>0.1)).float()
        # box_cls = box_cls * centerness[:, :, None]
        # print(box_cls.max(dim=1)[0])
        # print(centerness)
        
        results = []
        # 对每一张图片进行处理
        for i in range(N):
            # h*w C 
            per_box_cls = max_cls[i]

            # h*w C  bool
            per_candidate_inds = candidate_inds[i]

            # 1D cls
            per_box_cls = per_box_cls[per_candidate_inds]


            per_candidate_nonzeros = per_candidate_inds.nonzero()
            # loc ind
            # per_box_loc = per_candidate_nonzeros[:, 0]
            # # 类别信息
            # per_class = per_candidate_nonzeros[:, 1] + 1
            # loc ind
            per_box_loc = per_candidate_nonzeros
            # 类别信息
            per_class = max_cls_ind[i][per_candidate_inds] + 1

            per_box_regression = torch.pow(box_regression[i], 3)*normal_factor
            per_box_regression = per_box_regression[per_box_loc].squeeze(1)
            per_locations = locations[per_box_loc].squeeze(1)
            # print(per_box_regression.shape)

            per_pre_nms_top_n = pre_nms_top_n[i]
            # 结果大1000
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
            
            h, w = image_sizes[i]
            # N 5
            detections = torch.stack([
                per_locations[:, 0],
                per_locations[:, 1],
                per_locations[:, 0] - per_box_regression[:, 0],#x1
                per_locations[:, 1] - per_box_regression[:, 1],#y1
                per_locations[:, 0] - per_box_regression[:, 2],#x2
                per_locations[:, 1] - per_box_regression[:, 3],#y2
                per_box_regression[:, 4],#h
            ], dim=1)

            # center_xy = (detections[:,[0,1]]+detections[:,[2,3]])/2
            # wh = 
            
            boxlist = RBoxList(detections, (int(w), int(h)), mode="xywha")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist.add_field("levels", torch.full_like(per_class, layer))
            # boxlist.add_field("filter_score", filter_score)
            # boxlist = boxlist.clip_to_image(remove_empty=False)
            # boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results



    # 核心
    # def forward_for_single_feature_map(
    #         self, locations, box_cls,
    #         box_regression, centerness,
    #         image_sizes, normal_factor, layer):
    #     """
    #     Arguments:
    #         anchors: list[BoxList]
    #         box_cls: tensor of size N, A * C, H, W
    #         box_regression: tensor of size N, A * 4, H, W
    #     """
    #     N, C, H, W = box_cls.shape
    #     # C=C//5
    #     # put in the same format as locations
        

    #     box_cls = box_cls.permute(0, 2, 3, 1)#.view(N, C, H, W)
    #     # ori_box_cls = box_cls.detach()
        
    #     # print(box_cls)
    #     # box_cls = box_cls.reshape(N, -1, C).softmax(dim=2)

    #     # detax1 detay1 detax2 detay2 h
    #     box_regression = box_regression.permute(0, 2, 3, 1)#.view(N, -1, H, W)
    #     # box_regression = box_regression.reshape(N, -1, 10)[...,:5]#.chunk(2,2)
    #     box_regression = box_regression.reshape(N, -1, 5)#.chunk(2,2)
    #     centerness = centerness.permute(0, 2, 3, 1)#.view(N, 1, H, W) NCHW -> NHWC

    #     # heatmap
    #     # np.random.seed(0)
    #     max_cls, max_cls_ind = box_cls[0].detach().max(dim=2)
    #     uniform_data = max_cls.cpu().sigmoid()#centerness[0,:,:,0].detach().cpu().sigmoid()#np.random.rand(10, 12)
    #     # uniform_data, _ = ori_box_cls[0].detach().cpu().sigmoid().max(dim=2)#np.random.rand(10, 12)

    #     uniform_data = np.array(uniform_data)
    #     self.savePath  = "/media/liesmars/b71625db-4194-470b-a8ab-2d4cf46f4cdd/Object_detection/FCOS_pytorch/RFCOS/training_dir/test/heatmap"
    
    #     # if layer == 0:
    #     #     plt.cla()
    #     #     plt.clf()
    #     #     sns.set()
    #     #     ax = sns.heatmap(uniform_data, cmap='rainbow')
    #     #     plt.savefig(os.path.join(self.savePath, "{}_{}_{}_cls".format(layer, self.id, uniform_data.shape[0])))
    #     #     self.id +=1
        
    #     box_cls = box_cls.reshape(N, -1, C).sigmoid()#softmax(dim=2)#
    #     centerness = centerness.reshape(N, -1).sigmoid()
    #     # N h*w C 
    #     #  是根据检测结果计算得到的* centerness[:, :, None]
    #     candidate_inds = box_cls >self.pre_nms_thresh
    #     # candidate_inds = (box_cls>0.1)  * (centerness[:, :, None]>0.1)> self.pre_nms_thresh
    #     # N h*w*C -> N 1
    #     pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
    #     # N h*w*C (1000)
    #     pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

    #     # multiply the classification scores with centerness scores
    #     # N h*w C 
    #     # 现在的置信度可以直接算出来
    #     # box_cls = ((box_cls>0.1)  * (centerness[:, :, None]>0.1)).float()
    #     # box_cls = box_cls * centerness[:, :, None]
    #     # print(box_cls.max(dim=1)[0])
    #     # print(centerness)
        
    #     results = []
    #     # 对每一张图片进行处理
    #     for i in range(N):
    #         # h*w C 
    #         per_box_cls = box_cls[i]
    #         # h*w C  bool
    #         per_candidate_inds = candidate_inds[i]
    #         # 1D cls
    #         per_box_cls = per_box_cls[per_candidate_inds]

    #         per_candidate_nonzeros = per_candidate_inds.nonzero()
    #         # loc ind
    #         per_box_loc = per_candidate_nonzeros[:, 0]
    #         # 类别信息
    #         per_class = per_candidate_nonzeros[:, 1] + 1

    #         per_box_regression = torch.pow(box_regression[i], 3)*normal_factor
    #         per_box_regression = per_box_regression[per_box_loc]
    #         per_locations = locations[per_box_loc]

    #         per_pre_nms_top_n = pre_nms_top_n[i]
    #         # 结果大1000
    #         if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
    #             per_box_cls, top_k_indices = \
    #                 per_box_cls.topk(per_pre_nms_top_n, sorted=False)
    #             per_class = per_class[top_k_indices]
    #             per_box_regression = per_box_regression[top_k_indices]
    #             per_locations = per_locations[top_k_indices]
            
    #         h, w = image_sizes[i]
    #         # N 5
    #         detections = torch.stack([
    #             per_locations[:, 0],
    #             per_locations[:, 1],
    #             per_locations[:, 0] - per_box_regression[:, 0],#x1
    #             per_locations[:, 1] - per_box_regression[:, 1],#y1
    #             per_locations[:, 0] - per_box_regression[:, 2],#x2
    #             per_locations[:, 1] - per_box_regression[:, 3],#y2
    #             per_box_regression[:, 4],#h
    #         ], dim=1)

    #         # center_xy = (detections[:,[0,1]]+detections[:,[2,3]])/2
    #         # wh = 
            
    #         boxlist = RBoxList(detections, (int(w), int(h)), mode="xywha")
    #         boxlist.add_field("labels", per_class)
    #         boxlist.add_field("scores", per_box_cls)
    #         boxlist.add_field("levels", torch.full_like(per_class, layer))
    #         # boxlist.add_field("filter_score", filter_score)
    #         # boxlist = boxlist.clip_to_image(remove_empty=False)
    #         # boxlist = remove_small_boxes(boxlist, self.min_size)
    #         results.append(boxlist)

    #     return results

    def forward(self, locations, box_cls, box_regression, centerness, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        # rfcos
        normal_factor = [16, 32, 64, 128, 256]
        # normal_factor = [32,  64, 128, 256, 512]
        # fovea
        # normal_factor = [32,  64, 128, 256, 512]
        # normal_factor = [16, 48, 96, 192, 384]
        # normal_factor = [64, 128, 256, 512, 800]
        # 5个特征尺度
        for layer, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, image_sizes, normal_factor[layer], layer
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        # print("boxlists", boxlists)
        # boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            # skip the background
            for j in range(1, self.num_classes):
                inds = (labels == j).nonzero().view(-1)

                scores_j = scores[inds]
                boxes_j = boxes[inds, :].view(-1, 4)
                boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
                boxlist_for_class.add_field("scores", scores_j)
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, self.nms_thresh,
                    score_field="scores"
                )
                num_labels = len(boxlist_for_class)
                boxlist_for_class.add_field(
                    "labels", torch.full((num_labels,), j,
                                         dtype=torch.int64,
                                         device=scores.device)
                )
                result.append(boxlist_for_class)

            result = cat_boxlist(result)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_fcos_postprocessor(config):
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.FCOS.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG

    box_selector = RFCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.FCOS.NUM_CLASSES
    )

    return box_selector
