# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import cv2
import numpy as np
import random
from PIL import Image

from maskrcnn_benchmark.structures.rotation_box import RBoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o < 0 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class DOTARotateDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(DOTARotateDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        print('xxx:', len(self.ids))

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids
            print('yyy: ', len(self.ids))

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms
    
    def conver4xyTOxywha(self, boxes):
        result = []
        for rect in boxes:
            box = np.int0(rect)
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            result.append([x, y, w, h, theta])
        return result

    def __getitem__(self, idx):
        img, anno = super(DOTARotateDataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        
        # scale_seed = random.uniform(0.4,1)
        # img_tem = img1.resize((int(img1.size[0]*scale_seed), int(img1.size[1]*scale_seed)))
        # img = Image.new('RGB', (img1.size[0], img1.size[1]), (255, 255, 255))  
        # img.paste(img_tem, (0, 0))  
        boxes = [obj["bbox"] for obj in anno]
        
        # boxes = self.conver4xyTOxywha(boxes1, scale_seed)
        # if(len(boxes)<=0):
            # print('scale: ', scale_seed)
            # img = img1
        # new
        # boxes = self.conver4xyTOxywha(boxes)
        # print(len(boxes))
        boxes = torch.as_tensor(boxes).reshape(-1, 8)  # guard against no boxes    
        
        target = RBoxList(boxes, img.size, mode="4xy")
        
        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        # masks = [obj["segmentation"] for obj in anno]
        # masks = SegmentationMask(masks, img.size, mode='poly')
        # target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        if(len(target)>=800):
            target = target[0:800]
            # print(len(target))
        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
    
    

