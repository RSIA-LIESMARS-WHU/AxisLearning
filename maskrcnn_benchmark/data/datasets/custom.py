import os

import torch
import torch.utils.data
from PIL import Image
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from maskrcnn_benchmark.structures.bounding_box import BoxList


class CustomDataset(torch.utils.data.Dataset):

    CLASSES =  (
    'boat1','boat2','boat3','boat4','boat5','boat6','boat7','boat8',
    'building3',
    'car1','car10','car11','car12','car13','car14','car15','car16','car17','car18','car19','car2','car20','car21','car22','car23','car24','car3','car4','car5','car6','car8','car9',
    'drone1','drone2','drone3','drone4',
    'group2','group3',
    'horseride1',
    'paraglider1',
    'person1','person10','person11','person12','person13','person14','person15','person16','person17','person18','person19','person2','person20','person21','person22','person23',
    'person24','person25','person26','person27','person28','person29','person3','person4','person5','person6','person7','person8','person9',
    'riding1','riding10','riding11','riding12','riding13','riding14','riding15','riding16','riding17','riding2','riding3','riding4','riding5','riding6','riding7','riding8','riding9',
    'truck1','truck2',
    'wakeboard1','wakeboard2','wakeboard3','wakeboard4',
    'whale1')

    def __init__(self, data_file, transforms=None):
        # self.root = data_dir
        # self.image_set = split#train test
        # self.keep_difficult = use_difficult
        self.transforms = transforms

        # self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        # self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        # self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        # with open(self._imgsetpath % self.image_set) as f:
        #     self.ids = f.readlines()
        # self.ids = [x.strip("\n") for x in self.ids]
        # self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        # cls = PascalVOCDataset.CLASSES
        # self.class_to_ind = dict(zip(cls, range(len(cls))))

        self.data_list = []
        #self.preprocess = tr.Compose([tr.Scale((224, 224)), tr.ToTensor()])
        # image_root = os.path.join(root_dir, 'image')
        
        with open(data_file) as f:
            line=f.readline().strip('\n')
            while (line):
            #   line = line.strip()#移除字符串头尾指定的字符（默认为空格或换行符）
              line = line.split('  ')
              line[1:] = [int(i) for i in line[1].split(',')]
              self.data_list.append(line)
              line=f.readline().strip('\n')

    def __getitem__(self, index):
        data = self.data_list[index]
        img = Image.open(data[0]).convert("RGB")

        target = BoxList(data[1:5], (640, 360), mode="xyxy")
        # print('label', CustomDataset.CLASSES[data[5]])
        # target.add_field("labels", CustomDataset.CLASSES[data[5]])

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.data_list)
    

    # no need 
    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1
        
        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text, 
                bb.find("ymin").text, 
                bb.find("xmax").text, 
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.data_list[index]
        # anno = ET.parse(self._annopath % img_id).getroot()
        # size = anno.find("size")
        # im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": 360, "width": 640}

    def map_class_id_to_class_name(self, class_id):
        return CustomDataset.CLASSES[class_id]
