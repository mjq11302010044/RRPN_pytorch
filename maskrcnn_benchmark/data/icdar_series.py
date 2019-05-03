import os

import torch
import torch.utils.data
from PIL import Image
import sys
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList
import cv2

class ICDAR2013Dataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "text"
    )

    def __init__(self, data_dir, anno_dir, split, use_difficult=False, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms
        self.anno_dir = anno_dir
        # self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        # self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        # self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        #with open(self._imgsetpath % self.image_set) as f:
        #    self.ids = f.readlines()
        #self.ids = [x.strip("\n") for x in self.ids]

        image_list = os.listdir(self.root)
        self.ids = [image[:-4] for image in image_list]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = ICDAR2013Dataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def __getitem__(self, index):
        img_id = self.ids[index]

        im_path = os.path.join(self.root, img_id + '.jpg')
        img = Image.open(im_path).convert("RGB")
        im = cv2.imread(im_path)
        anno = self.get_groundtruth(index)
        anno["im_info"] = [im.shape[0], im.shape[1]]
        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        # anno = ET.parse(self._annopath % img_id).getroot()

        gt_path = os.path.join(self.anno_dir, 'gt_' + img_id + '.txt')
        anno = self._preprocess_annotation(gt_path)

        return anno

    def _preprocess_annotation(self, gt_path):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        # TO_REMOVE = 1

        gt_list = open(gt_path, 'r', encoding='utf-8').readlines()
        for gt_ele in gt_list:
            gt_ele = gt_ele.replace('\n', '').replace('\ufeff', '')
            gt = gt_ele.split(',')

            if len(gt) > 1:
                gt_ind = np.array(gt[:8], dtype=np.float32)
                gt_ind = np.array(gt_ind, dtype=np.int32)
                words = gt[8]
                gt_ind = gt_ind.reshape(4, 2)
                xs = gt_ind[:, 0].reshape(-1)
                ys = gt_ind[:, 1].reshape(-1)
                xmin = np.min(xs)
                xmax = np.max(xs)
                ymin = np.min(ys)
                ymax = np.max(ys)

                boxes.append([xmin, ymin, xmax, ymax])
                gt_classes.append(self.class_to_ind['text'])
                difficult_boxes.append(0)

        # size = target.find("size")
        # im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": None,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return ICDAR2013Dataset.CLASSES[class_id]
