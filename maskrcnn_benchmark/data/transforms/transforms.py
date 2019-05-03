# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import cv2
from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import RBoxList


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
        self.scale_jittering = False
        if isinstance(min_size, tuple):
            self.min_size_random_group = min_size
            self.scale_jittering = True
            self.group_size = len(min_size)

    # modified from torchvision to add support for max size
    def get_size(self, image_size):

        w, h = image_size
        size = self.min_size
        if self.scale_jittering:
            size = self.min_size_random_group[np.random.randint(self.group_size)]

        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        # print('size:', (oh, ow))
        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(target.size) # sth wrong with image.size
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomRotation(object):
    def __init__(self, prob, r_range=(360, 0), fixed_angle=-1, gt_margin=1.4):
        self.prob = prob
        self.fixed_angle = fixed_angle
        self.gt_margin = gt_margin
        self.rotate_range = r_range[0]
        self.shift = r_range[1]

    def rotate_boxes(self, target, angle):
        # def rotate_gt_bbox(iminfo, gt_boxes, gt_classes, angle):
        gt_boxes = target.bbox
        if isinstance(target.bbox, torch.Tensor):
            gt_boxes = target.bbox.data.cpu().numpy()

        gt_labels = target.get_field("labels")

        rotated_gt_boxes = np.empty((len(gt_boxes), 5), dtype=np.float32)

        iminfo = target.size

        im_height = iminfo[1]
        im_width = iminfo[0]
        origin_gt_boxes = gt_boxes

        # anti-clockwise to clockwise arc
        cos_cita = np.cos(np.pi / 180 * angle)
        sin_cita = np.sin(np.pi / 180 * angle)

        # clockwise matrix
        rotation_matrix = np.array([[cos_cita, sin_cita], [-sin_cita, cos_cita]])

        pts_ctr = origin_gt_boxes[:, 0:2]

        pts_ctr = pts_ctr - np.tile((im_width / 2, im_height / 2), (gt_boxes.shape[0], 1))
        # print('pts_ctr:', pts_ctr.shape)
        pts_ctr = np.array(np.dot(pts_ctr, rotation_matrix), dtype=np.int16)
        # print('pts_ctr:', pts_ctr.shape)
        pts_ctr = np.squeeze(pts_ctr, axis=-1) + np.tile((im_width / 2, im_height / 2), (gt_boxes.shape[0], 1))

        # print('pts_ctr:', pts_ctr, np.tile((im_width / 2, im_height / 2), (gt_boxes.shape[0], 1)).shape)
        origin_gt_boxes[:, 0:2] = pts_ctr
        # print origin_gt_boxes[:, 0:2]

        len_of_gt = len(origin_gt_boxes)

        # rectificate the angle in the range of [-45, 45]

        for idx in range(len_of_gt):
            ori_angle = origin_gt_boxes[idx, 4]
            height = origin_gt_boxes[idx, 3]
            width = origin_gt_boxes[idx, 2]

            # step 1: normalize gt (-45,135)
            if width < height:
                ori_angle += 90
                width, height = height, width

            # step 2: rotate (-45,495)
            rotated_angle = ori_angle + angle

            # step 3: normalize rotated_angle       (-45,135)
            while rotated_angle > 135:
                rotated_angle = rotated_angle - 180

            rotated_gt_boxes[idx, 0] = origin_gt_boxes[idx, 0]
            rotated_gt_boxes[idx, 1] = origin_gt_boxes[idx, 1]
            rotated_gt_boxes[idx, 3] = height * self.gt_margin
            rotated_gt_boxes[idx, 2] = width * self.gt_margin
            rotated_gt_boxes[idx, 4] = rotated_angle

        x_inbound = np.logical_and(rotated_gt_boxes[:, 0] >= 0, rotated_gt_boxes[:, 0] < im_width)
        y_inbound = np.logical_and(rotated_gt_boxes[:, 1] >= 0, rotated_gt_boxes[:, 1] < im_height)

        inbound = np.logical_and(x_inbound, y_inbound)

        inbound_th = torch.tensor(np.where(inbound)).long().view(-1)

        rotated_gt_boxes_th = torch.tensor(rotated_gt_boxes[inbound]).to(target.bbox.device)
        # print('gt_labels before:', gt_labels.size(), inbound_th.size())
        gt_labels = gt_labels[inbound_th]
        # print('gt_labels after:', gt_labels.size())
        difficulty = target.get_field("difficult")
        difficulty = difficulty[inbound_th]

        target_cpy = RBoxList(rotated_gt_boxes_th, iminfo, mode='xywha')
        target_cpy.add_field('difficult', difficulty)
        target_cpy.add_field('labels', gt_labels)
        # print('has word:', target.has_field("words"), target.get_field("words"))
        if target.has_field("words"):
            words = target.get_field("words")[inbound_th]
            target_cpy.add_field('words', words)
        if target.has_field("word_length"):
            word_length = target.get_field("word_length")[inbound_th]
            target_cpy.add_field('word_length', word_length)
        if target.has_field("masks"):
            seg_masks = target.get_field("masks")
            # print('seg_masks:', seg_masks)
            target_cpy.add_field('masks', seg_masks.rotate(torch.from_numpy(angle.astype(np.float32)), torch.tensor([im_width / 2, im_height / 2]))[inbound_th])
        # print('rotated_gt_boxes_th:', origin_gt_boxes[0], target_cpy.bbox[0])
        # print('rotated_gt_boxes_th:', target.bbox.size(), gt_boxes.shape)

        if target_cpy.bbox.size()[0] <= 0:
            return None

        return target_cpy

    def rotate_img(self, image, angle):
        # convert to cv2 image
        image = np.array(image)
        (h, w) = image.shape[:2]
        scale = 1.0
        # set the rotation center
        center = (w / 2, h / 2)
        # anti-clockwise angle in the function
        M = cv2.getRotationMatrix2D(center, angle, scale)
        image = cv2.warpAffine(image, M, (w, h))
        # back to PIL image
        image = Image.fromarray(image)
        return image

    def __call__(self, image, target):
        angle = np.array([np.max([0, self.fixed_angle])])
        if np.random.rand() <= self.prob:
            angle = np.array(np.random.rand(1) * self.rotate_range - self.shift, dtype=np.int16)
        # print('angle:', angle)
        # angle = np.array([20])
        return self.rotate_img(image, angle), self.rotate_boxes(target, angle)


class MixUp:
    def __init__(self, mix_ratio):
        assert mix_ratio <= 1, 'mix_ratio needs to be less than 1' + str(mix_ratio)
        self.mix_ratio = mix_ratio

    def __call__(self, image_src, image_mix, target):
        mix_ratio = np.random.rand() * self.mix_ratio
        image_mix = image_mix.resize(image_src.size)
        # print('mixup:', image_src.size, image_mix.size)
        image_mixed = np.array(image_src) * (1 - mix_ratio) + np.array(image_mix) * mix_ratio
        return Image.fromarray(np.array(image_mixed, np.uint8)), target