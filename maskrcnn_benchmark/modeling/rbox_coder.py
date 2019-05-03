# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch


class RBoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """

        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2]# - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3]# - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0]# + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1]# + 0.5 * ex_heights
        ex_angle = proposals[:, 4]

        gt_widths = reference_boxes[:, 2]# - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3]# - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0]# + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1]# + 0.5 * gt_heights
        gt_angle = reference_boxes[:, 4]

        wx, wy, ww, wh, wa = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        targets_da = wa * (gt_angle - ex_angle)
        #targets_da[np.where((gt_angle <= -30) & (ex_angle >= 120))] += 180
        #targets_da[np.where((gt_angle >= 120) & (ex_angle <= -30))] -= 180

        gtle30 = gt_angle.le(-30)
        exge120 = ex_angle.ge(120)
        gtge120 = gt_angle.ge(120)
        exle30 = ex_angle.le(-30)

        incre180 = gtle30 * exge120 * 180
        decre180 = gtge120 * exle30 * (-180)

        targets_da = targets_da + incre180.float()
        targets_da = targets_da + decre180.float()

        targets_da = 3.14159265358979323846264338327950288 / 180 * targets_da

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, targets_da), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2]# - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3]# - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0]# + 0.5 * widths
        ctr_y = boxes[:, 1]# + 0.5 * heights
        angle = boxes[:, 4]

        wx, wy, ww, wh, wa = self.weights
        dx = rel_codes[:, 0::5] / wx
        dy = rel_codes[:, 1::5] / wy
        dw = rel_codes[:, 2::5] / ww
        dh = rel_codes[:, 3::5] / wh
        da = rel_codes[:, 4::5] / wa

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        da = da * 1.0 / 3.141592653 * 180  # arc to angle
        pred_angle = da + angle[:, None]

        # print('pred_angle:', pred_angle.size())
        pred_boxes = torch.zeros_like(rel_codes)
        # ctr_x1
        pred_boxes[:, 0::5] = pred_ctr_x
        # ctr_y1
        pred_boxes[:, 1::5] = pred_ctr_y
        # width
        pred_boxes[:, 2::5] = pred_w
        # height
        pred_boxes[:, 3::5] = pred_h
        # angle
        pred_boxes[:, 4::5] = pred_angle

        return pred_boxes
