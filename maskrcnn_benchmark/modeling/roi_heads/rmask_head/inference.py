# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import cv2
import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.bounding_box import RBoxList
from maskrcnn_benchmark.structures.segmentation_for_rbox import rotate_pts

# TODO check if want to return a single BoxList or a composite
# object
class MaskPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, masker=None):
        super(MaskPostProcessor, self).__init__()
        self.masker = masker

    def forward(self, x, boxes):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        mask_prob = x.sigmoid()

        # select masks coresponding to the predicted classes
        num_masks = x.shape[0]
        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_masks, device=labels.device)
        mask_prob = mask_prob[index, labels][:, None]

        boxes_per_image = [len(box) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)

        if self.masker:
            mask_prob = self.masker(mask_prob, boxes)

        results = []
        for prob, box in zip(mask_prob, boxes):
            bbox = RBoxList(box.bbox, box.size, mode="xywha")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("mask", prob)
            results.append(bbox)

        return results


class MaskPostProcessorCOCOFormat(MaskPostProcessor):
    """
    From the results of the CNN, post process the results
    so that the masks are pasted in the image, and
    additionally convert the results to COCO format.
    """

    def forward(self, x, boxes):
        import pycocotools.mask as mask_util
        import numpy as np

        results = super(MaskPostProcessorCOCOFormat, self).forward(x, boxes)
        for result in results:
            masks = result.get_field("mask").cpu()
            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")
            result.add_field("mask", rles)
        return results


# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily gor paste_mask_in_image
def expand_boxes(boxes, scale):
    w_half = boxes[:, 2] * .5
    h_half = boxes[:, 3] * .5
    x_c = boxes[:, 0]
    y_c = boxes[:, 1]

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c
    boxes_exp[:, 2] = w_half * 2
    boxes_exp[:, 1] = y_c
    boxes_exp[:, 3] = h_half * 2
    boxes_exp[:, 4] = boxes[:, 4]
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale)[0]
    box = box.to(dtype=torch.int32)

    TO_REMOVE = 1
    w = int(box[2]) # int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3]) # int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    angle = box[-1]

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = mask.to(torch.float32)
    mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    if thresh >= 0:
        mask = mask > thresh
    else:
        # for visualization and debugging, we also
        # allow it to return an unmodified mask
        mask = (mask * 255).to(torch.uint8)

    mask_np = mask.data.cpu().numpy()
    # print('box:', box)
    # rotate pts to fit the angles and fill contours
    im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    contours = cv2.findContours((mask_np*1).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # pts_string = torch.from_numpy(contours[1][0].reshape(-1))
    # print('contours:', len(contours[1]), pts_string[0::2])
    # print('mask_center:', box)
    # shorten = im_h if im_h < im_w else im_w
    # scale = shorten / 896
    res_cons = []

    # for pts in  contours[1]:
    #    rt_pts = rotate_pts(torch.from_numpy(pts.reshape(-1)).float(),
    #    angle.float(), (w // 2, h // 2)).reshape(-1, 1, 2)
    #    rt_pts += torch.tensor([box[0], box[1]]).float() - torch.tensor([w // 2, h // 2]).float()
    #    # print('rt_ctr:', rt_pts.data.cpu().numpy().mean(0))
    #    res_cons.append(rt_pts)

    res_cons = [(rotate_pts(
        torch.from_numpy(
            pts.reshape(-1)).float(), angle.float(), (w//2, h//2)).reshape(-1, 1, 2) +
                 torch.tensor([box[0], box[1]]).float() -
                 torch.tensor([w // 2, h // 2]).float())
                for pts in contours[1]]
    res_cons = [res_group.data.cpu().numpy().astype(np.int) for res_group in res_cons]
    # print('res_cons:', res_cons)
    cv2.fillPoly(im_mask, res_cons, 255) #drawContours


    '''
    
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
    ]
    '''
    return torch.from_numpy(im_mask), res_cons


class Masker(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding

    def forward_single_image(self, masks, boxes):
        # boxes = boxes.convert("xyxy")
        im_w, im_h = boxes.size

        res_canvas = []
        res_polygons = []
        for mask, box in zip(masks, boxes.bbox):
            canvas, polygons = paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding)
            res_canvas.append(canvas)
            res_polygons.append(polygons)

        # res = [
        #    paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding)
        #    for mask, box in zip(masks, boxes.bbox)
        # ]
        if len(res_canvas) > 0:
            res_canvas = torch.stack(res_canvas, dim=0)[:, None]
        else:
            res_canvas = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        return res_canvas, res_polygons

    def __call__(self, masks, boxes):
        if isinstance(boxes, RBoxList):
            boxes = [boxes]

        # Make some sanity check
        assert len(boxes) == len(masks), "Masks and boxes should have the same length."

        # TODO:  Is this JIT compatible?
        # If not we should make it compatible.
        result_canvas = []
        result_polygons = []
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box), "Number of objects should be the same."
            res_can, res_poly = self.forward_single_image(mask, box)
            result_canvas.append(res_can)
            result_polygons.append(res_poly)
        return result_canvas, result_polygons


def make_roi_mask_post_processor(cfg):
    if cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS:
        mask_threshold = cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD
        masker = Masker(threshold=mask_threshold, padding=1)
    else:
        masker = None
    mask_post_processor = MaskPostProcessor(masker)
    return mask_post_processor
