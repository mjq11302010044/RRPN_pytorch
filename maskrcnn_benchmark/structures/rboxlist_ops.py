# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np
from .bounding_box import RBoxList

from maskrcnn_benchmark.layers import nms as _box_nms
from rotation.rotate_polygon_nms import rotate_gpu_nms
from rotation.rbbox_overlaps import rbbx_overlaps


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="score", GPU_ID=0):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maxium suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist

    # boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)

    ##################################################
    # convert to numpy before calculate
    boxes_np = boxes.data.cpu().numpy()
    score_np = score.data.cpu().numpy()
    # keep = _box_nms(boxes, score, nms_thresh)
    ch_proposals = boxes_np.copy()
    ch_proposals[:, 2:4] = ch_proposals[:, 3:1:-1]
    # x,y,h,w,a

    # print('ch_proposals:',ch_proposals.shape)
    # print('score_np:', score_np.shape)

    if ch_proposals.shape[0] < 1:
        return boxlist

    keep = rotate_gpu_nms(np.array(np.hstack((ch_proposals, score_np[..., np.newaxis])), np.float32), nms_thresh, GPU_ID)  # D
    # print time.time() - tic
    if max_proposals > 0:
        keep = keep[:max_proposals]

    keep_th = torch.tensor(keep, dtype=torch.long).to(boxlist.bbox.device)

    # print('keep_th:', keep_th.type())
    ##################################################
    # proposals = proposals[keep, :]
    # scores = scores[keep]

    # if max_proposals > 0:
    #     keep = keep[:max_proposals]
    boxlist = boxlist[keep_th]

    # print('boxlist:', boxlist.bbox.type())

    return boxlist #.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywha_boxes = boxlist.bbox
    _, _, ws, hs, a_s = xywha_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2, GPU_ID=0):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,5].
      box2: (BoxList) bounding boxes, sized [M,5].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    eps = 1e-8

    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    if boxlist1.bbox.size()[0] < 1 or boxlist1.bbox.size()[0] < 1:
        raise RuntimeError(
                "boxlists should have size larger than 0, got {}, {}".format(boxlist1.bbox.size()[0], boxlist1.bbox.size()[0]))

    ###########################################################
    box1, box2 = boxlist1.bbox, boxlist2.bbox

    box1_np = box1.data.cpu().numpy()
    box2_np = box2.data.cpu().numpy()

    ch_box1 = box1_np.copy()
    ch_box1[:, 2:4] = ch_box1[:, 3:1:-1]
    ch_box2 = box2_np.copy()
    ch_box2[:, 2:4] = ch_box2[:, 3:1:-1]

    #ch_box2[:, 2:4] += 16

    overlaps = rbbx_overlaps(np.ascontiguousarray(ch_box1, dtype=np.float32),
                             np.ascontiguousarray(ch_box2, dtype=np.float32), GPU_ID)

    #print('ch_box shape:', ch_box1.shape, ch_box2.shape)
    #print('ch_box shape:', ch_box1[:, 2:4], ch_box2[:, 2:4], ch_box2[:, 4])
    #print('overlaps_shape:', overlaps.shape)
    #print('overlaps:', np.unique(overlaps)[:10], np.unique(overlaps)[-10:])
    ############################################
    # Some unknown bug on complex coordinate
    overlaps[overlaps > 1.00000001] = 0.0
    ############################################

    ###########################################################
    '''
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    '''
    # print('bbox_shape_monitor:', overlaps.shape, boxlist1.bbox.device, boxlist2.bbox.device)

    overlaps_th = torch.tensor(overlaps).to(boxlist1.bbox.device)

    return overlaps_th


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, RBoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = RBoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes
