# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .rbox_head.box_head import build_roi_box_head
from .rec_head.rec_head import build_roi_rec_head
from .rmask_head.mask_head import build_roi_mask_head

class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.REC_ON and cfg.MODEL.ROI_REC_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.rec.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        if self.cfg.MODEL.FP4P_ON:
            # get you C4
            x, detections, loss_box = self.box((features[-1], ), proposals, targets)
        else:
            x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)

        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            # detach process
            mask_features_detach = [feature.detach() for feature in mask_features]
            x, detections, loss_mask = self.mask(mask_features_detach, detections, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.REC_ON:
            rec_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_REC_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                rec_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_rec = self.rec(rec_features, detections, targets)
            losses.update(loss_rec)
        return x, detections, losses


def build_roi_heads(cfg):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg)))
    if cfg.MODEL.REC_ON:
        roi_heads.append(("rec", build_roi_rec_head(cfg)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
