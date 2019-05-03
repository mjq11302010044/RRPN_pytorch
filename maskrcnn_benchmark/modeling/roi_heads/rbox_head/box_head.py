# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

        self.cfg = cfg

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        # if self.cfg.TEST.CASCADE:
        recur_iter = self.cfg.MODEL.ROI_HEADS.RECUR_ITER if self.cfg.TEST.CASCADE else 1

        recur_proposals = proposals
        x = None
        for i in range(recur_iter):

            if self.training:
                # Faster R-CNN subsamples during training the proposals with a fixed
                # positive / negative ratio
                with torch.no_grad():
                    recur_proposals = self.loss_evaluator.subsample(recur_proposals, targets)

            # extract features that will be fed to the final classifier. The
            # feature_extractor generally corresponds to the pooler + heads
            x = self.feature_extractor(features, recur_proposals)
            # final classifier that converts the features into predictions
            class_logits, box_regression = self.predictor(x)

            if not self.training:
                recur_proposals = self.post_processor((class_logits, box_regression), recur_proposals, recur_iter - i - 1) # result
            else:
                loss_classifier, loss_box_reg = self.loss_evaluator(
                    [class_logits], [box_regression]
                )
        if not self.training:
            return x, recur_proposals, {}

        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
