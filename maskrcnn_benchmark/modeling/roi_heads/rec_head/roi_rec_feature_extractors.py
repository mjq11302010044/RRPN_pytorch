# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.poolers import PyramidRROIAlign, Pooler
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling import registry


@registry.RROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5RecFeatureExtractor")
class ResNet50Conv5RecFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(ResNet50Conv5RecFeatureExtractor, self).__init__()
        # reso: [H, W]
        resolution = config.MODEL.ROI_REC_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_REC_HEAD.POOLER_SCALES
        pooler = PyramidRROIAlign(
            output_size=resolution,
            scales=scales,
        )

        self.word_margin = config.MODEL.ROI_REC_HEAD.BOXES_MARGIN
        self.det_margin = config.MODEL.RRPN.GT_BOX_MARGIN

        self.rescale = self.word_margin / self.det_margin

        # stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        '''
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )
        '''
        self.pooler = pooler
        # self.head = head

    def forward(self, x, proposals):
        resize_proposals = [proposal.rescale(self.rescale) for proposal in proposals]
        x = self.pooler(x, resize_proposals)
        # x = self.head(x)
        return x


class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio
        )
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = make_conv3x3(next_feature, layer_features, 
                dilation=dilation, stride=1, use_gn=use_gn
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x


_ROI_REC_FEATURE_EXTRACTORS = {
    "ResNet50Conv5RecFeatureExtractor": ResNet50Conv5RecFeatureExtractor,
    "MaskRCNNFPNFeatureExtractor": MaskRCNNFPNFeatureExtractor,
}


def make_roi_rec_feature_extractor(cfg):
    func = _ROI_REC_FEATURE_EXTRACTORS[cfg.MODEL.ROI_REC_HEAD.FEATURE_EXTRACTOR]
    return func(cfg)
