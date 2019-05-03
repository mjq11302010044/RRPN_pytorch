# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None, d2s=False
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

        # self.channel_aligned = conv_block(int(out_channels / 4), out_channels, 3, 1)
        # self.pxsf = F.pixel_shuffle

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):

            inner_lateral = getattr(self, inner_block)(feature)
            # print('inner_lateral.size:', inner_lateral.size())
            # inner_top_down = F.interpolate(last_inner, size=inner_lateral.size()[-2:], mode="bilinear") #scale_factor=2, nearest
            # TODO use size instead of scale to make it robust to different sizes

            #------------- D2S -------------
            #inner_top_down_pxsf = F.pixel_shuffle(last_inner, 2)
            #inner_top_down_bfal = F.upsample(inner_top_down_pxsf, size=inner_lateral.shape[-2:], mode='bilinear')#bilinear , align_corners=False
            #inner_top_down = self.channel_aligned(inner_top_down_bfal)
            # ------------------------------

            # print('shape:', inner_lateral.shape, inner_top_down.shape)
            inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[-2:], mode='bilinear', align_corners=False) #bilinear
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        if self.top_blocks is not None:
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]
