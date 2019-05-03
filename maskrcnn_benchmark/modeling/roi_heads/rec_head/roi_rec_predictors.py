# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F
import os
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.modeling.make_layers import group_norm as GN

class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv2dGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, gn=False):
        super(Conv2dGroup, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.gn = GN(out_channels) if gn else None # nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if gn else None #
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RECG(nn.Module):
    def __init__(self, char_class, g_feat_channel=1024, inter_channel=256, gn=True):
        super(RECG, self).__init__()

        self.rec_conv1 = nn.Sequential(Conv2dGroup(g_feat_channel, inter_channel, 3, same_padding=True, gn=gn),
                                       Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, gn=gn),
                                       nn.Conv2d(inter_channel, inter_channel, 3, (2, 1), 1))

        inter_channel *= 2

        self.rec_conv2 = nn.Sequential(Conv2dGroup(inter_channel // 2, inter_channel, 3, same_padding=True, gn=gn),
                                       Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, gn=gn),
                                       nn.Conv2d(inter_channel, inter_channel, 3, (2, 1), 1))

        inter_channel *= 2

        self.rec_conv3 = nn.Sequential(Conv2dGroup(inter_channel // 2, inter_channel, 3, same_padding=True, gn=gn),
                                       Conv2dGroup(inter_channel, inter_channel, 3, same_padding=True, gn=gn),
                                       nn.Conv2d(inter_channel, inter_channel, 3, (2, 1), 1))

        # input with shape of [w, b, c] --> [20 timestamps, x fg_nums, 256 channels]
        self.blstm = nn.LSTM(inter_channel, int(inter_channel/2), bidirectional=True)
        self.embeddings = FC(inter_channel, char_class, relu=False)

    def forward(self, rec_pooled_features):

        rec_x = self.rec_conv1(rec_pooled_features)
        rec_x = self.rec_conv2(rec_x)
        rec_x = self.rec_conv3(rec_x)
        c_feat = rec_x.squeeze(2)
        c_feat = c_feat.permute(2, 0, 1)#.contiguous()

        recurrent, _ = self.blstm(c_feat)
        T, b, h = recurrent.size()
        rec_x = recurrent.view(T * b, h)
        predict = self.embeddings(rec_x)
        predict = predict.view(T, b, -1)

        return predict


class RRPNRecC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(RRPNRecC4Predictor, self).__init__()

        al_profile = cfg.MODEL.ROI_REC_HEAD.ALPHABET

        if os.path.isfile(al_profile):
            num_classes = len(open(al_profile, 'r').read())+1
        else:
            print("We don't expect you to use default class number...Retry it")
            num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 3
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        # (c2, c3, c4)
        if cfg.MODEL.FP4P_ON:
            num_inputs = 1024 + 512 + 256

        # input feature size with [N, 1024, 8, 35]
        self.rec_head = RECG(num_classes, num_inputs, dim_reduced)

        for name, param in self.named_parameters():
            # print('name:', name)
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name and not 'gn' in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif "weight" in name and 'bn' in name:
                param.data.fill_(1)
            elif "bias" in name and 'bn' in name:
                param.data.fill_(0)
    def forward(self, x):
        return self.rec_head(x)


class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


_ROI_REC_PREDICTOR = {"MaskRCNNC4Predictor": MaskRCNNC4Predictor,
                       "RRPNE2EC4Predictor": RRPNRecC4Predictor}


def make_roi_rec_predictor(cfg):
    func = _ROI_REC_PREDICTOR[cfg.MODEL.ROI_REC_HEAD.PREDICTOR]
    return func(cfg)
