# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .coco import COCODataset
from .voc import PascalVOCDataset
from .concat_dataset import ConcatDataset
from .icdar_series import ICDAR2013Dataset
from .rotation_series import RotationDataset
from .rrpn_e2e_series import SpottingDataset
from .rotation_mask_datasets import RotationMaskDataset

__all__ = ["COCODataset", "ConcatDataset", "PascalVOCDataset", 'ICDAR2013Dataset', 'RotationDataset', 'SpottingDataset', 'RotationMaskDataset']
