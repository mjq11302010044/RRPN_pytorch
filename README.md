# RRPN_pytorch
RRPN in pytorch, which is implemented into facebook's benchmark: https://github.com/facebookresearch/maskrcnn-benchmark. 
Its caffe version can be viewed at: https://github.com/mjq11302010044/RRPN.

## Highlights
- **From original repo:** In pytorch 1.0, Somehow faster than original repo in both training and inference.
- **Training and evaluation checked:** Testing in IC15 with training data in {IC13, IC15, IC17}, and receives Fscore of 83% vs. 81% in caffe repo.
- **What's new:** RRoI Pooling is replaced with RRoI Align, FPN structure supported, easy to change various backbones for different purposes.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Training 
```bash
# In your root of RRPN
python tools/train_net.py --config-file=configs/rrpn/e2e_rrpn_R_50_C4_1x_ICDAR13_15_17_trial.yaml
```
