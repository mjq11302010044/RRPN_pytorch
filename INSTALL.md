## Installation

### Requirements:
- PyTorch 1.0 from a nightly release. Installation instructions can be found in https://pytorch.org/get-started/locally/
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV 



```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name rrpn_pytorch python=3.6
source activate rrpn_pytorch

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0 
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0
pip install --upgrade torchvision==0.4.0

#the following packages can be installed either from source or pip
# install pycocotools
cd ~/github
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
cd ~/github
git clone https://github.com/mjq11302010044/RRPN_pytorch.git
cd RRPN_pytorch
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop
#you may encounter some 'no module' problems,just pip 
rm -rf build/
#-------
python rotation setup.py install
#rm -rf rrpn egginfo later
#lib has suffix, use TAB to auto complete
mv build/lib/rotation/*.so ./rotation
#-------

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```
