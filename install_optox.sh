#!/bin/bash

# For installing optox
cd /tmp
git clone git@github.com:VLOGroup/optox.git
cd optox
export COMPUTE_CAPABILITY=7.5
export CUDA_ROOT_DIR=/usr/local/cuda 
export CUDA_SDK_ROOT_DIR=/use/local/cuda/samples
mkdir build
cd build
cmake .. -DWITH_TENSORFLOW=ON
make install
python -m unittest optotf.nabla # To check that it was installed correctly
cd ../..
rm -r -f optox

