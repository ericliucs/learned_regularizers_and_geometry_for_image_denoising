#!/bin/bash

# For installing optox
cd optox_local_clone
export COMPUTE_CAPABILITY=7.5
export CUDA_ROOT_DIR=/usr/local/cuda 
export CUDA_SDK_ROOT_DIR=/home/liue/duq-project/cuda-samples-master
mkdir build
cd build
cmake .. -DWITH_TENSORFLOW=ON
make install
python -m unittest optotf.nabla # To check that it was installed correctly
cd ../..


