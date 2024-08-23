#!/bin/bash

# For installing optox
# Change below path to match the optox folder
cd optox-master
export COMPUTE_CAPABILITY=7.5
export CUDA_ROOT_DIR=/usr/local/cuda
# Change below path to match where these files are on your machine
export CUDA_SDK_ROOT_DIR=/home/liue/duq-project/learned_regularizers_and_geometry_for_image_processing/cuda-samples
mkdir build
cd build
cmake .. -DWITH_TENSORFLOW=ON
make install
python -m unittest optotf.nabla # To check that it was installed correctly
cd ../..


