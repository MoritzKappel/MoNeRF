#! /usr/bin/env bash
# create new env
ENV_NAME=monerf
conda create --name $ENV_NAME python=3.11
source ~/.bashrc
conda activate $ENV_NAME
# dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy tqdm natsort GitPython av ffmpeg-python pyyaml munch tabulate wandb opencv-python kornia torchmetrics lpips einops setuptools
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
conda install packaging
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex
