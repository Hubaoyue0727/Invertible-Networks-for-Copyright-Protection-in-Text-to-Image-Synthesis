#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python train.py \
    --inputpath_all "./data/VGGFace2_demo"\
    --copyrightpath "./data/copyright.png" \
    --T2Imodel "stabilityai/stable-diffusion-2-1"