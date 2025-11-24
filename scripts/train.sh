#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
    --inputpath_all "<PATH_TO_ORIGINAL_DATASET>"\
    --copyrightpath "<PATH_TO_WATERMARK_IMAGE>" \
    --outputpath "<PATH_TO_OUTPUT_DIR>" \
    --T2Imodel "<PATH_TO_Pretrained_T2I_Model>"