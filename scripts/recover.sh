#!/bin/bash

model_dirs=($(find <PATH_TO_SAVED_MODELS> -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))

i_4=(0)

for model_name  in "${model_dirs[@]}"; do
    for i in "${i_4[@]}"; do
        CUDA_VISIBLE_DEVICES=3 python recover.py --testmodelpath="<PATH_TO_SAVED_MODELS>/${model_name}/${i}" \
        --adv_path="<PATH_TO_ATTACKED_IMAGES>/${model_name}/${i}.png" \
        --r_path="<PATH_TO_RECOVERED_IMAGES>/${model_name}/${i}.png" \
        --test_img="${model_name}" \
        --test_i="${i}" \
        --cover_path="<PATH_TO_COVER_IMAGES>/${model_name}/${i}.png" 
    done
done
