#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 10000
do
CUDA_VISIBLE_DEVICES=2 python train_deform.py -d $data_path \
--data_name bunny --model_name GaussianImage_Cholesky --num_points $num_points --iterations 600000 

#--model_path "./checkpoints/"
done
