#!/usr/bin/env bash
# env definf
device=$1
data_type=$2 # regular (r, b) / temp (b)
task=$3 # ECQA / COSE / ESNLI / QUARTZ (--logging_steps 100)
epochs=$4
lr=$5

MODEL_NAME="t5-large"
OUT_DIR="./output"

python -m rev_train \
        --task ${task} \
        --data_type ${data_type} \
        --out_dir ${OUT_DIR}/${task}_${data_type}-${MODEL_NAME} \
        --model_name_or_path ${MODEL_NAME} \
        --device ${device} \
        --num_train_epochs ${epochs} \
        --learning_rate ${lr} \
        --do_train \
        --do_eval \
        --eval_during_train \
        --save_total_limit 1 \
        --overwrite_cache \
        --max_input_length 300 \
        --max_output_length 20 \
        --logging_steps 1000 \
        --gradient_accumulation_steps 8 \
        --train_batch_size 8 \
        --eval_batch_size 8 \
        --overwrite_out_dir \

