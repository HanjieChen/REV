#!/usr/bin/env bash

device=$1
split=$2 # test
test_type=$3 # gold / gen
out_type=$4 # YR / R / RY (test_type = gen), Y (test_type = gold)
model_name=$5 # t5-large
task=$6 # ECQA / COSE / ESNLI / QUARTZ

python -m rev_eval \
        --task ${task} \
        --test_type ${test_type} \
        --model_name ${model_name} \
        --out_type ${out_type} \
        --split ${split} \
        --beams 1 \
        --device ${device} \
        --min_length 1 \
