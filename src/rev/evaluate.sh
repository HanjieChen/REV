#!/usr/bin/env bash

device=$1
split=$2 # val / test
test_type=$3 # gold / gen / temp
out_type=$4 # YR / R / RY
model_name=$6 # t5-large
task=$7 # ECQA / COSE / ESNLI / QUARTZ

declare -a lms=(t5-large) #(bart-large) # t5-large) #(gpt2 gpt2xl bart-large t5-large)


# for lm in "${lms[@]}"
# do
python -m rev_eval \
        --task ${task} \
        --test_type ${test_type} \
        --model_name ${model_name} \
        --out_type ${out_type} \
        --split ${split} \
        --beams 1 \
        --device ${device} \
        --min_length 1 \
#     wait
#   done
# done
