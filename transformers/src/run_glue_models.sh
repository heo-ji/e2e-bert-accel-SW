#!/bin/bash

# Define arrays of model paths and task names
model_paths=("ModelTC/bert-base-uncased-mnli" \
            "ModelTC/bert-base-uncased-mrpc" \
            "ModelTC/bert-base-uncased-qnli" \
            "ModelTC/bert-base-uncased-qqp" \
            "ModelTC/bert-base-uncased-rte" \
            "ModelTC/bert-base-uncased-sst2" \
            "ModelTC/bert-base-uncased-stsb" \
            "ModelTC/bert-base-uncased-cola")

task_names=("mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "stsb" "cola" )

# Ensure both arrays have the same length
length=${#model_paths[@]}

# Loop through each model path and task name
for (( i=0; i<${length}; i++ ))
do
    python3 run_glue.py \
    --model_name_or_path ${model_paths[$i]} \
    --task_name ${task_names[$i]} \
    --do_eval \
    --max_seq_length 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir /home/user/HJH/tmp/out/${task_names[$i]}/ \
    --overwrite_output_dir
done


# python run_glue.py \
#   --model_name_or_path ModelTC/bert-base-uncased-cola \
#   --task_name cola  \
# --do_eval \
#   --max_seq_length 128 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3 \
#   --output_dir /home/user/HJH/tmp/out/cola/ \
# --overwrite_output_dir
