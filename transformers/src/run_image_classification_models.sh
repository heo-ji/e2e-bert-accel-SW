#finetuned model
python run_image_classification.py \
    --model_name_or_path nateraw/vit-base-patch16-224-cifar10 \
    --dataset_name cifar10 \
    --output_dir /home/user/HJH/results_vit_base/cifar10 \
    --remove_unused_columns False \
    --label_column_name label \
    --image_column_name img \
    --do_eval \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337