#finetuned model
python3 run_image_classification.py \
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
    --seed 1337 \
    --layernorm_method original  \
    --softmax_method original

# python3 run_image_classification.py \
#     --model_name_or_path timm/vit_base_patch32_clip_448.laion2b_ft_in12k_in1k \
#     --dataset_name theodor1289/imagenet-1k_tiny\
#     --output_dir /home/user/HJH/results_vit_base/imagenet-1k_tiny \
#     --remove_unused_columns False \
#     --do_eval \
#     --per_device_eval_batch_size 8 \
#     --logging_strategy steps \
#     --logging_steps 10 \
#     --eval_strategy epoch \
#     --save_strategy epoch \
#     --load_best_model_at_end True \
#     --save_total_limit 3 \
#     --seed 1337 \
#     --layernorm_method original  \
#     --softmax_method original
