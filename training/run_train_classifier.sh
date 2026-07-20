#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 TRAIN_JSON EVAL_JSON OUTPUT_DIR [MODEL] [CUDA_DEVICE]"
    exit 2
fi

train_json=$1
eval_json=$2
output_dir=$3
model=${4:-meta-llama/Llama-3.1-8B}
cuda_device=${5:-0}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CUDA_VISIBLE_DEVICES="$cuda_device" python "$script_dir/train_classifier.py" \
    --model_name_or_path "$model" \
    --data_path "$train_json" \
    --eval_data_path "$eval_json" \
    --output_dir "$output_dir" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --do_train \
    --do_eval
