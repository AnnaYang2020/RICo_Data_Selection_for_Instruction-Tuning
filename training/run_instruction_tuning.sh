#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 TRAIN_JSON OUTPUT_DIR [MODEL] [CUDA_DEVICE] [MASTER_PORT]"
    exit 2
fi

train_json=$1
output_dir=$2
model=${3:-meta-llama/Llama-3.1-8B}
cuda_device=${4:-0}
master_port=${5:-1191}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export WANDB_DISABLED=true
export WANDB_MODE=offline

CUDA_VISIBLE_DEVICES="$cuda_device" torchrun \
    --nproc_per_node=1 \
    --master_port="$master_port" \
    "$script_dir/train.py" \
    --model_name_or_path "$model" \
    --data_path "$train_json" \
    --bf16 True \
    --output_dir "$output_dir" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --deepspeed "$script_dir/config/default_offload_opt_param.json" \
    --logging_steps 1 \
    --tf32 True
