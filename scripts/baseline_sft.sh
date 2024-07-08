#!/bin/bash
# bash scripts/baseline_sft.sh sft 3 deita llama

loss=$1
epoch=$2
dataset_type=$3
model_type=$4

cache_dir="/home/workspace/sapo/${model_type}-${dataset_type}-${loss}-${epoch}epoch"

data_dir="${cache_dir}/datasets"

huggingface_token='your huggingface token'
huggingface_model_prefix='your huggingface model prefix'

huggingface-cli login --token ${huggingface_token}

python sapo/reformat.py --output_dir ${data_dir} --data HuggingFaceH4/deita-10k-v0-sft


ACCELERATE_LOG_LEVEL=info

if [[ "$model_type" == *"mistral"* ]]; then
    model="mistralai/Mistral-7B-v0.1"
    hub_model_id="${huggingface_model_prefix}/wandb-zephyr-7b-${dataset_type}-${loss}-${epoch}epoch"
elif [[ "$model_type" == *"llama"* ]]; then
    model="meta-llama/Meta-Llama-3-8B"
    hub_model_id="${huggingface_model_prefix}/llama3-8b-${dataset_type}-${loss}-${epoch}epoch"
fi

train_cmd="accelerate launch --config_file configs/deepspeed_zero3.yaml --num_processes=8 sapo/run_sft.py configs/config_sft.yaml --model_name_or_path=${model} --dataset_mixer=${data_dir} --hub_model_id=${hub_model_id} --output_dir=${cache_dir}/outputs/ --num_train_epochs=${epoch}"

eval $train_cmd

echo "train done"