#!/bin/bash
# bash scripts/sapo_orpo.sh orpo 4 256 0.5 capybara 1 1 2000 mistral
# bash scripts/sapo_orpo.sh orpo 4 256 0.5 capybara 1 1 2000 llama

export PATH="/home/aiscuser/.local/bin:$PATH"

loss=$1
epoch=$2
num_token_gen=$3
ema_beta=$4
dataset_type=$5
sample_every_n_steps=$6
responses_per_prompt=$7
max_replay_buffer_size=$8
model_type=$9

cache_dir="/home/workspace/sapo/sapo-${dataset_type}-ema${ema_beta}-${model_type}-numtg${num_token_gen}-${loss}-${epoch}epoch-${sample_every_n_steps}samplestep-${responses_per_prompt}response-buffersize${max_replay_buffer_size}"


data_dir="${cache_dir}/datasets"

huggingface_token='your huggingface token'
huggingface_model_prefix='your huggingface model prefix'

huggingface-cli login --token ${huggingface_token}

python sapo/reformat.py --output_dir ${data_dir} --data argilla/distilabel-capybara-dpo-7k-binarized

ACCELERATE_LOG_LEVEL=info

if [[ "$model_type" == *"mistral"* ]]; then
    model="mistralai/Mistral-7B-v0.1"
    hub_model_id="${huggingface_model_prefix}/sapo-zephyr-7b-${dataset_type}-ema${ema_beta}-numtg${num_token_gen}-${loss}-${epoch}epoch-${sample_every_n_steps}sf-${responses_per_prompt}rp-bs${max_replay_buffer_size}"
elif [[ "$model_type" == *"llama"* ]]; then
    model="meta-llama/Meta-Llama-3-8B"
    hub_model_id="${huggingface_model_prefix}/sapo-llama3-8b-${dataset_type}-ema${ema_beta}-numtg${num_token_gen}-${loss}-${epoch}epoch-${sample_every_n_steps}sf-${responses_per_prompt}rp-bs${max_replay_buffer_size}"
fi


if [[ "$model_type" == *"mistral"* ]]; then
    train_cmd="accelerate launch --config_file configs/deepspeed_zero3.yaml --num_processes=8 sapo/run_orpo.py configs/config_orpo.yaml --model_name_or_path=${model} --dataset_mixer=${data_dir} --hub_model_id=${hub_model_id} --output_dir=${cache_dir}/outputs/ --num_train_epochs=${epoch} --max_length=2048 --max_prompt_length=1792 --token_generate_length=${num_token_gen} --ema_beta=${ema_beta} --sample_every_n_steps=${sample_every_n_steps} --responses_per_prompt=${responses_per_prompt} --max_replay_buffer_size=${max_replay_buffer_size}"
elif [[ "$model_type" == *"llama"* ]]; then
    train_cmd="accelerate launch --config_file configs/deepspeed_zero3.yaml --num_processes=8 sapo/run_orpo.py configs/config_orpo_llama.yaml --model_name_or_path=${model} --dataset_mixer=${data_dir} --hub_model_id=${hub_model_id} --output_dir=${cache_dir}/outputs/ --num_train_epochs=${epoch} --max_length=2048 --max_prompt_length=1792 --token_generate_length=${num_token_gen} --ema_beta=${ema_beta} --sample_every_n_steps=${sample_every_n_steps} --responses_per_prompt=${responses_per_prompt} --max_replay_buffer_size=${max_replay_buffer_size}"
fi    

eval $train_cmd

echo "train done"