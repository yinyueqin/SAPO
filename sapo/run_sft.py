#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Supervised fine-tuning script for decoder language models.
"""

import logging
import random
import sys

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
from typing import Any, Dict

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    setup_chat_format
)

logger = logging.getLogger(__name__)
from trl import SFTTrainer

import time
import wandb


def apply_chat_template(
    example, tokenizer, task, assistant_prefix="<|assistant|>\n"
):
    messages = example["real"]
    example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    return example

logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    dir = data_args.dataset_mixer
    data_args.dataset_mixer = {dir: 1}

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")


    data_args.dataset_splits = ['train']
    dataset_name_annotation = 'deita-10k'


    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
    )
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path

    # For ChatML we need to add special tokens and resize the embedding layer
    if "<|im_start|>" in tokenizer.chat_template and "gemma-tokenizer-chatml" not in tokenizer.name_or_path:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        model, tokenizer = setup_chat_format(model, tokenizer)
        model_kwargs = None
    
    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "sft",
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Applying chat template",
    )

    dataset_name = list(data_args.dataset_mixer.keys())[0]


    # Filter out seq > max_length
    #############################
    if training_args.max_seq_length is not None:
        unfiltered_train_samples = len(raw_datasets["train"])
        if "test" in raw_datasets:
            unfiltered_test_samples = len(raw_datasets["test"])

        def filter_fn(sample: Dict[str, Any]) -> Dict[str, Any]:
            max_seq_length = tokenizer(
                sample["text"],
                return_tensors="pt",
            )[
                "input_ids"
            ].size(dim=-1)

            return max_seq_length < training_args.max_seq_length

        raw_datasets = raw_datasets.filter(
            filter_fn,
            desc="Filtering out the samples where len(text_prompt) > max_prompt_length",
        )

        filtered_train_samples = unfiltered_train_samples - len(raw_datasets["train"])
        logger.info(
            f"Filtered out {filtered_train_samples} training samples out of the {unfiltered_train_samples} samples."
        )
        if "test" in raw_datasets:
            filtered_test_samples = unfiltered_test_samples - len(raw_datasets["test"])
            logger.info(
                f"Filtered out {filtered_test_samples} test samples out of the {unfiltered_test_samples} samples."
            )

    train_dataset = raw_datasets["train"]

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=raw_datasets["test"] if "test" in raw_datasets else None,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        peft_config=get_peft_config(model_args),
    )

    start = time.gmtime()

    run_name = f"{model_args.model_name_or_path.split('/')[-1]}-{dataset_name_annotation}-sft-epoch{training_args.num_train_epochs}-{start.tm_mday}-{start.tm_hour}-{start.tm_min}"
    if trainer.accelerator.is_main_process:
        wandb.init(name=run_name, project='SAPO')

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = last_checkpoint
    if checkpoint is not None:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
    else:
        train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
