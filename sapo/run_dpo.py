#!/usr/bin/env python
# 
# Adapted from https://github.com/huggingface/alignment-handbook 
import logging
import os
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
from typing import Any, Dict


from accelerate import Accelerator
from alignment import (
    DataArguments,
    SAPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
    get_checkpoint
)
from peft import PeftConfig, PeftModel
from alignment import SAPOTrainer
from torch.utils.data import Subset
import re
import json
import wandb
import time

def apply_chat_template(
    example, tokenizer, task, assistant_prefix="<|assistant|>\n"
):

    prompt_messages =  example["real"][:-1]

    real_messages = example["real"][-1:]
    example["text_real"] = tokenizer.apply_chat_template(real_messages, tokenize=False)
    example["text_prompt"] = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False
    )

    return example

logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SAPOConfig))
    model_args, data_args, training_args = parser.parse()
    dir = data_args.dataset_mixer
    data_args.dataset_mixer = {dir: 1}

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")


    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Increase distributed timeout to 3h to enable push to Hub to complete
    accelerator = Accelerator()


    data_args.dataset_splits = ['train']
    dataset_name_annotation = 'capybara'

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "sapo"},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Filter out seq > max_length
    #############################
    if training_args.max_prompt_length is not None:
        unfiltered_train_samples = len(raw_datasets["train"])
        if "test" in raw_datasets:
            unfiltered_test_samples = len(raw_datasets["test"])

        def filter_fn(sample: Dict[str, Any]) -> Dict[str, Any]:
            prompt_length = tokenizer(
                sample["text_prompt"],
                return_tensors="pt",
            )[
                "input_ids"
            ].size(dim=-1)

            return prompt_length < training_args.max_prompt_length

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

    # Replace column names with what TRL needs, text_real -> real and text_generated -> generated
    for split in data_args.dataset_splits:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_real": "real"}
        )

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
    ref_model = model

    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    #########################
    # Instantiate sapo trainer
    #########################
    sapo_trainer = SAPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"] if "test" in raw_datasets else None,
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        token_generate_length=training_args.token_generate_length,
        peft_config=get_peft_config(model_args),
        loss_mode=training_args.loss_mode,
        ema_beta=training_args.ema_beta,
        max_replay_buffer_size=training_args.max_replay_buffer_size,
        sample_every_n_steps=training_args.sample_every_n_steps,
        responses_per_prompt=training_args.responses_per_prompt,
        batch_size=training_args.per_device_train_batch_size,
        ref_update_mode=training_args.ref_update_mode,
    )

    start = time.gmtime()

    run_name = f"{model_args.model_name_or_path.split('/')[-1]}-{dataset_name_annotation}-{training_args.loss_mode}-epoch{training_args.num_train_epochs}-{start.tm_mday}-{start.tm_hour}-{start.tm_min}"
    if sapo_trainer.accelerator.is_main_process:
        wandb.init(name=run_name, project='SAPO')

    ###############
    # Training loop
    ###############
    checkpoint = last_checkpoint
    if checkpoint is not None:
        train_result = sapo_trainer.train(resume_from_checkpoint=checkpoint)
    else:
        train_result = sapo_trainer.train()
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(raw_datasets["train"])
    )
    metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
    sapo_trainer.log_metrics("train", metrics)
    sapo_trainer.save_metrics("train", metrics)
    sapo_trainer.save_state()
    
    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    sapo_trainer.save_model(training_args.output_dir)
    sapo_trainer.save_ema_model(os.path.join(training_args.output_dir, "checkpoint-ema"))
    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        sapo_trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        sapo_trainer.model.config.use_cache = True
        sapo_trainer.model.config.save_pretrained(training_args.output_dir)

    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    accelerator.wait_for_everyone()

    logger.info("*** Run complete! ***")


if __name__ == "__main__":
    main()
