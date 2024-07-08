# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
# 
# Adapted from https://github.com/huggingface/alignment-handbook 

import os
from typing import Dict, TYPE_CHECKING, Literal, Optional, Tuple, Union
from pathlib import Path
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer, PreTrainedModel

from accelerate import Accelerator
from huggingface_hub import list_repo_files
from huggingface_hub.utils._validators import HFValidationError
from peft import LoraConfig, PeftConfig

from .configs import DataArguments, ModelArguments
from .data import DEFAULT_CHAT_TEMPLATE
from dataclasses import dataclass
from transformers.trainer_utils import get_last_checkpoint


def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map() -> Dict[str, int] | None:
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None


def get_quantization_config(model_args) -> BitsAndBytesConfig | None:
    if model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # For consistency with model weights, we use the same value as `torch_dtype` which is float16 for PEFT models
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_tokenizer(model_args: ModelArguments, data_args: DataArguments) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set reasonable default for models without max length
    # if tokenizer.model_max_length > 100_000:
    #     tokenizer.model_max_length = 2048

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template
    elif tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer


def get_peft_config(model_args: ModelArguments) -> PeftConfig | None:
    if model_args.use_peft is False:
        return None

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
    )

    return peft_config


def is_adapter_model(model_name_or_path: str, revision: str = "main") -> bool:
    try:
        # Try first if model on a Hub repo
        repo_files = list_repo_files(model_name_or_path, revision=revision)
    except HFValidationError:
        # If not, check local repo
        repo_files = os.listdir(model_name_or_path)
    return "adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files


@dataclass
class ChatMlSpecialTokens:
    """Dataclass for special tokens used in ChatML, including system, user, assistant, bos, eos, and pad tokens."""

    bos_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"
    pad_token: str = "<|im_end|>"

    @property
    def system(self):
        return f"{self.bos_token}system"

    @property
    def user(self):
        return f"{self.bos_token}user"

    @property
    def assistant(self):
        return f"{self.bos_token}assistant"

    @property
    def chat_template(self):
        return (
            "{% for message in messages %}"
            f"{{{{'{self.bos_token}' + message['role'] + '\n' + message['content'] + '{self.eos_token}' + '\n'}}}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            f"{{{{ '{self.assistant}\n' }}}}"
            "{% endif %}"
        )


FORMAT_MAPPING = {"chatml": ChatMlSpecialTokens}


def setup_chat_format(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    format: Optional[Literal["chatml"]] = "chatml",
    resize_to_multiple_of: Optional[int] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Setup chat format by adding special tokens to the tokenizer, setting the correct format, and extending the embedding layer of the model based on the new special tokens.

    Args:
      model (`~transformers.PreTrainedModel`): The model to be modified.
      tokenizer (`~transformers.PreTrainedTokenizer`): The tokenizer to be modified.
      format (`Optional[Literal["chatml"]]`): The format to be set. Defaults to "chatml".
      resize_to_multiple_of (`Optional[int]`): Number to resize the embedding layer to. Defaults to None.
    Returns:
      model (`~transformers.PreTrainedModel`): The modified model.
      tokenizer (`~transformers.PreTrainedTokenizer`): The modified tokenizer.
    """
    # check if format available and retrieve
    if format not in FORMAT_MAPPING:
        raise ValueError(f"Format {format} not available. Please use one of {FORMAT_MAPPING.keys()}")

    chat_format = FORMAT_MAPPING[format]()

    # set special tokens and them
    tokenizer.eos_token = chat_format.eos_token
    tokenizer.pad_token = chat_format.pad_token
    tokenizer.bos_token = chat_format.bos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [chat_format.bos_token, chat_format.eos_token]})
    # set chat format for tokenizer
    tokenizer.chat_template = chat_format.chat_template

    # resize embedding layer to a multiple of 64, https://x.com/karpathy/status/1621578354024677377
    model.resize_token_embeddings(
        len(tokenizer), pad_to_multiple_of=resize_to_multiple_of if resize_to_multiple_of is not None else None
    )
    # Update the model config to use the new eos & bos tokens
    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
    # Update the generation config to use the new eos & bos token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def get_checkpoint(training_args) -> Path | None:
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint