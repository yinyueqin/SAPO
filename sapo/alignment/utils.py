# Adapted from https://github.com/huggingface/alignment-handbook 

from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import DataCollatorForLanguageModeling, PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
import random


from accelerate.utils import is_deepspeed_available

from contextlib import contextmanager

if is_deepspeed_available():
    import deepspeed


if TYPE_CHECKING:
    from accelerate import Accelerator
    from deepspeed.runtime.engine import DeepSpeedEngine
    from torch.nn.parallel.distributed import DistributedDataParallel


class ReplayBuffer:
    def __init__(self, max_size=20000,tokenizer_pad_token=0, label_pad_token_id=-100):
        self.buffer = deque(maxlen=max_size)
        self.tokenizer_pad_token = tokenizer_pad_token
        self.label_pad_token_id = -100
        self.sample_counts = defaultdict(int) # 维护每个样本的采样次数

    def save(self, buffer_file):
        if buffer_file:
            torch.save({
                "buffer": list(self.buffer),
                "sample_counts": self.sample_counts
            }, buffer_file)

    def load(self, buffer_file):
        if buffer_file and os.path.exists(buffer_file):
            data = torch.load(buffer_file)
            self.buffer.extend(data["buffer"])
            self.sample_counts = data["sample_counts"]


    def add(self, experience):
        """ Add experiences to the replay buffer. """
        if len(self.buffer) >= self.buffer.maxlen:
            removed_exp = self.buffer.popleft()
            self.sample_counts.pop(id(removed_exp), None)
        self.buffer.append(experience)

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                # adapted from https://stackoverflow.com/questions/73256206
                if "prompt" in k:
                    to_pad = [ex[k][::-1] for ex in batch]
                else:
                    to_pad = [ex[k] for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer_pad_token
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                elif k.endswith("_attention_mask"):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                # for the prompt, flip back so padding is on left side
                if "prompt" in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]
                
        return padded_batch

    def sample(self, batch_size):
        weights = [1.0 / (self.sample_counts[id(exp)] + 1) for exp in self.buffer]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        sampled_experiences = random.choices(self.buffer, weights=probabilities, k=min(len(self.buffer), batch_size))
        for exp in sampled_experiences:
            self.sample_counts[id(exp)] += 1

        return self.collate(sampled_experiences)

    def __len__(self):
        """ Return the current size of the buffer. """
        return len(self.buffer)

    def clear(self):
        """ Clear the buffer. """
        self.buffer.clear()
        self.sample_counts.clear()



@dataclass
class DataCollatorWithPadding:
    r"""
    DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        model (Optional[`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
        max_target_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the target to be processed. Only useful for encoder-decoder architectures.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt.
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    is_encoder_decoder: Optional[bool] = False
    max_target_length: Optional[int] = None
    token_generate_length: int = 10

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
    ) -> Dict:
        batch = {}

        if not self.is_encoder_decoder:
            chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)

            eos_token_id = self.tokenizer.eos_token_id
            eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask = [
                0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
            ]
            prompt_tokens["attention_mask"] = new_attention_mask

            eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_c = [
                0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
            ]
            chosen_tokens["attention_mask"] = new_attention_mask_c
            
            chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            actual_token_generate_length = min(self.token_generate_length, len(chosen_tokens["input_ids"]))
            available_space = self.max_length - min(len(prompt_tokens["input_ids"]),self.max_prompt_length) - actual_token_generate_length
            max_truncate_at = len(chosen_tokens['input_ids']) -3 - actual_token_generate_length 

            if max_truncate_at > 0:
                truncate_at = random.randint(0, min(max_truncate_at, available_space))
            else:
                truncate_at = 0  

            # Truncate prompt
            if len(prompt_tokens["input_ids"]) + len(chosen_tokens["input_ids"]) > self.max_length:
                if self.truncation_mode == "keep_start":
                    prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
                elif self.truncation_mode == "keep_end":
                    prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            if len(prompt_tokens["input_ids"]) + len(chosen_tokens["input_ids"]) > self.max_length:
                additional_truncate = len(prompt_tokens["input_ids"]) + truncate_at + actual_token_generate_length - self.max_length
                truncate_at = max(0, truncate_at - additional_truncate)
    
                chosen_tokens = {k: v[:self.max_length - len(prompt_tokens["input_ids"])] for k, v in chosen_tokens.items()}

            # Create labels
            chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}

            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                prompt_tokens["input_ids"]
            )

            for k, toks in {
                "real": chosen_sequence_tokens,
                "prompt": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}_{type_key}"] = tokens

        batch["prompt_length"] = len(prompt_tokens["input_ids"])
        batch["chosen_input_length"] = len(prompt_tokens['input_ids']) + len(chosen_tokens['input_ids'])
        batch["truncate_at"] = truncate_at
        batch["actual_token_generate_length"] = actual_token_generate_length
        batch["prompt"] = prompt
        batch["real"] = prompt + chosen
        batch["chosen_response_only"] = chosen

        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (k.startswith("real")) or (k.startswith("generated")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    padded_batch[k] = to_pad
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["real"]

            batch_element = self.tokenize_batch_element(prompt, chosen)
            tokenized_batch.append(batch_element)

        return self.collate(tokenized_batch)

def remove_hooks(model: "DeepSpeedEngine") -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []


def add_hooks(model: "DeepSpeedEngine") -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    optimizer_offload._register_hooks_recursively(optimizer_offload.module)


@contextmanager
def unwrap_model_for_generation(
    model: Union["DistributedDataParallel", "DeepSpeedEngine"], accelerator: "Accelerator", is_peft_model: bool = False
) -> Union["PreTrainedModelWrapper", "DeepSpeedEngine"]:
    """Context manager to unwrap a model for generation.

    For ZeRO-3 models, we gather the weights once to speed up generation.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    if is_peft_model:
        unwrapped_model.pretrained_model.disable_adapter()
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        with deepspeed.zero.GatheredParameters(model.parameters()):
            remove_hooks(model)
            yield model
            add_hooks(model)
    else:
        yield unwrapped_model
