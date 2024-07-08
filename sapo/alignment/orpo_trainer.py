import inspect
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import EvalLoopOutput

from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import disable_dropout_in_model, pad_to_length
from torch.nn.utils.rnn import pad_sequence

from .utils import DataCollatorWithPadding, unwrap_model_for_generation, ReplayBuffer


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

from sentence_transformers import SentenceTransformer

from .orpo_config import ORPOConfig

from copy import deepcopy

from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from huggingface_hub import upload_folder, create_repo

TRAINING_ARGS_NAME = "training_args.bin"


def _z3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]

def find_latest_checkpoint(saved_dir):
    if not os.path.exists(saved_dir):
        return None
    checkpoints = [d for d in os.listdir(saved_dir) if d.startswith("checkpoint-")]

    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]), reverse=True)[0]    
    return os.path.join(saved_dir, latest_checkpoint)

class ORPOTrainer(Trainer):
    r"""
    Initialize ORPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        args (`ORPOConfig`):
            The ORPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
    """

    _tag_names = ["trl", "orpo"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ema_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[ORPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
    ):  
        self.ema_beta = args.ema_beta
        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the ORPOTrainer. But your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            model_init_kwargs["torch_dtype"] = (
                model_init_kwargs["torch_dtype"]
                if model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, model_init_kwargs["torch_dtype"])
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the ORPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            # if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
            #     peft_module_casting_to_bf16(model)
            #     # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
            #     self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        if self.is_encoder_decoder:
            self.decoder_start_token_id = model.config.decoder_start_token_id
            self.pad_token_id = model.config.pad_token_id

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a ORPO dataset.")
        if args.max_length is None:
            warnings.warn(
                "`max_length` is not set in the ORPOConfig's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        else:
            max_length = args.max_length
        if args.max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the ORPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128
        else:
            max_prompt_length = args.max_prompt_length

        if args.max_completion_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_completion_length` in the ORPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            self.max_completion_length = 128
        else:
            self.max_completion_length = args.max_completion_length

        self.token_generate_length = args.token_generate_length
        print('self.token_generate_length', self.token_generate_length)

            
        if data_collator is None:
            data_collator = DataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                token_generate_length=self.token_generate_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using SAPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_data_collator = True
        else:
            self.use_data_collator = False
        if args.disable_dropout:
            disable_dropout_in_model(model)

        self.max_length = max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = args.truncation_mode
        self.tokenizer = tokenizer

        self.beta = args.beta

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        # with PartialState().local_main_process_first():
        #     # tokenize the dataset
        #     train_dataset = train_dataset.map(self.tokenize_row, num_proc=args.dataset_num_proc)
        #     if eval_dataset is not None:
        #         eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=args.dataset_num_proc)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )


        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )
        
        resume_ema_model_path = find_latest_checkpoint(os.path.join(self.args.output_dir, "checkpoint-ema"))
        if resume_ema_model_path is not None:
            ema_model = AutoModelForCausalLM.from_pretrained(resume_ema_model_path, **model_init_kwargs)

        if self.is_deepspeed_enabled:
            ema_model = self._prepare_deepspeed(ema_model)
        else:
            ema_model = self.accelerator.prepare_model(ema_model, evaluation_mode=True)

        self.ema_model = ema_model
        self.time_steps = defaultdict(int)

        self.max_replay_buffer_size = args.max_replay_buffer_size

        self.replay_buffer = ReplayBuffer(max_size=self.max_replay_buffer_size,tokenizer_pad_token=self.tokenizer.pad_token_id, label_pad_token_id=self.label_pad_token_id)
        resume_replaybuffer_path =  find_latest_checkpoint(os.path.join(self.args.output_dir, "checkpoint-replaybuffer"))
        if resume_replaybuffer_path is not None:
            rank_replaybuffer_path = os.path.join(resume_replaybuffer_path, f"replay_buffer_{self.accelerator.state.process_index}.pt")
            self.replay_buffer.load(rank_replaybuffer_path)
            
        self.sample_every_n_steps = args.sample_every_n_steps
        self.responses_per_prompt = args.responses_per_prompt
        self.batch_size = args.per_device_train_batch_size

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def concatenated_inputs(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["real_labels"].shape[1], batch["generated_labels"].shape[1])
        else:
            max_length = max(batch["real_input_ids"].shape[1], batch["generated_input_ids"].shape[1])

        for k in batch:
            if k.startswith("real") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("real", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("generated") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("generated", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    def odds_ratio_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute ORPO's odds ratio (OR) loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the ORPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
            The log odds ratio of the chosen responses over the rejected responses ratio for logging purposes.
            The `log(sigmoid(log_odds_chosen))` for logging purposes.
        """

        # Derived from Eqs. (4) and (7) from https://arxiv.org/abs/2403.07691 by using log identities and exp(log(P(y|x)) = P(y|x)
        log_odds = (policy_chosen_logps - policy_rejected_logps) - (
            torch.log1p(-torch.exp(policy_chosen_logps)) - torch.log1p(-torch.exp(policy_rejected_logps))
        )
        sig_ratio = F.sigmoid(log_odds)
        ratio = torch.log(sig_ratio)
        losses = self.beta * ratio

        chosen_rewards = self.beta * (policy_chosen_logps.to(self.accelerator.device)).detach()
        rejected_rewards = self.beta * (policy_rejected_logps.to(self.accelerator.device)).detach()

        return losses, chosen_rewards, rejected_rewards, torch.mean(ratio).item(), torch.mean(log_odds).item()

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch=batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = len(batch["real_labels"])

        model_kwargs = (
            {
                "decoder_input_ids": self._shift_right(concatenated_batch["concatenated_labels"]),
            }
            if self.is_encoder_decoder
            else {}
        )

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        if self.is_encoder_decoder:
            labels = concatenated_batch["concatenated_labels"].clone()
        else:
            labels = concatenated_batch["concatenated_input_ids"].clone()

        chosen_nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_nll_loss)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the ORPO loss and other metrics for the given batch of inputs for train or test."""
        
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = self.concatenated_forward(model, batch)

        losses, chosen_rewards, rejected_rewards, log_odds_ratio, log_odds_chosen = self.odds_ratio_loss(
            policy_chosen_logps, policy_rejected_logps
        )
        # full ORPO loss
        loss = policy_nll_loss - losses.mean()

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()
        metrics[f"{prefix}log_odds_ratio"] = log_odds_ratio
        metrics[f"{prefix}log_odds_chosen"] = log_odds_chosen

        return loss, metrics
    
    def generate_responses(self, model, batch):

        real_input_ids = batch['real_input_ids']
        real_attention_mask = batch['real_attention_mask']
        real_labels = batch['real_labels']
        input_lengths = batch['chosen_input_length']
        prompt_lengths = batch['prompt_length']
        truncate_ats = batch['truncate_at']
        actual_token_generate_lengths = batch['actual_token_generate_length']

        truncated_input_ids = [ids[:prompt_length+truncate_at] for ids, prompt_length,truncate_at in zip(real_input_ids, prompt_lengths, truncate_ats)]
        truncated_attention_masks = [mask[:prompt_length+truncate_at] for mask, prompt_length,truncate_at in zip(real_attention_mask, prompt_lengths, truncate_ats)]

        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        padded_inputs = self.tokenizer.pad(
            {
                "input_ids": truncated_input_ids,
                "attention_mask": truncated_attention_masks
            },
            padding=True,
            max_length=None, 
            pad_to_multiple_of=None,  
            return_tensors="pt"
        ).to(self.accelerator.device)  

        max_output_length = padded_inputs['input_ids'].shape[1] + max(actual_token_generate_lengths)

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            generated_outputs = unwrapped_model.generate(
                input_ids=padded_inputs['input_ids'],
                attention_mask=padded_inputs['attention_mask'],
                max_length=max_output_length,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=self.responses_per_prompt,
                do_sample=True,
                top_k=50, 
                top_p=0.95,
                temperature=0.7, 
            )

        split_generated_outputs = generated_outputs.view(-1, self.responses_per_prompt, generated_outputs.size(-1))

        experiences = []

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        for i in range(len(real_input_ids)):
            original_input_ids = real_input_ids[i]
            original_attention_mask = real_attention_mask[i]
            original_labels = real_labels[i]

            for j in range(self.responses_per_prompt):
                generated_response = split_generated_outputs[i, j]
                original_part_length = len(truncated_input_ids[i])
                gen_length = actual_token_generate_lengths[i]
                first_non_pad = (generated_response != pad_token_id).nonzero(as_tuple=True)[0]
                start_index = first_non_pad.min().item() if len(first_non_pad) > 0 else 0
                gen_part_start_index = start_index + original_part_length

                gen_part = generated_response[gen_part_start_index:gen_part_start_index + gen_length]
                if eos_token_id in gen_part:
                    eos_index = (gen_part == eos_token_id).nonzero(as_tuple=False).min().item()
                    gen_part = gen_part[:eos_index + 1]

                    full_length_input_ids = torch.cat([original_input_ids[:original_part_length], gen_part])
                    full_length_attention_mask = torch.cat([
                        original_attention_mask[:original_part_length],
                        torch.ones(len(gen_part), dtype=torch.long, device=self.accelerator.device),
                    ])
                else:
                    full_length_input_ids = torch.cat([original_input_ids[:original_part_length], gen_part, original_input_ids[original_part_length + gen_length:]])
                    full_length_attention_mask = torch.cat([
                        original_attention_mask[:original_part_length],
                        torch.ones(len(gen_part), dtype=torch.long, device=self.accelerator.device),
                        original_attention_mask[original_part_length + gen_length:]
                    ])

                labels = full_length_input_ids.clone()
                labels[:prompt_lengths[i]] = torch.full((prompt_lengths[i],), self.label_pad_token_id, dtype=torch.long)

                experience = {
                    'generated_input_ids': full_length_input_ids,
                    'generated_attention_mask': full_length_attention_mask,
                    'generated_labels': labels,
                    'real_input_ids': original_input_ids,
                    'real_attention_mask': original_attention_mask,
                    'real_labels': original_labels
                }
                experiences.append(experience)

        self.tokenizer.padding_side = padding_side_default
        return experiences

    def populate_replay_buffer(self, model, inputs):
        experiences = self.generate_responses(model, inputs)
        for experience in experiences:
            self.replay_buffer.add(experience)


    def sample_from_replay_buffer(self):
        return self.replay_buffer.sample(self.batch_size)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        self.moving_average(self.model, self.ema_model, self.args.gradient_accumulation_steps, self.ema_beta)

        if self.state.global_step % self.sample_every_n_steps == 0:
            self.populate_replay_buffer(self.ema_model, inputs)

        sampled_inputs = self.sample_from_replay_buffer()
        loss, metrics = self.get_batch_loss_metrics(model, sampled_inputs, train_eval="train")

         # Check if it's time to save the model and replay buffer
        if self.state.global_step % self.args.save_steps == 0:
            self.save_ema_replay_buffer()

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def moving_average(self, model, model_ema, gradient_accumulation_steps, beta=0.992):
        self.time_steps["ema"] += 1
        if self.is_deepspeed_enabled:
            deepspeed_plugin = self.accelerator.state.deepspeed_plugin
            config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if (self.time_steps["ema"] - 1) % gradient_accumulation_steps == 0:
            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    if self.is_deepspeed_enabled:
                        if param.requires_grad:
                            if config_kwargs["zero_optimization"]["stage"] != 3:
                                data = param.data.to(param_ema.device)
                                param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)
                            else:
                                # TODO: use prefiltering for efficiency
                                params_to_fetch = _z3_params_to_fetch([param, param_ema])
                                with deepspeed.zero.GatheredParameters(params_to_fetch, modifier_rank=0, enabled=len(params_to_fetch) > 0):
                                    data = param.data.to(param_ema.device)
                                    param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)
                    else:
                        data = param.data.to(param_ema.device)
                        param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)


    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        return policy_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy"],
                        rows=[
                            [prompt, pol[len(prompt) :]]
                            for prompt, pol in zip(random_batch["prompt"], policy_output_decoded)
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    def _shift_right(self, input_ids):
        if self.decoder_start_token_id is None:
            raise ValueError(
                "model.config.decoder_start_token_id has to be defined. It is usually set to the pad_token_id."
            )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), self.decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = self.decoder_start_token_id

        if self.pad_token_id is None:
            raise ValueError("model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, self.pad_token_id)

        return shifted_input_ids
    
    def save_ema(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.ema_model, supported_classes):
            if state_dict is None:
                state_dict = self.ema_model.state_dict()

            if isinstance(unwrap_model(self.ema_model), supported_classes):
                unwrap_model(self.ema_model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.ema_model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


    def init_hf_repo_ema(self, model_id):
        """
        Initializes a git repo in `self.args.hub_model_id`.
        """
        # Only on process zero
        if not self.is_world_process_zero():
            return

        repo_url = create_repo(model_id, token=self.args.hub_token, private=self.args.hub_private_repo, exist_ok=True)
        self.push_in_progress = None

    def push_to_hub_ema(self, model_id, output_dir, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:

        ema_model_id = model_id
        model_name = ema_model_id.split("/")[-1]

        self.init_hf_repo_ema(ema_model_id)

        # Only push from one node.
        if not self.is_world_process_zero():
            return

        # Add additional tags in the case the model has already some tags and users pass
        # "tags" argument to `push_to_hub` so that trainer automatically handles internal tags
        # from all models since Trainer does not call `model.push_to_hub`.
        if "tags" in kwargs and getattr(self.model, "model_tags", None) is not None:
            # If it is a string, convert it to a list
            if isinstance(kwargs["tags"], str):
                kwargs["tags"] = [kwargs["tags"]]

            for model_tag in self.model.model_tags:
                if model_tag not in kwargs["tags"]:
                    kwargs["tags"].append(model_tag)

        self.create_model_card(model_name=model_name, **kwargs)

        # Wait for the current upload to be finished.
        self._finish_current_push()
        return upload_folder(
            repo_id=ema_model_id,
            folder_path=output_dir,
            commit_message=commit_message,
            token=self.args.hub_token,
            run_as_future=not blocking,
            ignore_patterns=["_*", f"checkpoint-*"],
        )

    def save_ema_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.
        Will only save from the main process.
        """

        if self.is_fsdp_enabled:
            if ("FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type)) and (
                version.parse(accelerate_version) > version.parse("0.24.1")
            ):
                state_dict = self.accelerator.get_state_dict(self.ema_model)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
        elif self.is_deepspeed_enabled:
            print('deepspeed save ckpt')
            try:
                state_dict = self.accelerator.get_state_dict(self.ema_model)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                print(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.ema_model.save_checkpoint(output_dir)

        elif self.args.should_save:
            self.save_ema(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub_ema(model_id=f'{self.args.hub_model_id}_ema', output_dir=output_dir, commit_message="Model save")

    def save_ema_replay_buffer(self):
        # Save the replay buffer state
        buffer_directory = os.path.join(self.args.output_dir, "checkpoint-replaybuffer", f"checkpoint-{self.state.global_step}")
        rank = self.accelerator.state.process_index
        os.makedirs(buffer_directory, exist_ok=True)
        buffer_file = f"{buffer_directory}/replay_buffer_{rank}.pt"
        self.replay_buffer.save(buffer_file)
        print(f"Replay buffer saved on process {rank} at step {self.state.global_step}")


        self.save_ema_model(output_dir=os.path.join(self.args.output_dir, f"checkpoint-ema/checkpoint-{self.state.global_step}"), _internal_call=True)
        print(f"EMA model saved on main process at the end of epoch {self.state.epoch}")

        # Ensure all processes have finished saving
        self.accelerator.wait_for_everyone()