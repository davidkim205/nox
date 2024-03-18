import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional, Any, Dict


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="davidkim205/komt-solar-10.7b-sft-v5",
        metadata={"help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."}
    )
    adapter_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the adapter weight or identifier from huggingface.co/models."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    resize_vocab: Optional[bool] = field(
        default=False, metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."}
    )
    split_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    quantization_bit: Optional[int] = field(
        default=None, metadata={"help": "The number of bits to quantize the model."}
    )
    quantization_type: Optional[Literal["fp4", "nf4"]] = field(
        default="nf4", metadata={"help": "Quantization data type to use in int4 training."}
    )
    double_quantization: Optional[bool] = field(
        default=True, metadata={"help": "Whether or not to use double quantization in int4 training."}
    )
    shift_attn: Optional[bool] = field(
        default=False, metadata={"help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."}
    )
    use_unsloth: Optional[bool] = field(
        default=False, metadata={"help": "Whether or not to use unsloth's optimization for the LoRA training."}
    )
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "Auth token to log in with Hugging Face Hub."})
    export_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to the directory to save the exported model."}
    )
    export_size: Optional[int] = field(
        default=1, metadata={"help": "The file shard size (in GB) of the exported model."}
    )
    export_quantization_bit: Optional[int] = field(
        default=None, metadata={"help": "The number of bits to quantize the exported model."}
    )
    export_quantization_dataset: Optional[str] = field(
        default=None, metadata={"help": "Path to the dataset or dataset name to use in quantizing the exported model."}
    )
    export_quantization_nsamples: Optional[int] = field(
        default=128, metadata={"help": "The number of samples used for quantization."}
    )
    export_quantization_maxlen: Optional[int] = field(
        default=1024, metadata={"help": "The maximum length of the model inputs used for quantization."}
    )
    export_legacy_format: Optional[bool] = field(
        default=False, metadata={"help": "Whether or not to save the `.bin` files instead of `.safetensors`."}
    )
    export_hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the repository if push the model to the Hugging Face hub."}
    )
    def __post_init__(self):
        self.compute_dtype = None
        self.model_max_length = None

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError("`split_special_tokens` is only supported for slow tokenizers.")

        if self.adapter_name_or_path is not None:  # support merging multiple lora weights
            self.adapter_name_or_path = [path.strip() for path in self.adapter_name_or_path.split(",")]

        assert self.quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.export_quantization_bit in [None, 8, 4, 3, 2], "We only accept 2/3/4/8-bit quantization."

        if self.export_quantization_bit is not None and self.export_quantization_dataset is None:
            raise ValueError("Quantization dataset is necessary for exporting.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    dataset: Optional[str] = field(
        default="/work/kollm/datasets_komt_v4_124k/komt-124k.jsonl",
        metadata={"help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."},
    )
    split: Optional[str] = field(
        default="train", metadata={"help": "Which dataset split to use for training and evaluation."}
    )
    cutoff_len: Optional[int] = field(
        default=1024, metadata={"help": "The maximum length of the model inputs after tokenization."}
    )
    reserved_label_len: Optional[int] = field(
        default=1, metadata={"help": "The maximum length reserved for label after tokenization."}
    )

    mix_strategy: Optional[Literal["concat", "interleave_under", "interleave_over"]] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."},
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"},
    )
    ignore_pad_token_for_loss: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    num_workers: Optional[int] = field(
        default=16, metadata={"help": "The number of processes to use for the preprocessing."}
    )

@dataclass
class FreezeArguments:
    r"""
    Arguments pertaining to the freeze (partial-parameter) training.
    """
    name_module_trainable: Optional[str] = field(
        default="mlp",
        metadata={
            "help": 'Name of trainable modules for partial-parameter (freeze) fine-tuning. \
                  Use commas to separate multiple modules. \
                  LLaMA choices: ["mlp", "self_attn"], \
                  BLOOM & Falcon & ChatGLM choices: ["mlp", "self_attention"], \
                  Qwen choices: ["mlp", "attn"], \
                  Phi choices: ["mlp", "mixer"], \
                  Others choices: the same as LLaMA.'
        },
    )
    num_layer_trainable: Optional[int] = field(
        default=3, metadata={"help": "The number of trainable layers for partial-parameter (freeze) fine-tuning."}
    )


@dataclass
class LoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """
    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name(s) of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."
        },
    )
    lora_alpha: Optional[int] = field(
        default=None, metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."}
    )
    lora_dropout: Optional[float] = field(default=0.0, metadata={"help": "Dropout rate for the LoRA fine-tuning."})
    lora_rank: Optional[int] = field(default=8, metadata={"help": "The intrinsic dimension for LoRA fine-tuning."})
    # lora_bf16_mode: Optional[bool] = field(
    #     default=False, metadata={"help": "Whether or not to train lora adapters in bf16 precision."}
    # )
    create_new_adapter: Optional[bool] = field(
        default=False, metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."}
    )


@dataclass
class RLHFArguments:
    r"""
    Arguments pertaining to the PPO and DPO training.
    """
    dpo_beta: Optional[float] = field(default=0.1, metadata={"help": "The beta parameter for the DPO loss."})
    dpo_loss: Optional[Literal["sigmoid", "hinge", "ipo", "kto"]] = field(
        default="sigmoid", metadata={"help": "The type of DPO loss to use."}
    )
    dpo_ftx: Optional[float] = field(
        default=0, metadata={"help": "The supervised fine-tuning loss coefficient in DPO training."}
    )
    ppo_buffer_size: Optional[int] = field(
        default=1,
        metadata={"help": "The number of mini-batches to make experience buffer in a PPO optimization step."},
    )
    ppo_epochs: Optional[int] = field(
        default=4, metadata={"help": "The number of epochs to perform in a PPO optimization step."}
    )
    ppo_logger: Optional[str] = field(
        default=None, metadata={"help": 'Log with either "wandb" or "tensorboard" in PPO training.'}
    )
    ppo_score_norm: Optional[bool] = field(
        default=False, metadata={"help": "Use score normalization in PPO training."}
    )
    ppo_target: Optional[float] = field(
        default=6.0, metadata={"help": "Target KL value for adaptive KL control in PPO training."}
    )
    ppo_whiten_rewards: Optional[bool] = field(
        default=False, metadata={"help": "Whiten the rewards before compute advantages in PPO training."}
    )
    ref_model: Optional[str] = field(
        default=None, metadata={"help": "Path to the reference model used for the PPO or DPO training."}
    )
    ref_model_adapters: Optional[str] = field(
        default=None, metadata={"help": "Path to the adapters of the reference model."}
    )
    ref_model_quantization_bit: Optional[int] = field(
        default=None, metadata={"help": "The number of bits to quantize the reference model."}
    )
    reward_model: Optional[str] = field(
        default=None, metadata={"help": "Path to the reward model used for the PPO training."}
    )
    reward_model_adapters: Optional[str] = field(
        default=None, metadata={"help": "Path to the adapters of the reward model."}
    )
    reward_model_quantization_bit: Optional[int] = field(
        default=None, metadata={"help": "The number of bits to quantize the reward model."}
    )
    reward_model_type: Optional[Literal["lora", "full", "api"]] = field(
        default="lora",
        metadata={"help": "The type of the reward model in PPO training. Lora model only supports lora training."},
    )


@dataclass
class FinetuningArguments(FreezeArguments, LoraArguments, RLHFArguments):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    stage: Optional[Literal["pt", "sft", "rm", "ppo", "dpo"]] = field(
        default="sft", metadata={"help": "Which stage will be performed in training."}
    )
    finetuning_type: Optional[Literal["lora", "freeze", "full"]] = field(
        default="lora", metadata={"help": "Which fine-tuning method to use."}
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.name_module_trainable = split_arg(self.name_module_trainable)
        self.lora_alpha = self.lora_alpha or self.lora_rank * 2
        self.additional_target = split_arg(self.additional_target)

        assert self.finetuning_type in ["lora", "freeze", "full"], "Invalid fine-tuning method."
        assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."

        if self.stage == "ppo" and self.reward_model is None:
            raise ValueError("Reward model is necessary for PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "lora" and self.finetuning_type != "lora":
            raise ValueError("Freeze/Full PPO training needs `reward_model_type=full`.")

    def save_to_json(self, json_path: str):
        r"""Saves the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        r"""Creates an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()

        return cls(**json.loads(text))
