import os
import inspect
import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from typing import Any, Dict, Optional, Tuple
from transformers import BitsAndBytesConfig, GPTQConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils.versions import require_version
from transformers.utils import is_torch_cuda_available

from arguments import FinetuningArguments, ModelArguments
from logger import GetLogger

logger = GetLogger

def get_current_device():
    if is_torch_cuda_available():
        if os.environ.get("LOCAL_RANK") == None:
            device = torch.cuda.current_device()
        else:
            device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))
    else:
        device = "cpu"
        {'':torch.cuda.current_device()}
    return device
def configure_quantization(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    config_kwargs: Dict[str, Any],
) -> None:
    r"""
    Priority: GPTQ-quantized (training) > AutoGPTQ (export) > Bitsandbytes (training)
    """
    if getattr(config, "quantization_config", None):  # gptq
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        config_kwargs["device_map"] = {"": get_current_device()}
        quantization_config: Dict[str, Any] = getattr(config, "quantization_config", None)
        if quantization_config.get("quant_method", None) == "gptq" and quantization_config.get("bits", -1) == 4:
            quantization_config["use_exllama"] = False  # disable exllama
        logger.info("Loading {}-bit GPTQ-quantized model.".format(quantization_config.get("bits", -1)))

    elif model_args.quantization_bit is not None:  # bnb
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        if model_args.quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        elif model_args.quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type,
            )

        config_kwargs["device_map"] = {"": get_current_device()}
        logger.info("Quantizing model to {} bit.".format(model_args.quantization_bit))

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param
def init_adapter(
    model: "PreTrainedModel", model_args: "ModelArguments", finetuning_args: "FinetuningArguments", is_trainable: bool
) -> "PreTrainedModel":

    if finetuning_args.finetuning_type == "lora":
        logger.info("Fine-tuning method: LoRA")
        adapter_to_resume = None

        if model_args.adapter_name_or_path is not None:
            is_mergeable = True
            if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
                assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
                is_mergeable = False

            if is_deepspeed_zero3_enabled():
                assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
                is_mergeable = False

            if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
                adapter_to_merge = model_args.adapter_name_or_path[:-1]
                adapter_to_resume = model_args.adapter_name_or_path[-1]
            else:
                adapter_to_merge = model_args.adapter_name_or_path

            for adapter in adapter_to_merge:
                model = PeftModel.from_pretrained(model, adapter)
                model = model.merge_and_unload()

            if len(adapter_to_merge) > 0:
                logger.info("Merged {} adapter(s).".format(len(adapter_to_merge)))

            if adapter_to_resume is not None:  # resume lora training
                model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_trainable)

        peft_kwargs = {
            "r": finetuning_args.lora_rank,
            "target_modules": ['down_proj', 'q_proj', 'o_proj', 'v_proj', 'up_proj', 'k_proj', 'gate_proj'],
            "lora_alpha": finetuning_args.lora_alpha,
            "lora_dropout": finetuning_args.lora_dropout,
        }

        if model_args.use_unsloth:
            from unsloth import FastLlamaModel, FastMistralModel  # type: ignore

            unsloth_peft_kwargs = {"model": model, "max_seq_length": model_args.model_max_length}
            if "loftq_config" in inspect.signature(FastLlamaModel.get_peft_model).parameters:
                unsloth_peft_kwargs["loftq_config"] = {}

            if getattr(model.config, "model_type", None) == "llama":
                model = FastLlamaModel.get_peft_model(**peft_kwargs, **unsloth_peft_kwargs)
            elif getattr(model.config, "model_type", None) == "mistral":
                model = FastMistralModel.get_peft_model(**peft_kwargs, **unsloth_peft_kwargs)
            else:
                raise NotImplementedError

        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                modules_to_save=finetuning_args.additional_target,
                **peft_kwargs,
            )

            model = get_peft_model(model, lora_config)
            print('!!!!!!!!!!!!!!1')
            print(model)
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.bfloat16)

    if model_args.adapter_name_or_path is not None:
        logger.info("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    return model


def load_model_and_tokenizer(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: Optional[bool] = False,
) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """

    #try_download_model_from_ms(model_args)

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right",
        **config_kwargs,
    )
    #patch_tokenizer(tokenizer)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    # config_kwargs["attn_implementation"] = "eager"
    print('config_kwargs....1')
    print(config_kwargs)
    configure_quantization(config, tokenizer, model_args, config_kwargs)
    print('config_kwargs....2')
    print(config_kwargs)
    #if model is None:
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=model_args.compute_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        **config_kwargs,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False  # turn off when gradient checkpointing is enabled
    logger.info("Gradient checkpointing enabled.")

    model = init_adapter(model, model_args, finetuning_args, is_trainable)

    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.bfloat16)

    model.train()
    print('!!!!!!!!1')
    print(model)
    trainable_params, all_param = count_parameters(model)
    logger.info(
        "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    )

    return model, tokenizer


def create_ref_model(model_args, finetuning_args, add_valuehead = False):
    r"""
    Creates reference model for PPO/DPO training. Evaluation mode is not supported.

    The valuehead parameter is randomly initialized since it is useless for PPO training.
    """
    if finetuning_args.ref_model is not None:
        ref_model_args_dict = model_args.to_dict()
        ref_model_args_dict.update(
            dict(
                model_name_or_path=finetuning_args.ref_model,
                adapter_name_or_path=finetuning_args.ref_model_adapters,
                quantization_bit=finetuning_args.ref_model_quantization_bit,
            )
        )
        ref_model_args = ModelArguments(**ref_model_args_dict)
        ref_finetuning_args = FinetuningArguments(finetuning_type="lora")
        ref_model, _ = load_model_and_tokenizer(
            ref_model_args, ref_finetuning_args, is_trainable=False, add_valuehead=add_valuehead
        )
        logger.info("Created reference model from {}".format(finetuning_args.ref_model))
    else:
        if finetuning_args.finetuning_type == "lora":
            ref_model = None
        else:
            ref_model, _ = load_model_and_tokenizer(
                model_args, finetuning_args, is_trainable=False, add_valuehead=add_valuehead
            )
            logger.info("Created reference model from the model itself.")

    return ref_model