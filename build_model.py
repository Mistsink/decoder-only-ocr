from typing import Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftConfig,
)

from config.config import Config
from model.xglm import XGLMPatchForCausalLM


def build_model_peft(
    cfg: Config, inference: bool = False
) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Use LoRA
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, cache_dir=cfg.cache_dir)
    num_new_tokens = tokenizer.add_tokens(["\n", " "])

    custom_layers = [
        "patch_embeddings",
    ]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # load_in_8bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        # bnb_4bit_compute_dtype=torch.float32,
        # llm_int8_threshold=6.0,
        # llm_int8_has_fp16_weight=False,
        llm_int8_skip_modules=custom_layers,
    )

    model = XGLMPatchForCausalLM.from_pretrained(
        cfg.model,
        patch_config=cfg,
        cache_dir=cfg.cache_dir,
        quantization_config=bnb_config,
    )

    # 调整模型的embedding层大小
    if num_new_tokens > 0:
        model.resize_token_embeddings(model.config.vocab_size + num_new_tokens)
    model.config.sep_token_id = tokenizer.sep_token_id

    model = prepare_model_for_kbit_training(model)

    if not inference:
        lora_modules = "all-linear"
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=lora_modules,
            modules_to_save=custom_layers,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        peft_config = PeftConfig.from_pretrained(cfg.cache_dir)
        model = PeftModel.from_pretrained(model, cfg.cache_dir, config=peft_config)

    return model, tokenizer


def build_model_vanilla(cfg: Config) -> Tuple[nn.Module, AutoTokenizer]:
    """
    No peft
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, cache_dir=cfg.cache_dir)
    num_new_tokens = tokenizer.add_tokens(["\n", " "])

    model = XGLMPatchForCausalLM.from_pretrained(
        cfg.model, patch_config=cfg, cache_dir=cfg.cache_dir
    )

    # 调整模型的embedding层大小
    if num_new_tokens > 0:
        model.resize_token_embeddings(model.config.vocab_size + num_new_tokens)
    model.config.sep_token_id = tokenizer.sep_token_id

    return model, tokenizer
