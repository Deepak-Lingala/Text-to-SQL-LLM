"""
DIESEL — Model Loader
=======================
QLoRA model setup with 4-bit NF4 quantization 
and LoRA adapter configuration for Llama-3.1-8B-Instruct.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)
from typing import Optional, Tuple

from .config import DieselConfig, get_default_config


def get_quantization_config(config: DieselConfig) -> BitsAndBytesConfig:
    """
    Build BitsAndBytes 4-bit quantization config.
    """
    compute_dtype = getattr(torch, config.model.bnb_4bit_compute_dtype)
    
    return BitsAndBytesConfig(
        load_in_4bit=config.model.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.model.bnb_4bit_use_double_quant,
    )


def get_lora_config(config: DieselConfig) -> LoraConfig:
    """
    Build LoRA adapter configuration.
    """
    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.lora.target_modules,
    )


def load_tokenizer(
    config: Optional[DieselConfig] = None,
    model_name: Optional[str] = None
) -> AutoTokenizer:
    """
    Load and configure tokenizer for Llama-3.1-8B-Instruct.
    
    Sets:
        - pad_token = eos_token (Llama doesn't have a pad token)
        - padding_side = 'right' for training, 'left' for generation
    """
    if config is None:
        config = get_default_config()
    
    name = model_name or config.model.model_name
    
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
    )
    
    # Llama 3.1 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Right padding for training (causal LM)
    tokenizer.padding_side = "right"
    
    return tokenizer


def load_base_model(
    config: Optional[DieselConfig] = None,
    model_name: Optional[str] = None,
    device_map: str = "auto"
) -> AutoModelForCausalLM:
    """
    Load the base model with 4-bit quantization.
    
    Args:
        config: Project config
        model_name: Override model name
        device_map: Device mapping strategy
        
    Returns:
        Quantized base model
    """
    if config is None:
        config = get_default_config()
    
    name = model_name or config.model.model_name
    bnb_config = get_quantization_config(config)
    
    print(f"Loading base model: {name}")
    print(f"  Quantization: 4-bit {config.model.bnb_4bit_quant_type}")
    print(f"  Compute dtype: {config.model.bnb_4bit_compute_dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=getattr(torch, config.model.torch_dtype),
    )
    
    # Disable caching for training (incompatible with gradient checkpointing)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    print(f"  Model loaded on: {model.device}")
    print(f"  Trainable parameters before LoRA: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def prepare_model_for_training(
    model: AutoModelForCausalLM,
    config: Optional[DieselConfig] = None,
) -> AutoModelForCausalLM:
    """
    Apply LoRA adapters and prepare for QLoRA training.
    
    Args:
        model: Base quantized model
        config: Project config
        
    Returns:
        PEFT model with LoRA adapters
    """
    if config is None:
        config = get_default_config()
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.training.gradient_checkpointing,
    )
    
    # Apply LoRA
    lora_config = get_lora_config(config)
    model = get_peft_model(model, lora_config)
    
    # Print trainable stats
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    
    print(f"\n  LoRA Configuration:")
    print(f"    Rank (r): {config.lora.r}")
    print(f"    Alpha: {config.lora.lora_alpha}")
    print(f"    Target modules: {config.lora.target_modules}")
    print(f"    Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    
    return model


def load_finetuned_model(
    adapter_path: str,
    config: Optional[DieselConfig] = None,
    device_map: str = "auto",
    merge: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a fine-tuned model (base + LoRA adapter).
    
    Args:
        adapter_path: Path to saved LoRA adapter weights
        config: Project config
        device_map: Device mapping
        merge: Whether to merge adapter into base model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if config is None:
        config = get_default_config()
    
    print(f"Loading fine-tuned model from: {adapter_path}")
    
    # Load base model
    base_model = load_base_model(config, device_map=device_map)
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map=device_map,
    )
    
    if merge:
        print("  Merging adapter into base model...")
        model = model.merge_and_unload()
    
    # Load tokenizer
    tokenizer = load_tokenizer(config)
    
    # Set to eval mode
    model.eval()
    
    return model, tokenizer


def setup_for_training(
    config: Optional[DieselConfig] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    One-call setup: load base model, apply LoRA, load tokenizer.
    Ready for SFTTrainer.
    
    Returns:
        Tuple of (peft_model, tokenizer)
    """
    if config is None:
        config = get_default_config()
    
    tokenizer = load_tokenizer(config)
    base_model = load_base_model(config)
    model = prepare_model_for_training(base_model, config)
    
    return model, tokenizer


def get_generation_config(
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> dict:
    """
    Build generation kwargs for SQL inference.
    Low temperature for deterministic SQL generation.
    """
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
