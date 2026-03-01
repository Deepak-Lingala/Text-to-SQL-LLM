"""
DIESEL — Training Module
==========================
SFTTrainer-based fine-tuning with WandB logging,
gradient checkpointing, and LoRA adapter saving.
"""

import os
from typing import Optional, Tuple
from datetime import datetime

import torch
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from .config import DieselConfig, get_default_config
from .model_loader import setup_for_training, load_tokenizer


def get_training_arguments(
    config: DieselConfig,
    output_dir: str,
    run_name: Optional[str] = None,
) -> SFTConfig:
    """
    Build TrainingArguments from config.
    
    Args:
        config: Project config
        output_dir: Where to save checkpoints
        run_name: WandB run name
    """
    tc = config.training
    
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"diesel_r{config.lora.r}_lr{tc.learning_rate}_{timestamp}"
    
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=tc.num_train_epochs,
        max_steps=tc.max_steps,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        gradient_checkpointing=tc.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        warmup_ratio=tc.warmup_ratio,
        lr_scheduler_type=tc.lr_scheduler_type,
        fp16=tc.fp16,
        bf16=tc.bf16,
        optim=tc.optim,
        logging_steps=tc.logging_steps,
        eval_strategy=tc.eval_strategy,
        eval_steps=tc.eval_steps,
        save_strategy=tc.save_strategy,
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit,
        load_best_model_at_end=tc.load_best_model_at_end,
        metric_for_best_model=tc.metric_for_best_model,
        report_to=tc.report_to,
        run_name=run_name,
        seed=tc.seed,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        group_by_length=True,
        # SFT-specific (moved from SFTTrainer params in trl v0.12+)
        max_seq_length=config.training.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )


def train(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[DieselConfig] = None,
    output_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> Tuple:
    """
    Run supervised fine-tuning with SFTTrainer.
    
    Args:
        train_dataset: Training dataset with 'text' column
        eval_dataset: Validation dataset with 'text' column
        config: Project config
        output_dir: Checkpoint output directory
        run_name: WandB run name
        resume_from_checkpoint: Path to resume from
        
    Returns:
        Tuple of (trainer, train_result)
    """
    if config is None:
        config = get_default_config()
    
    if output_dir is None:
        output_dir = config.paths.round1_dir
    
    # Setup model + tokenizer
    print("=" * 60)
    print("DIESEL Training Pipeline")
    print("=" * 60)
    
    model, tokenizer = setup_for_training(config)
    
    # Training arguments
    training_args = get_training_arguments(config, output_dir, run_name)
    
    # SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Print training summary
    print(f"\n{'Training Configuration':=^60}")
    print(f"  Model: {config.model.model_name}")
    print(f"  LoRA rank: {config.lora.r}, alpha: {config.lora.lora_alpha}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.num_train_epochs}")
    print(f"  Effective batch size: "
          f"{config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")
    print(f"  Max seq length: {config.training.max_seq_length}")
    print(f"  Train examples: {len(train_dataset)}")
    if eval_dataset:
        print(f"  Eval examples: {len(eval_dataset)}")
    print(f"  Output dir: {output_dir}")
    print("=" * 60)
    
    # Train
    train_result = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )
    
    # Save final adapter
    final_adapter_path = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    print(f"\n{'Training Complete':=^60}")
    print(f"  Final adapter saved to: {final_adapter_path}")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Train runtime: {metrics.get('train_runtime', 0):.1f}s")
    
    return trainer, train_result


def train_round2(
    augmented_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[DieselConfig] = None,
    round1_adapter_path: Optional[str] = None,
) -> Tuple:
    """
    Round 2 fine-tuning with TGDA augmented data.
    Starts from Round 1 checkpoint and continues training
    on augmented data targeting weak error categories.
    
    Args:
        augmented_dataset: TGDA-augmented training data
        eval_dataset: Validation data for evaluation
        config: Project config
        round1_adapter_path: Path to Round 1 adapter (for reference)
        
    Returns:
        Tuple of (trainer, train_result)
    """
    if config is None:
        config = get_default_config()
    
    # Use a lower learning rate for Round 2 (continued training)
    from dataclasses import replace
    round2_training = replace(
        config.training,
        learning_rate=config.training.learning_rate * 0.5,  # halve LR
        num_train_epochs=2,  # fewer epochs for targeted refinement
    )
    round2_config = replace(config, training=round2_training)
    
    print("=" * 60)
    print("DIESEL Round 2: Taxonomy-Guided Fine-Tuning")
    print("=" * 60)
    print(f"  Augmented training examples: {len(augmented_dataset)}")
    print(f"  Learning rate: {round2_training.learning_rate}")
    print(f"  Epochs: {round2_training.num_train_epochs}")
    
    return train(
        train_dataset=augmented_dataset,
        eval_dataset=eval_dataset,
        config=round2_config,
        output_dir=config.paths.round2_dir,
        run_name="diesel_round2_tgda",
    )
