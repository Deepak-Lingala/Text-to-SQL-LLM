"""
DIESEL — Notebook 02: Round 1 Training
========================================
QLoRA fine-tuning of Llama-3.1-8B-Instruct on Spider dataset.
Colab-ready with environment setup.

Usage (Colab):
    1. Runtime → Change runtime type → L4 GPU
    2. Run all cells below

Environment Setup (run in first Colab cell):
    !pip install -q torch transformers peft trl bitsandbytes
    !pip install -q accelerate datasets sqlparse sqlglot
    !pip install -q wandb matplotlib seaborn scipy scikit-learn
    !pip install -q python-dotenv tqdm

    # Login to HuggingFace (for gated model access)
    from huggingface_hub import login
    login(token="YOUR_HF_TOKEN")

    # Optional: WandB
    import wandb
    wandb.login(key="YOUR_WANDB_KEY")
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gc

from src.config import get_config, TrainingConfig
from src.data_loader import load_spider_dataset, prepare_training_data
from src.train import train

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

# Set to small number for dry-run testing, -1 for full training
DRY_RUN_STEPS = -1  # Change to e.g. 10 for smoke test

config = get_config()

# Override for dry-run
if DRY_RUN_STEPS > 0:
    from dataclasses import replace
    config = replace(config, training=replace(
        config.training,
        max_steps=DRY_RUN_STEPS,
        logging_steps=1,
        eval_steps=DRY_RUN_STEPS,
        save_steps=DRY_RUN_STEPS,
        report_to="none",
    ))
    print(f"*** DRY RUN MODE: {DRY_RUN_STEPS} steps ***")

# ═══════════════════════════════════════════════════════════
# GPU Check
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("DIESEL — Round 1 Training")
print("=" * 60)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print("WARNING: No GPU detected! Training will be extremely slow.")

# ═══════════════════════════════════════════════════════════
# Load & Prepare Data
# ═══════════════════════════════════════════════════════════
print("\n1. Loading Spider dataset...")
dataset, schema_manager = load_spider_dataset(config)

print("\n2. Formatting training prompts...")
train_data = prepare_training_data(dataset, schema_manager, config, split="train")
eval_data = prepare_training_data(dataset, schema_manager, config, split="validation")

print(f"\n   Train: {len(train_data)} formatted examples")
print(f"   Eval:  {len(eval_data)} formatted examples")

# Show sample
print("\n   Sample prompt (truncated):")
print(train_data[0]["text"][:300] + "...")

# ═══════════════════════════════════════════════════════════
# Train
# ═══════════════════════════════════════════════════════════
print("\n3. Starting Round 1 fine-tuning...")
print(f"   Output: {config.paths.round1_dir}")

trainer, result = train(
    train_dataset=train_data,
    eval_dataset=eval_data,
    config=config,
    output_dir=config.paths.round1_dir,
    run_name="diesel_round1",
)

# ═══════════════════════════════════════════════════════════
# Memory Cleanup
# ═══════════════════════════════════════════════════════════
print("\n4. Cleaning up GPU memory...")
del trainer
gc.collect()
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════
# Training Summary
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Round 1 Training Complete!")
print("=" * 60)
print(f"  Adapter saved to: {os.path.join(config.paths.round1_dir, 'final_adapter')}")
print(f"  Loss: {result.metrics.get('train_loss', 'N/A')}")
print(f"  Runtime: {result.metrics.get('train_runtime', 0):.0f}s")
print(f"\nNext step: Run 03_error_analysis.py to analyze model errors")
