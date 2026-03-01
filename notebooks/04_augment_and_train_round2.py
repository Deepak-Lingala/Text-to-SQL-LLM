"""
DIESEL — Notebook 04: TGDA Augmentation + Round 2 Training (Novel)
=====================================================================
Applies Taxonomy-Guided Data Augmentation based on error analysis,
then runs Round 2 fine-tuning on the augmented dataset.

Usage (Colab):
    !python notebooks/04_augment_and_train_round2.py \
        --error_analysis outputs/error_analysis/error_analysis_round1_finetuned.json \
        --round1_adapter outputs/round1/final_adapter

Prerequisites:
    - Round 1 training completed (02_train_round1.py)
    - Error analysis completed (03_error_analysis.py)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import gc

from src.config import get_config
from src.data_loader import load_spider_dataset, prepare_training_data
from src.augmentor import run_tgda
from src.train import train_round2

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="DIESEL Round 2: TGDA + Training")
parser.add_argument("--error_analysis", type=str,
                    default="outputs/error_analysis/error_analysis_round1_finetuned.json",
                    help="Path to Round 1 error analysis JSON")
parser.add_argument("--round1_adapter", type=str,
                    default="outputs/round1/final_adapter",
                    help="Path to Round 1 adapter")
parser.add_argument("--dry_run", action="store_true",
                    help="Dry run with limited steps")
args = parser.parse_args()

config = get_config()

if args.dry_run:
    from dataclasses import replace
    config = replace(config, training=replace(
        config.training,
        max_steps=10,
        logging_steps=1,
        eval_steps=10,
        save_steps=10,
        report_to="none",
    ))
    print("*** DRY RUN MODE ***")

# ═══════════════════════════════════════════════════════════
# Load Resources
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("DIESEL — Round 2: Taxonomy-Guided Data Augmentation")
print("=" * 60)

# Load error analysis
print("\n1. Loading error analysis results...")
with open(args.error_analysis, "r") as f:
    error_analysis = json.load(f)

print(f"   Model: {error_analysis.get('model_name', 'unknown')}")
print(f"   Accuracy: {error_analysis.get('overall_accuracy', 0):.1%}")
print(f"   Errors classified: {error_analysis.get('total_incorrect', 0)}")

# Load dataset
print("\n2. Loading Spider dataset...")
dataset, schema_manager = load_spider_dataset(config)

# Prepare original training data
print("\n3. Preparing original training data...")
train_data = prepare_training_data(dataset, schema_manager, config, split="train")
eval_data = prepare_training_data(dataset, schema_manager, config, split="validation")

# ═══════════════════════════════════════════════════════════
# Run TGDA
# ═══════════════════════════════════════════════════════════
print("\n4. Running Taxonomy-Guided Data Augmentation...")

# Convert train_data to list for TGDA
train_list = [train_data[i] for i in range(len(train_data))]

augmented_dataset = run_tgda(
    error_analysis=error_analysis,
    original_train_data=train_list,
    eval_predictions=error_analysis.get("classified_errors", []),
    schema_manager=schema_manager,
    config=config,
)

print(f"\n   Augmented dataset: {len(augmented_dataset)} examples")

# ═══════════════════════════════════════════════════════════
# Round 2 Training
# ═══════════════════════════════════════════════════════════
print("\n5. Starting Round 2 fine-tuning on augmented data...")

trainer, result = train_round2(
    augmented_dataset=augmented_dataset,
    eval_dataset=eval_data,
    config=config,
    round1_adapter_path=args.round1_adapter,
)

# ═══════════════════════════════════════════════════════════
# Cleanup & Summary
# ═══════════════════════════════════════════════════════════
del trainer
gc.collect()
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("Round 2 Training Complete!")
print("=" * 60)
print(f"  Adapter saved to: {os.path.join(config.paths.round2_dir, 'final_adapter')}")
print(f"  Loss: {result.metrics.get('train_loss', 'N/A')}")
print(f"  Runtime: {result.metrics.get('train_runtime', 0):.0f}s")
print(f"\nNext step: Run 05_final_evaluation.py for full comparison")
