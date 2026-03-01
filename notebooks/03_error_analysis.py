"""
DIESEL — Notebook 03: Error Analysis (Novel Contribution)
===========================================================
Evaluates the Round 1 fine-tuned model, classifies errors using
the 6-category taxonomy, and generates the Difficulty × Error-Type
matrix — the core empirical contribution of the paper.

Usage (Colab):
    !python notebooks/03_error_analysis.py \
        --adapter outputs/round1/final_adapter \
        --spider_db_dir /path/to/spider/database

Prerequisites:
    - Round 1 training completed (02_train_round1.py)
    - Spider SQLite databases downloaded
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import get_config
from src.data_loader import load_spider_dataset, prepare_training_data
from src.model_loader import load_finetuned_model, load_base_model, load_tokenizer
from src.evaluate import evaluate_model, compare_models
from src.error_analyzer import analyze_errors, compare_error_distributions, ErrorCategory

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="DIESEL Error Analysis")
parser.add_argument("--adapter", type=str, default="outputs/round1/final_adapter",
                    help="Path to Round 1 LoRA adapter")
parser.add_argument("--spider_db_dir", type=str, default="spider_databases",
                    help="Path to Spider SQLite databases")
parser.add_argument("--max_samples", type=int, default=None,
                    help="Max eval samples (None=all)")
parser.add_argument("--eval_base", action="store_true", default=True,
                    help="Also evaluate base model for comparison")
args = parser.parse_args()

config = get_config()
sns.set_theme(style="whitegrid", font_scale=1.1)
FIGURES_DIR = config.paths.figures_dir
os.makedirs(FIGURES_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# Load Data
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("DIESEL — Error Analysis Pipeline")
print("=" * 60)

dataset, schema_manager = load_spider_dataset(config)
eval_data = dataset["validation"]

if args.max_samples:
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    print(f"Using {len(eval_data)} evaluation examples (subset)")

# ═══════════════════════════════════════════════════════════
# Step 1: Evaluate Base Model (Zero-Shot)
# ═══════════════════════════════════════════════════════════
all_results = []

if args.eval_base:
    print("\n" + "=" * 60)
    print("Step 1: Evaluating BASE model (zero-shot)")
    print("=" * 60)
    
    base_model = load_base_model(config)
    base_tokenizer = load_tokenizer(config)
    base_model.eval()
    
    base_results = evaluate_model(
        model=base_model,
        tokenizer=base_tokenizer,
        eval_data=eval_data,
        schema_manager=schema_manager,
        spider_db_dir=args.spider_db_dir,
        config=config,
        model_name="base_zero_shot",
        save_dir=config.paths.eval_dir,
    )
    all_results.append(base_results)
    
    # Cleanup
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════
# Step 2: Evaluate Fine-Tuned Model (Round 1)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 2: Evaluating ROUND 1 fine-tuned model")
print("=" * 60)

ft_model, ft_tokenizer = load_finetuned_model(args.adapter, config)

ft_results = evaluate_model(
    model=ft_model,
    tokenizer=ft_tokenizer,
    eval_data=eval_data,
    schema_manager=schema_manager,
    spider_db_dir=args.spider_db_dir,
    config=config,
    model_name="round1_finetuned",
    save_dir=config.paths.eval_dir,
)
all_results.append(ft_results)

# Cleanup
del ft_model
gc.collect()
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════
# Step 3: Statistical Comparison
# ═══════════════════════════════════════════════════════════
if len(all_results) > 1:
    print("\n" + "=" * 60)
    print("Step 3: Statistical Comparison (McNemar's Test)")
    print("=" * 60)
    
    comparison = compare_models(all_results, save_dir=config.paths.eval_dir)

# ═══════════════════════════════════════════════════════════
# Step 4: Error Taxonomy Analysis (NOVEL)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 4: Error Taxonomy Analysis")
print("=" * 60)

# Analyze Round 1 errors
ft_error_analysis = analyze_errors(
    eval_results=ft_results,
    schema_map=schema_manager.schema_map,
    config=config,
)

# Optionally analyze base model errors for comparison
if args.eval_base:
    base_error_analysis = analyze_errors(
        eval_results=base_results,
        schema_map=schema_manager.schema_map,
        config=config,
    )
    
    # Compare error distributions
    shift_analysis = compare_error_distributions(
        base_error_analysis, ft_error_analysis,
        save_dir=config.paths.error_analysis_dir,
    )

# ═══════════════════════════════════════════════════════════
# Step 5: Generate Publication Figures
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 5: Generating Publication Figures")
print("=" * 60)

# Figure 4: Error Category Distribution (Bar Chart)
print("Generating Figure 4: Error Category Distribution...")
fig, ax = plt.subplots(figsize=(10, 6))

cats = ErrorCategory.ALL
labels = [ErrorCategory.LABELS[c] for c in cats]
pcts = [ft_error_analysis["category_distribution"][c]["percentage_of_errors"] for c in cats]

colors = sns.color_palette("Set2", len(cats))
bars = ax.bar(labels, pcts, color=colors, edgecolor='gray', linewidth=0.5)
ax.set_ylabel("Percentage of Errors (%)")
ax.set_title("Error Category Distribution — Round 1 Fine-Tuned Model")
ax.set_xticklabels(labels, rotation=30, ha='right')

for bar, pct in zip(bars, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig4_error_distribution.png"), bbox_inches='tight')
plt.close()

# Figure 5: Difficulty × Error-Type Heatmap (THE KEY FIGURE)
print("Generating Figure 5: Difficulty × Error-Type Heatmap...")
matrix_data = ft_error_analysis.get("difficulty_error_matrix", {})

if matrix_data:
    difficulties = sorted(matrix_data.keys())
    matrix = np.zeros((len(difficulties), len(cats)))
    
    for i, diff in enumerate(difficulties):
        for j, cat in enumerate(cats):
            matrix[i, j] = matrix_data[diff].get(cat, 0)
    
    # Normalize rows to percentages
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_pct = matrix / row_sums * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=[ErrorCategory.LABELS[c] for c in cats],
                yticklabels=difficulties, ax=axes[0])
    axes[0].set_title("Error Counts by Difficulty × Category")
    axes[0].set_xlabel("Error Category")
    axes[0].set_ylabel("Difficulty Level")
    
    # Percentages
    sns.heatmap(matrix_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=[ErrorCategory.LABELS[c] for c in cats],
                yticklabels=difficulties, ax=axes[1])
    axes[1].set_title("Error Distribution (%) by Difficulty × Category")
    axes[1].set_xlabel("Error Category")
    axes[1].set_ylabel("Difficulty Level")
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_difficulty_error_heatmap.png"), bbox_inches='tight')
    plt.close()

# Figure 6: Base vs Fine-Tuned Error Comparison
if args.eval_base:
    print("Generating Figure 6: Error Distribution Shift...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(cats))
    width = 0.35
    
    base_pcts = [base_error_analysis["category_distribution"][c]["percentage_of_errors"]
                 for c in cats]
    ft_pcts = [ft_error_analysis["category_distribution"][c]["percentage_of_errors"]
               for c in cats]
    
    ax.bar(x - width/2, base_pcts, width, label='Base (Zero-Shot)', 
           color='#FF7043', alpha=0.85)
    ax.bar(x + width/2, ft_pcts, width, label='Round 1 (Fine-Tuned)', 
           color='#42A5F5', alpha=0.85)
    
    ax.set_ylabel("Percentage of Errors (%)")
    ax.set_title("Error Distribution Shift: Base → Fine-Tuned")
    ax.set_xticks(x)
    ax.set_xticklabels([ErrorCategory.LABELS[c] for c in cats], rotation=30, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_error_shift.png"), bbox_inches='tight')
    plt.close()

# Figure 7: Difficulty-Stratified Accuracy
print("Generating Figure 7: Difficulty-Stratified Accuracy...")
fig, ax = plt.subplots(figsize=(10, 6))

if all_results:
    for res in all_results:
        diffs = sorted(res.get("difficulty_breakdown", {}).keys())
        accs = [res["difficulty_breakdown"][d]["accuracy"] * 100 for d in diffs]
        ax.plot(diffs, accs, 'o-', label=res["model_name"], linewidth=2, markersize=8)
    
    ax.set_xlabel("Query Difficulty")
    ax.set_ylabel("Execution Accuracy (%)")
    ax.set_title("Execution Accuracy by Query Difficulty")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig7_difficulty_accuracy.png"), bbox_inches='tight')
plt.close()

print(f"\n{'Analysis Complete':=^60}")
print(f"  Figures saved to: {FIGURES_DIR}")
print(f"  Error analysis saved to: {config.paths.error_analysis_dir}")
print(f"\nTop-3 weaknesses identified for TGDA:")
for i, w in enumerate(ft_error_analysis["ranked_weaknesses"][:3], 1):
    print(f"  {i}. {w['label']}: {w['error_rate']:.1f}% of errors")
print(f"\nNext step: Run 04_augment_and_train_round2.py")
