"""
DIESEL — Notebook 01: Exploratory Data Analysis
=================================================
Run this script to explore the Spider dataset statistics
and generate publication-ready visualizations.

Usage (Colab):
    !pip install -r requirements.txt
    !python notebooks/01_eda.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
from collections import Counter

from src.config import get_config, PathConfig
from src.data_loader import load_spider_dataset, compute_dataset_statistics

# ─── Configuration ─────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'

config = get_config()
FIGURES_DIR = config.paths.figures_dir
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Load Data ─────────────────────────────────────────────
print("=" * 60)
print("DIESEL — Exploratory Data Analysis")
print("=" * 60)

dataset, schema_manager = load_spider_dataset(config)
stats = compute_dataset_statistics(dataset, schema_manager)

print(f"\nDataset Statistics:")
print(f"  Training examples:    {stats['num_train']:,}")
print(f"  Validation examples:  {stats['num_val']:,}")
print(f"  Unique databases:     {stats['num_databases']}")
print(f"  Avg query length:     {np.mean(stats['query_lengths']):.0f} chars")
print(f"  Avg question length:  {np.mean(stats['question_lengths']):.0f} chars")
print(f"  Avg tables per DB:    {np.mean(stats['tables_per_db']):.1f}")
print(f"  Avg columns per DB:   {np.mean(stats['cols_per_db']):.1f}")

# ─── Figure 1: SQL Keyword Distribution ───────────────────
print("\nGenerating Figure 1: SQL Keyword Distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
keywords = sorted(stats["query_keywords"].items(), key=lambda x: x[1], reverse=True)
kw_names = [k[0] for k in keywords]
kw_counts = [k[1] for k in keywords]

colors = sns.color_palette("viridis", len(kw_names))
bars = ax.barh(kw_names[::-1], kw_counts[::-1], color=colors[::-1])
ax.set_xlabel("Frequency")
ax.set_title("SQL Keyword Distribution in Spider Dataset")
for bar, count in zip(bars, kw_counts[::-1]):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
            f'{count:,}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig1_keyword_distribution.png"), bbox_inches='tight')
plt.close()

# ─── Figure 2: Query Length Distribution ──────────────────
print("Generating Figure 2: Query Length Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(stats["query_lengths"], bins=50, color='#2196F3', alpha=0.8, edgecolor='white')
axes[0].set_xlabel("SQL Query Length (chars)")
axes[0].set_ylabel("Count")
axes[0].set_title("SQL Query Length Distribution")
axes[0].axvline(np.mean(stats["query_lengths"]), color='red', linestyle='--',
                label=f'Mean: {np.mean(stats["query_lengths"]):.0f}')
axes[0].legend()

axes[1].hist(stats["question_lengths"], bins=50, color='#4CAF50', alpha=0.8, edgecolor='white')
axes[1].set_xlabel("Question Length (chars)")
axes[1].set_ylabel("Count")
axes[1].set_title("NL Question Length Distribution")
axes[1].axvline(np.mean(stats["question_lengths"]), color='red', linestyle='--',
                label=f'Mean: {np.mean(stats["question_lengths"]):.0f}')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig2_length_distributions.png"), bbox_inches='tight')
plt.close()

# ─── Figure 3: Database Complexity ────────────────────────
print("Generating Figure 3: Database Complexity...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(stats["tables_per_db"], bins=20, color='#FF9800', alpha=0.8, edgecolor='white')
axes[0].set_xlabel("Number of Tables")
axes[0].set_ylabel("Number of Databases")
axes[0].set_title("Tables per Database")

axes[1].hist(stats["cols_per_db"], bins=30, color='#9C27B0', alpha=0.8, edgecolor='white')
axes[1].set_xlabel("Number of Columns")
axes[1].set_ylabel("Number of Databases")
axes[1].set_title("Columns per Database")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig3_db_complexity.png"), bbox_inches='tight')
plt.close()

# ─── Figure 4: Sample Prompt ──────────────────────────────
print("\nSample formatted prompt:")
from src.data_loader import format_example
sample = dataset["train"][0]
formatted = format_example(sample, schema_manager, config)
print(formatted["text"][:600])
print("..." if len(formatted["text"]) > 600 else "")

print(f"\nAll figures saved to: {FIGURES_DIR}")
print("EDA complete!")
