"""
DIESEL — Notebook 05: Final Evaluation & Comparison
======================================================
Three-way comparison: Base (zero-shot) vs Round 1 vs Round 2 (TGDA).
Generates all publication tables, figures, and statistical tests.

Usage (Colab):
    !python notebooks/05_final_evaluation.py \
        --round1_adapter outputs/round1/final_adapter \
        --round2_adapter outputs/round2/final_adapter \
        --spider_db_dir /path/to/spider/database
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
from src.data_loader import load_spider_dataset
from src.model_loader import load_finetuned_model, load_base_model, load_tokenizer
from src.evaluate import evaluate_model, compare_models
from src.error_analyzer import analyze_errors, compare_error_distributions, ErrorCategory

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="DIESEL Final Evaluation")
parser.add_argument("--round1_adapter", type=str, default="outputs/round1/final_adapter")
parser.add_argument("--round2_adapter", type=str, default="outputs/round2/final_adapter")
parser.add_argument("--spider_db_dir", type=str, default="spider_databases")
parser.add_argument("--max_samples", type=int, default=None)
args = parser.parse_args()

config = get_config()
sns.set_theme(style="whitegrid", font_scale=1.1)
FIGURES_DIR = config.paths.figures_dir
os.makedirs(FIGURES_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# Load Data
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("DIESEL — Final Three-Way Evaluation")
print("=" * 60)

dataset, schema_manager = load_spider_dataset(config)
eval_data = dataset["validation"]
if args.max_samples:
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))

all_results = []
all_analyses = []

# ═══════════════════════════════════════════════════════════
# Evaluate All 3 Models
# ═══════════════════════════════════════════════════════════
models_to_eval = [
    ("base_zero_shot", None),
    ("round1_finetuned", args.round1_adapter),
    ("round2_tgda", args.round2_adapter),
]

for model_name, adapter_path in models_to_eval:
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    if adapter_path:
        model, tokenizer = load_finetuned_model(adapter_path, config)
    else:
        model = load_base_model(config)
        tokenizer = load_tokenizer(config)
        model.eval()
    
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        eval_data=eval_data,
        schema_manager=schema_manager,
        spider_db_dir=args.spider_db_dir,
        config=config,
        model_name=model_name,
        save_dir=config.paths.eval_dir,
    )
    all_results.append(results)
    
    # Error analysis
    analysis = analyze_errors(
        eval_results=results,
        schema_map=schema_manager.schema_map,
        config=config,
    )
    all_analyses.append(analysis)
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════
# Statistical Comparison
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Pairwise Statistical Comparisons")
print("=" * 60)

comparison = compare_models(all_results, save_dir=config.paths.eval_dir)

# ═══════════════════════════════════════════════════════════
# Error Distribution Shifts
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Error Distribution Shifts")
print("=" * 60)

if len(all_analyses) >= 2:
    # Base → Round 1
    shift_r1 = compare_error_distributions(
        all_analyses[0], all_analyses[1],
        save_dir=config.paths.error_analysis_dir,
    )

if len(all_analyses) >= 3:
    # Round 1 → Round 2
    shift_r2 = compare_error_distributions(
        all_analyses[1], all_analyses[2],
        save_dir=config.paths.error_analysis_dir,
    )

# ═══════════════════════════════════════════════════════════
# Publication Figures
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Generating Final Publication Figures")
print("=" * 60)

# Figure 8: Three-Way Accuracy Comparison
print("Figure 8: Three-Way Accuracy Comparison...")
fig, ax = plt.subplots(figsize=(10, 6))

model_names = [r["model_name"] for r in all_results]
accuracies = [r["overall_accuracy"] * 100 for r in all_results]
ci_lower = [r.get("confidence_interval", {}).get("lower_95", 0) * 100 for r in all_results]
ci_upper = [r.get("confidence_interval", {}).get("upper_95", 0) * 100 for r in all_results]
errors_lower = [a - l for a, l in zip(accuracies, ci_lower)]
errors_upper = [u - a for a, u in zip(accuracies, ci_upper)]

colors = ['#FF7043', '#42A5F5', '#66BB6A']
bars = ax.bar(model_names, accuracies, color=colors[:len(model_names)], 
              edgecolor='gray', linewidth=0.5)
ax.errorbar(model_names, accuracies, yerr=[errors_lower, errors_upper],
            fmt='none', color='black', capsize=5, capthick=2)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel("Execution Accuracy (%)")
ax.set_title("DIESEL: Three-Way Model Comparison on Spider Dev")
ax.set_ylim(0, max(accuracies) + 15)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig8_three_way_comparison.png"), bbox_inches='tight')
plt.close()

# Figure 9: Difficulty-Stratified Three-Way Comparison
print("Figure 9: Difficulty-Stratified Comparison...")
fig, ax = plt.subplots(figsize=(12, 7))

all_diffs = set()
for r in all_results:
    all_diffs.update(r.get("difficulty_breakdown", {}).keys())
difficulties = sorted(all_diffs)

x = np.arange(len(difficulties))
width = 0.25

for idx, (res, color) in enumerate(zip(all_results, colors)):
    accs = [
        res.get("difficulty_breakdown", {}).get(d, {}).get("accuracy", 0) * 100
        for d in difficulties
    ]
    ax.bar(x + idx * width, accs, width, label=res["model_name"], color=color, alpha=0.85)

ax.set_xlabel("Query Difficulty")
ax.set_ylabel("Execution Accuracy (%)")
ax.set_title("Accuracy by Difficulty Level")
ax.set_xticks(x + width)
ax.set_xticklabels(difficulties)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig9_difficulty_comparison.png"), bbox_inches='tight')
plt.close()

# Figure 10: Error Distribution Shift (3 models stacked)
print("Figure 10: Error Distribution Shift...")
fig, ax = plt.subplots(figsize=(14, 7))

cats = ErrorCategory.ALL
cat_labels = [ErrorCategory.LABELS[c] for c in cats]
x = np.arange(len(cats))
width = 0.25

for idx, (analysis, color, name) in enumerate(zip(
    all_analyses, colors,
    [r["model_name"] for r in all_results]
)):
    pcts = [
        analysis["category_distribution"][c]["percentage_of_errors"]
        for c in cats
    ]
    ax.bar(x + idx * width, pcts, width, label=name, color=color, alpha=0.85)

ax.set_xlabel("Error Category")
ax.set_ylabel("Percentage of Errors (%)")
ax.set_title("Error Category Distribution: Base → Round 1 → Round 2 (TGDA)")
ax.set_xticks(x + width)
ax.set_xticklabels(cat_labels, rotation=30, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig10_error_shift_3way.png"), bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════
# Publication Results Table (LaTeX)
# ═══════════════════════════════════════════════════════════
print("\nGenerating LaTeX Results Table...")
latex_table = r"""
\begin{table}[t]
\centering
\caption{Execution accuracy (\%) on Spider dev set. CI = 95\% bootstrap confidence interval. $\dagger$ denotes statistical significance at $p < 0.05$ vs. Round~1 (McNemar's test).}
\label{tab:main_results}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Overall} & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} & \textbf{Extra} \\
\midrule
"""

for res in all_results:
    name = res["model_name"].replace("_", " ").title()
    overall = f"{res['overall_accuracy']*100:.1f}"
    ci = res.get("confidence_interval", {})
    ci_str = f"[{ci.get('lower_95',0)*100:.1f}, {ci.get('upper_95',0)*100:.1f}]"
    
    diff_accs = []
    for d in ["easy", "medium", "hard", "extra"]:
        acc = res.get("difficulty_breakdown", {}).get(d, {}).get("accuracy", 0)
        diff_accs.append(f"{acc*100:.1f}")
    
    latex_table += f"{name} & {overall} {ci_str} & {' & '.join(diff_accs)} \\\\\n"

latex_table += r"""
\bottomrule
\end{tabular}
\end{table}
"""

table_path = os.path.join(FIGURES_DIR, "results_table.tex")
with open(table_path, "w") as f:
    f.write(latex_table)

print(latex_table)

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DIESEL — Final Evaluation Complete")
print("=" * 60)
print(f"\nResults:")
for res in all_results:
    ci = res.get("confidence_interval", {})
    print(f"  {res['model_name']:25s}: {res['overall_accuracy']*100:.1f}% "
          f"(95% CI: [{ci.get('lower_95',0)*100:.1f}%, {ci.get('upper_95',0)*100:.1f}%])")

print(f"\nAll figures saved to: {FIGURES_DIR}")
print(f"Results saved to: {config.paths.eval_dir}")
print(f"Error analyses saved to: {config.paths.error_analysis_dir}")
