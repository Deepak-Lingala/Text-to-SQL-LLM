"""
DIESEL — Evaluation Module
=============================
Execution accuracy evaluation on Spider dev set with
difficulty-stratified breakdown and base vs fine-tuned comparison.
"""

import os
import json
import time
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset

from .config import DieselConfig, get_default_config
from .data_loader import SpiderSchemaManager, format_prompt, get_spider_db_path
from .model_loader import (
    load_finetuned_model, load_base_model, load_tokenizer,
    get_generation_config
)
from .utils import (
    extract_sql, execution_accuracy, normalize_sql,
    mcnemar_test, bootstrap_confidence_interval
)


# ═══════════════════════════════════════════════════════════
# SQL Generation
# ═══════════════════════════════════════════════════════════

def generate_sql(
    model,
    tokenizer,
    question: str,
    schema_ddl: str,
    system_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """
    Generate SQL for a single question using the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        question: Natural language question
        schema_ddl: Database schema DDL
        system_prompt: System prompt
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated SQL string
    """
    prompt = format_prompt(
        question=question,
        schema_ddl=schema_ddl,
        system_prompt=system_prompt,
        sql=None,
        include_response=True  # adds assistant header
    )
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated tokens (skip the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return extract_sql(generated_text)


# ═══════════════════════════════════════════════════════════
# Evaluation Pipeline
# ═══════════════════════════════════════════════════════════

def evaluate_model(
    model,
    tokenizer,
    eval_data: Dataset,
    schema_manager: SpiderSchemaManager,
    spider_db_dir: str,
    config: Optional[DieselConfig] = None,
    max_samples: Optional[int] = None,
    model_name: str = "model",
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Evaluate a model on Spider dev set with execution accuracy.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        eval_data: Evaluation dataset
        schema_manager: Schema manager
        spider_db_dir: Path to Spider SQLite databases
        config: Project config
        max_samples: Limit number of examples (None = all)
        model_name: Name for this model in results
        save_dir: Directory to save results
        
    Returns:
        Dict with overall accuracy, per-difficulty breakdown,
        and individual predictions
    """
    if config is None:
        config = get_default_config()
    
    if save_dir is None:
        save_dir = config.paths.eval_dir
    
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # Prepare evaluation examples
    examples = list(eval_data)
    if max_samples:
        examples = examples[:max_samples]
    
    results = {
        "model_name": model_name,
        "num_examples": len(examples),
        "predictions": [],
        "overall_accuracy": 0.0,
        "difficulty_breakdown": {},
        "per_database_accuracy": {},
    }
    
    correct = 0
    difficulty_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    db_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    
    print(f"\nEvaluating {model_name} on {len(examples)} examples...")
    start_time = time.time()
    
    for i, ex in enumerate(tqdm(examples, desc=f"Evaluating {model_name}")):
        db_id = ex["db_id"]
        question = ex["question"]
        gold_sql = ex["query"]
        difficulty = ex.get("difficulty", "unknown")
        
        # Get schema DDL
        ddl = schema_manager.get_ddl(
            db_id,
            include_types=config.data.include_column_types,
            include_fks=config.data.include_foreign_keys,
        )
        
        # Generate SQL
        try:
            pred_sql = generate_sql(
                model, tokenizer, question, ddl,
                config.data.system_prompt
            )
        except Exception as e:
            pred_sql = ""
        
        # Compute execution accuracy
        db_path = get_spider_db_path(db_id, spider_db_dir)
        
        if os.path.exists(db_path):
            is_correct, details = execution_accuracy(db_path, pred_sql, gold_sql)
        else:
            is_correct = False
            details = {"error": f"Database not found: {db_path}"}
        
        # Record
        pred_record = {
            "idx": i,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "difficulty": difficulty,
            "is_correct": is_correct,
            "details": details,
        }
        results["predictions"].append(pred_record)
        
        if is_correct:
            correct += 1
        
        difficulty_counts[difficulty]["total"] += 1
        if is_correct:
            difficulty_counts[difficulty]["correct"] += 1
        
        db_counts[db_id]["total"] += 1
        if is_correct:
            db_counts[db_id]["correct"] += 1
        
        # Progress logging
        if (i + 1) % 50 == 0:
            running_acc = correct / (i + 1)
            print(f"  [{i+1}/{len(examples)}] Running accuracy: {running_acc:.1%}")
    
    elapsed = time.time() - start_time
    
    # Overall accuracy
    results["overall_accuracy"] = correct / len(examples) if examples else 0
    results["num_correct"] = correct
    results["eval_time_seconds"] = elapsed
    results["avg_time_per_example"] = elapsed / len(examples) if examples else 0
    
    # Difficulty breakdown
    for diff, counts in sorted(difficulty_counts.items()):
        acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        results["difficulty_breakdown"][diff] = {
            "correct": counts["correct"],
            "total": counts["total"],
            "accuracy": acc,
        }
    
    # Per-database accuracy
    for db, counts in sorted(db_counts.items()):
        acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        results["per_database_accuracy"][db] = {
            "correct": counts["correct"],
            "total": counts["total"],
            "accuracy": acc,
        }
    
    # Bootstrap confidence interval
    correctness_array = np.array([p["is_correct"] for p in results["predictions"]])
    mean, lower, upper = bootstrap_confidence_interval(correctness_array)
    results["confidence_interval"] = {
        "mean": mean,
        "lower_95": lower,
        "upper_95": upper,
    }
    
    # Print summary
    print(f"\n{'Results for ' + model_name:=^60}")
    print(f"  Overall Accuracy: {results['overall_accuracy']:.1%} "
          f"(95% CI: [{lower:.1%}, {upper:.1%}])")
    print(f"  Evaluation Time: {elapsed:.1f}s ({results['avg_time_per_example']:.2f}s/example)")
    print(f"\n  Difficulty Breakdown:")
    for diff, data in sorted(results["difficulty_breakdown"].items()):
        print(f"    {diff:12s}: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")
    
    # Save results
    results_path = os.path.join(save_dir, f"results_{model_name}.json")
    
    # Make results JSON-serializable
    serializable_results = json.loads(
        json.dumps(results, default=str)
    )
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n  Results saved to: {results_path}")
    
    return results


# ═══════════════════════════════════════════════════════════
# Comparative Evaluation
# ═══════════════════════════════════════════════════════════

def compare_models(
    results_list: List[Dict],
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Compare evaluation results across multiple models.
    Computes McNemar's test for significance between each pair.
    
    Args:
        results_list: List of evaluation result dicts
                      (from evaluate_model)
        save_dir: Where to save comparison results
        
    Returns:
        Dict with pairwise comparisons and summary table
    """
    comparison = {
        "models": [],
        "pairwise_tests": [],
        "summary_table": [],
    }
    
    # Summary for each model
    for res in results_list:
        ci = res.get("confidence_interval", {})
        comparison["models"].append({
            "name": res["model_name"],
            "accuracy": res["overall_accuracy"],
            "ci_lower": ci.get("lower_95", 0),
            "ci_upper": ci.get("upper_95", 0),
            "num_examples": res["num_examples"],
            "difficulty_breakdown": res.get("difficulty_breakdown", {}),
        })
    
    # Pairwise McNemar's test
    for i in range(len(results_list)):
        for j in range(i + 1, len(results_list)):
            res_a = results_list[i]
            res_b = results_list[j]
            
            # Align predictions by index
            correct_a = np.array([p["is_correct"] for p in res_a["predictions"]])
            correct_b = np.array([p["is_correct"] for p in res_b["predictions"]])
            
            if len(correct_a) != len(correct_b):
                print(f"  Warning: {res_a['model_name']} ({len(correct_a)}) vs "
                      f"{res_b['model_name']} ({len(correct_b)}) — length mismatch")
                min_len = min(len(correct_a), len(correct_b))
                correct_a = correct_a[:min_len]
                correct_b = correct_b[:min_len]
            
            chi2, p_value = mcnemar_test(correct_a, correct_b)
            
            comparison["pairwise_tests"].append({
                "model_a": res_a["model_name"],
                "model_b": res_b["model_name"],
                "accuracy_a": res_a["overall_accuracy"],
                "accuracy_b": res_b["overall_accuracy"],
                "mcnemar_chi2": chi2,
                "p_value": p_value,
                "significant_at_05": p_value < 0.05,
                "significant_at_01": p_value < 0.01,
            })
    
    # Print comparison
    print(f"\n{'Model Comparison':=^60}")
    for m in comparison["models"]:
        print(f"  {m['name']:20s}: {m['accuracy']:.1%} "
              f"(95% CI: [{m['ci_lower']:.1%}, {m['ci_upper']:.1%}])")
    
    print(f"\n  Pairwise Statistical Tests (McNemar's):")
    for test in comparison["pairwise_tests"]:
        sig = "***" if test["significant_at_01"] else ("*" if test["significant_at_05"] else "n.s.")
        print(f"    {test['model_a']} vs {test['model_b']}: "
              f"χ²={test['mcnemar_chi2']:.2f}, p={test['p_value']:.4f} {sig}")
    
    # Save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "model_comparison.json")
        with open(path, "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"\n  Comparison saved to: {path}")
    
    return comparison
