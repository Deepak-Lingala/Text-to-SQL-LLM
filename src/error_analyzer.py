"""
DIESEL — Error Analyzer (Novel Contribution)
===============================================
6-category SQL error taxonomy classifier that analyzes 
incorrect predictions from the fine-tuned model.

Taxonomy:
    E1: Schema Linking    — wrong table/column, hallucinated columns
    E2: JOIN Errors        — missing/wrong join type, unnecessary tables
    E3: Aggregation        — wrong aggregate function, GROUP BY, HAVING
    E4: Filter/Condition   — wrong WHERE clause, operators, values
    E5: Nesting/Subquery   — missing/incorrect subqueries, IN/EXISTS
    E6: Syntax/Format      — unparseable SQL, extra text, dialect errors

The error classifier uses a combination of:
    1. AST-level structural diffing (via sqlparse / sqlglot)
    2. Component set comparison (tables, columns, aggregators)  
    3. Structural pattern matching (subquery depth, JOIN count)
"""

import re
import json
import os
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter

import sqlparse
import numpy as np

from .config import DieselConfig, get_default_config
from .utils import parse_sql_components, normalize_sql


# ═══════════════════════════════════════════════════════════
# Error Classification Engine
# ═══════════════════════════════════════════════════════════

class ErrorCategory:
    """Error category constants."""
    SCHEMA_LINKING = "E1_SCHEMA_LINKING"
    JOIN_ERROR = "E2_JOIN_ERROR"
    AGGREGATION = "E3_AGGREGATION"
    FILTER_CONDITION = "E4_FILTER_CONDITION"
    NESTING_SUBQUERY = "E5_NESTING_SUBQUERY"
    SYNTAX_FORMAT = "E6_SYNTAX_FORMAT"
    
    ALL = [
        SCHEMA_LINKING, JOIN_ERROR, AGGREGATION,
        FILTER_CONDITION, NESTING_SUBQUERY, SYNTAX_FORMAT
    ]
    
    LABELS = {
        "E1_SCHEMA_LINKING": "Schema Linking",
        "E2_JOIN_ERROR": "JOIN Errors",
        "E3_AGGREGATION": "Aggregation",
        "E4_FILTER_CONDITION": "Filter/Condition",
        "E5_NESTING_SUBQUERY": "Nesting/Subquery",
        "E6_SYNTAX_FORMAT": "Syntax/Format",
    }


class SQLErrorClassifier:
    """
    Rule-based SQL error classifier using structural comparison
    between predicted and gold SQL queries.
    
    Design rationale: 
    We use a rule-based approach (rather than LLM-based classification)
    for reproducibility and transparency — a key requirement for the
    ablation study in the paper. Each classification decision can be
    traced to a specific structural difference.
    """
    
    def __init__(self):
        self.category = ErrorCategory
    
    def classify(
        self,
        pred_sql: str,
        gold_sql: str,
        db_schema: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Classify the error(s) in a predicted SQL vs gold SQL.
        A single prediction can have MULTIPLE error types.
        
        Args:
            pred_sql: Predicted SQL query
            gold_sql: Gold standard SQL query
            db_schema: Optional schema dict for column validation
            
        Returns:
            Dict with:
                - categories: List[str] of error categories
                - primary_category: str — most likely primary cause
                - details: Dict with specific error descriptions
                - severity: int (1-3, where 3 = most severe)
        """
        result = {
            "categories": [],
            "primary_category": None,
            "details": {},
            "severity": 0,
        }
        
        # Check E6 first: Syntax/Format errors
        if self._check_syntax_error(pred_sql):
            result["categories"].append(ErrorCategory.SYNTAX_FORMAT)
            result["details"]["syntax"] = self._describe_syntax_error(pred_sql)
            result["severity"] = 3
            result["primary_category"] = ErrorCategory.SYNTAX_FORMAT
            # If it can't even be parsed, we can't do deeper analysis
            return result
        
        # Parse both queries
        pred_components = parse_sql_components(pred_sql)
        gold_components = parse_sql_components(gold_sql)
        
        # E1: Schema Linking
        schema_errors = self._check_schema_linking(
            pred_components, gold_components, db_schema
        )
        if schema_errors:
            result["categories"].append(ErrorCategory.SCHEMA_LINKING)
            result["details"]["schema_linking"] = schema_errors
        
        # E2: JOIN Errors
        join_errors = self._check_join_errors(
            pred_components, gold_components
        )
        if join_errors:
            result["categories"].append(ErrorCategory.JOIN_ERROR)
            result["details"]["join_errors"] = join_errors
        
        # E3: Aggregation
        agg_errors = self._check_aggregation(
            pred_components, gold_components
        )
        if agg_errors:
            result["categories"].append(ErrorCategory.AGGREGATION)
            result["details"]["aggregation"] = agg_errors
        
        # E4: Filter/Condition
        filter_errors = self._check_filter_condition(
            pred_sql, gold_sql, pred_components, gold_components
        )
        if filter_errors:
            result["categories"].append(ErrorCategory.FILTER_CONDITION)
            result["details"]["filter_condition"] = filter_errors
        
        # E5: Nesting/Subquery
        nesting_errors = self._check_nesting(
            pred_components, gold_components
        )
        if nesting_errors:
            result["categories"].append(ErrorCategory.NESTING_SUBQUERY)
            result["details"]["nesting"] = nesting_errors
        
        # If no specific category found, default to E1 (most common)
        if not result["categories"]:
            result["categories"].append(ErrorCategory.SCHEMA_LINKING)
            result["details"]["fallback"] = (
                "No specific structural difference detected; "
                "likely subtle column/value selection error"
            )
        
        # Determine primary category (first detected = most fundamental)
        result["primary_category"] = result["categories"][0]
        
        # Severity (number of distinct error types)
        result["severity"] = min(len(result["categories"]), 3)
        
        return result
    
    def _check_syntax_error(self, sql: str) -> bool:
        """Check if SQL has syntax/format errors."""
        if not sql or not sql.strip():
            return True
        
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return True
            # Check if it starts with a valid SQL keyword
            sql_upper = sql.strip().upper()
            valid_starts = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']
            if not any(sql_upper.startswith(kw) for kw in valid_starts):
                return True
            return False
        except Exception:
            return True
    
    def _describe_syntax_error(self, sql: str) -> Dict:
        """Describe the syntax error."""
        if not sql or not sql.strip():
            return {"type": "empty_output", "description": "Model produced empty or no SQL"}
        return {
            "type": "invalid_sql",
            "description": f"SQL failed to parse: {sql[:100]}..."
        }
    
    def _check_schema_linking(
        self, pred: Dict, gold: Dict, schema: Optional[Dict]
    ) -> Optional[Dict]:
        """Check for schema linking errors (E1)."""
        errors = {}
        
        pred_tables = pred.get("tables", set())
        gold_tables = gold.get("tables", set())
        
        # Missing tables
        missing = gold_tables - pred_tables
        if missing:
            errors["missing_tables"] = list(missing)
        
        # Extra tables (hallucinated)
        extra = pred_tables - gold_tables
        if extra:
            errors["extra_tables"] = list(extra)
        
        # Check for hallucinated columns against schema
        if schema:
            schema_columns = set()
            for _, col_name in schema.get("column_names_original", []):
                if col_name != "*":
                    schema_columns.add(col_name.lower())
            
            # Extract column references from predicted SQL
            pred_col_refs = self._extract_column_refs(str(pred))
            hallucinated = pred_col_refs - schema_columns
            if hallucinated:
                errors["hallucinated_columns"] = list(hallucinated)
        
        return errors if errors else None
    
    def _check_join_errors(self, pred: Dict, gold: Dict) -> Optional[Dict]:
        """Check for JOIN-related errors (E2)."""
        errors = {}
        
        pred_joins = pred.get("joins", [])
        gold_joins = gold.get("joins", [])
        
        pred_join_count = len(pred_joins)
        gold_join_count = len(gold_joins)
        
        if pred_join_count != gold_join_count:
            errors["join_count_mismatch"] = {
                "predicted": pred_join_count,
                "gold": gold_join_count
            }
        
        # Compare join types
        pred_join_types = [j[0].strip().upper() for j in pred_joins]
        gold_join_types = [j[0].strip().upper() for j in gold_joins]
        
        if sorted(pred_join_types) != sorted(gold_join_types):
            errors["join_type_mismatch"] = {
                "predicted": pred_join_types,
                "gold": gold_join_types
            }
        
        # Compare joined tables
        pred_join_tables = set(j[1] for j in pred_joins)
        gold_join_tables = set(j[1] for j in gold_joins)
        
        if pred_join_tables != gold_join_tables:
            missing = gold_join_tables - pred_join_tables
            extra = pred_join_tables - gold_join_tables
            if missing:
                errors["missing_join_tables"] = list(missing)
            if extra:
                errors["extra_join_tables"] = list(extra)
        
        return errors if errors else None
    
    def _check_aggregation(self, pred: Dict, gold: Dict) -> Optional[Dict]:
        """Check for aggregation/grouping errors (E3)."""
        errors = {}
        
        pred_aggs = sorted(pred.get("aggregations", []))
        gold_aggs = sorted(gold.get("aggregations", []))
        
        if pred_aggs != gold_aggs:
            errors["aggregation_mismatch"] = {
                "predicted": pred_aggs,
                "gold": gold_aggs
            }
        
        pred_gb = pred.get("group_by", [])
        gold_gb = gold.get("group_by", [])
        
        if bool(pred_gb) != bool(gold_gb):
            errors["group_by_presence"] = {
                "predicted_has_group_by": bool(pred_gb),
                "gold_has_group_by": bool(gold_gb),
            }
        elif sorted(pred_gb) != sorted(gold_gb):
            errors["group_by_columns"] = {
                "predicted": pred_gb,
                "gold": gold_gb,
            }
        
        pred_having = pred.get("having", [])
        gold_having = gold.get("having", [])
        
        if bool(pred_having) != bool(gold_having):
            errors["having_presence"] = {
                "predicted_has_having": bool(pred_having),
                "gold_has_having": bool(gold_having),
            }
        
        return errors if errors else None
    
    def _check_filter_condition(
        self, pred_sql: str, gold_sql: str, pred: Dict, gold: Dict
    ) -> Optional[Dict]:
        """Check for WHERE clause / condition errors (E4)."""
        errors = {}
        
        pred_where = pred.get("where_conditions", [])
        gold_where = gold.get("where_conditions", [])
        
        pred_has_where = bool(pred_where)
        gold_has_where = bool(gold_where)
        
        if pred_has_where != gold_has_where:
            errors["where_presence"] = {
                "predicted_has_where": pred_has_where,
                "gold_has_where": gold_has_where,
            }
        elif pred_has_where and gold_has_where:
            # Compare operators used
            pred_ops = self._extract_operators(pred_sql)
            gold_ops = self._extract_operators(gold_sql)
            
            if pred_ops != gold_ops:
                errors["operator_mismatch"] = {
                    "predicted": list(pred_ops),
                    "gold": list(gold_ops),
                }
        
        # Check LIMIT, ORDER BY differences
        pred_upper = pred_sql.upper() if pred_sql else ""
        gold_upper = gold_sql.upper() if gold_sql else ""
        
        pred_has_limit = "LIMIT" in pred_upper
        gold_has_limit = "LIMIT" in gold_upper
        if pred_has_limit != gold_has_limit:
            errors["limit_mismatch"] = {
                "predicted_has_limit": pred_has_limit,
                "gold_has_limit": gold_has_limit,
            }
        
        return errors if errors else None
    
    def _check_nesting(self, pred: Dict, gold: Dict) -> Optional[Dict]:
        """Check for subquery/nesting errors (E5)."""
        errors = {}
        
        pred_subq = len(pred.get("subqueries", []))
        gold_subq = len(gold.get("subqueries", []))
        
        if pred_subq != gold_subq:
            errors["subquery_count_mismatch"] = {
                "predicted": pred_subq,
                "gold": gold_subq,
            }
        
        return errors if errors else None
    
    def _extract_column_refs(self, sql_str: str) -> Set[str]:
        """Extract column-like references from SQL."""
        # Simple heuristic: words that appear after SELECT, ON, WHERE, etc.
        cols = re.findall(r'(?:SELECT|ON|WHERE|AND|OR|BY)\s+(\w+)', sql_str, re.IGNORECASE)
        return set(c.lower() for c in cols if c.upper() not in {
            'DISTINCT', 'ALL', 'FROM', 'AS', 'AND', 'OR', 'NOT', 'NULL',
            'TRUE', 'FALSE', 'SELECT', 'WHERE', 'JOIN', 'ON', 'IN',
            'EXISTS', 'BETWEEN', 'LIKE', 'IS', 'CASE', 'WHEN', 'THEN',
            'ELSE', 'END', 'ASC', 'DESC', 'LIMIT', 'OFFSET'
        })
    
    def _extract_operators(self, sql: str) -> Set[str]:
        """Extract comparison operators from SQL."""
        ops = set()
        for op in ['>=', '<=', '!=', '<>', '=', '>', '<', 'LIKE', 'BETWEEN', 'IN', 'NOT IN', 'IS NOT', 'IS']:
            if op in sql.upper():
                ops.add(op)
        return ops


# ═══════════════════════════════════════════════════════════
# Full Error Analysis Pipeline
# ═══════════════════════════════════════════════════════════

def analyze_errors(
    eval_results: Dict,
    schema_map: Optional[Dict] = None,
    config: Optional[DieselConfig] = None,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Run full error taxonomy analysis on evaluation results.
    
    Args:
        eval_results: Output from evaluate.evaluate_model()
        schema_map: Optional db_id→schema mapping for deeper analysis
        config: Project config
        save_dir: Where to save analysis results
        
    Returns:
        Comprehensive error analysis dict
    """
    if config is None:
        config = get_default_config()
    if save_dir is None:
        save_dir = config.paths.error_analysis_dir
    
    os.makedirs(save_dir, exist_ok=True)
    
    classifier = SQLErrorClassifier()
    
    # Filter to incorrect predictions only
    incorrect = [p for p in eval_results["predictions"] if not p["is_correct"]]
    correct = [p for p in eval_results["predictions"] if p["is_correct"]]
    
    print(f"\nError Analysis: {len(incorrect)} incorrect out of "
          f"{len(eval_results['predictions'])} total "
          f"({len(incorrect)/len(eval_results['predictions']):.1%} error rate)")
    
    # Classify each error
    classified_errors = []
    category_counts = Counter()
    primary_counts = Counter()
    difficulty_error_matrix = defaultdict(lambda: Counter())
    
    for pred in incorrect:
        db_schema = schema_map.get(pred["db_id"]) if schema_map else None
        
        classification = classifier.classify(
            pred_sql=pred.get("pred_sql", ""),
            gold_sql=pred.get("gold_sql", ""),
            db_schema=db_schema,
        )
        
        error_record = {
            **pred,
            "error_classification": classification,
        }
        classified_errors.append(error_record)
        
        # Count categories
        for cat in classification["categories"]:
            category_counts[cat] += 1
        
        primary_counts[classification["primary_category"]] += 1
        
        # Difficulty × Error matrix
        difficulty = pred.get("difficulty", "unknown")
        for cat in classification["categories"]:
            difficulty_error_matrix[difficulty][cat] += 1
    
    # Build analysis results
    total_errors = len(incorrect)
    total_examples = len(eval_results["predictions"])
    
    analysis = {
        "model_name": eval_results.get("model_name", "unknown"),
        "total_examples": total_examples,
        "total_correct": len(correct),
        "total_incorrect": total_errors,
        "overall_accuracy": eval_results.get("overall_accuracy", 0),
        
        # Category distribution (each error can be in multiple)
        "category_distribution": {
            cat: {
                "count": category_counts.get(cat, 0),
                "percentage_of_errors": (
                    category_counts.get(cat, 0) / total_errors * 100 
                    if total_errors > 0 else 0
                ),
                "percentage_of_total": (
                    category_counts.get(cat, 0) / total_examples * 100
                    if total_examples > 0 else 0
                ),
            }
            for cat in ErrorCategory.ALL
        },
        
        # Primary category (each error assigned one)
        "primary_category_distribution": {
            cat: {
                "count": primary_counts.get(cat, 0),
                "percentage": (
                    primary_counts.get(cat, 0) / total_errors * 100
                    if total_errors > 0 else 0
                ),
            }
            for cat in ErrorCategory.ALL
        },
        
        # Difficulty × Error matrix (the novel 2D analysis)
        "difficulty_error_matrix": {
            diff: {
                cat: counts.get(cat, 0)
                for cat in ErrorCategory.ALL
            }
            for diff, counts in difficulty_error_matrix.items()
        },
        
        # Ranked weaknesses (for TGDA targeting)
        "ranked_weaknesses": sorted(
            [
                {
                    "category": cat, 
                    "label": ErrorCategory.LABELS.get(cat, cat),
                    "count": category_counts.get(cat, 0),
                    "error_rate": (
                        category_counts.get(cat, 0) / total_errors * 100
                        if total_errors > 0 else 0
                    ),
                }
                for cat in ErrorCategory.ALL
            ],
            key=lambda x: x["count"],
            reverse=True
        ),
        
        # Detailed classified errors (for inspection)
        "classified_errors": classified_errors,
    }
    
    # Print summary
    _print_error_summary(analysis)
    
    # Save
    save_path = os.path.join(save_dir, f"error_analysis_{eval_results.get('model_name', 'model')}.json")
    serializable = json.loads(json.dumps(analysis, default=str))
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Full analysis saved to: {save_path}")
    
    return analysis


def _print_error_summary(analysis: Dict):
    """Pretty-print error analysis summary."""
    print(f"\n{'Error Taxonomy Analysis':=^60}")
    print(f"  Model: {analysis['model_name']}")
    print(f"  Accuracy: {analysis['overall_accuracy']:.1%}")
    print(f"  Errors: {analysis['total_incorrect']}/{analysis['total_examples']}")
    
    print(f"\n  {'Category':<25s} {'Count':>6s} {'% of Errors':>12s} {'% of Total':>12s}")
    print(f"  {'─'*55}")
    for cat in ErrorCategory.ALL:
        data = analysis["category_distribution"][cat]
        label = ErrorCategory.LABELS.get(cat, cat)
        print(f"  {label:<25s} {data['count']:>6d} "
              f"{data['percentage_of_errors']:>11.1f}% "
              f"{data['percentage_of_total']:>11.1f}%")
    
    print(f"\n  Difficulty × Error-Type Matrix:")
    difficulties = sorted(analysis["difficulty_error_matrix"].keys())
    if difficulties:
        header = f"  {'Difficulty':<12s}"
        short_labels = ["E1", "E2", "E3", "E4", "E5", "E6"]
        for sl in short_labels:
            header += f" {sl:>6s}"
        print(header)
        print(f"  {'─'*50}")
        for diff in difficulties:
            row = f"  {diff:<12s}"
            for cat in ErrorCategory.ALL:
                count = analysis["difficulty_error_matrix"][diff].get(cat, 0)
                row += f" {count:>6d}"
            print(row)
    
    print(f"\n  Top Weaknesses (for TGDA targeting):")
    for i, w in enumerate(analysis["ranked_weaknesses"][:3], 1):
        print(f"    {i}. {w['label']}: {w['count']} errors ({w['error_rate']:.1f}%)")


def compare_error_distributions(
    analysis_before: Dict,
    analysis_after: Dict,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Compare error distributions between two models (e.g., Round 1 vs Round 2).
    Shows which error categories improved, worsened, or stayed the same.
    
    Args:
        analysis_before: Error analysis from first model
        analysis_after: Error analysis from second model
        save_dir: Where to save comparison
        
    Returns:
        Dict with shift analysis
    """
    shift = {
        "model_before": analysis_before.get("model_name", "before"),
        "model_after": analysis_after.get("model_name", "after"),
        "accuracy_change": (
            analysis_after["overall_accuracy"] - analysis_before["overall_accuracy"]
        ),
        "category_shifts": {},
    }
    
    for cat in ErrorCategory.ALL:
        before_pct = analysis_before["category_distribution"][cat]["percentage_of_total"]
        after_pct = analysis_after["category_distribution"][cat]["percentage_of_total"]
        change = after_pct - before_pct
        
        shift["category_shifts"][cat] = {
            "label": ErrorCategory.LABELS.get(cat, cat),
            "before_pct": before_pct,
            "after_pct": after_pct,
            "change_pct": change,
            "direction": "improved" if change < 0 else ("worsened" if change > 0 else "unchanged"),
        }
    
    # Print
    print(f"\n{'Error Distribution Shift':=^60}")
    print(f"  {shift['model_before']} → {shift['model_after']}")
    print(f"  Accuracy change: {shift['accuracy_change']:+.1%}")
    print(f"\n  {'Category':<25s} {'Before':>8s} {'After':>8s} {'Change':>8s} {'Dir':>10s}")
    print(f"  {'─'*60}")
    for cat in ErrorCategory.ALL:
        s = shift["category_shifts"][cat]
        print(f"  {s['label']:<25s} {s['before_pct']:>7.1f}% {s['after_pct']:>7.1f}% "
              f"{s['change_pct']:>+7.1f}% {s['direction']:>10s}")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "error_distribution_shift.json")
        with open(path, "w") as f:
            json.dump(shift, f, indent=2, default=str)
    
    return shift
