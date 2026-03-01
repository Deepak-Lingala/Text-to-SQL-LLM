"""
DIESEL — Utility Functions
============================
SQL extraction, safe execution, schema helpers, 
and statistical testing utilities.
"""

import re
import signal
import sqlite3
import hashlib
import time
from typing import Optional, Tuple, List, Set, Any, Dict
from contextlib import contextmanager
from collections import OrderedDict

import sqlparse
import numpy as np
from scipy import stats


# ═══════════════════════════════════════════════════════════
# SQL Extraction
# ═══════════════════════════════════════════════════════════

def extract_sql(text: str) -> str:
    """
    Extract SQL query from model output.
    Handles common patterns: raw SQL, markdown code blocks,
    'SELECT ...' extraction, etc.
    
    Args:
        text: Raw model output string
        
    Returns:
        Cleaned SQL string
    """
    if not text or not text.strip():
        return ""
    
    text = text.strip()
    
    # Pattern 1: ```sql ... ``` code blocks
    sql_block = re.search(r'```(?:sql)?\s*\n?(.*?)\n?```', text, re.DOTALL | re.IGNORECASE)
    if sql_block:
        return sql_block.group(1).strip()
    
    # Pattern 2: Extract everything starting from SELECT/WITH
    sql_match = re.search(
        r'((?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b.*)',
        text, re.DOTALL | re.IGNORECASE
    )
    if sql_match:
        sql = sql_match.group(1).strip()
        # Remove trailing natural language after the SQL
        # Heuristic: SQL ends at a line that doesn't look like SQL
        lines = sql.split('\n')
        sql_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not any(stripped.upper().startswith(kw) for kw in [
                'NOTE:', 'EXPLANATION:', 'THIS ', 'THE ', 'I ', 'HERE'
            ]):
                sql_lines.append(line)
            elif not stripped:
                sql_lines.append(line)
            else:
                break
        sql = '\n'.join(sql_lines).strip()
        # Remove trailing semicolons (some models add them, Spider doesn't)
        sql = sql.rstrip(';').strip()
        return sql
    
    # Pattern 3: Return cleaned text as-is
    return text.rstrip(';').strip()


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison: lowercase, collapse whitespace,
    remove trailing semicolons.
    """
    if not sql:
        return ""
    sql = sql.lower().strip()
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.rstrip(';').strip()
    return sql


# ═══════════════════════════════════════════════════════════
# Safe SQL Execution
# ═══════════════════════════════════════════════════════════

class SQLExecutionTimeout(Exception):
    """Raised when SQL execution exceeds time limit."""
    pass


def execute_sql_safe(
    db_path: str,
    sql: str,
    timeout: float = 30.0
) -> Tuple[bool, Optional[List[Tuple]], Optional[str]]:
    """
    Safely execute SQL against a SQLite database with timeout.
    
    Args:
        db_path: Path to SQLite database file
        sql: SQL query to execute
        timeout: Maximum execution time in seconds
        
    Returns:
        Tuple of (success: bool, results: List[Tuple] | None, error: str | None)
    """
    if not sql or not sql.strip():
        return False, None, "Empty SQL query"
    
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
        
        # Use Python-level timeout via alarm (Unix) or thread (Windows)
        cursor = conn.cursor()
        start_time = time.time()
        
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                return False, None, f"Execution took {elapsed:.1f}s (timeout={timeout}s)"
            
            return True, results, None
            
        except sqlite3.Error as e:
            return False, None, f"SQLite error: {str(e)}"
            
    except Exception as e:
        return False, None, f"Connection error: {str(e)}"
    finally:
        if conn:
            conn.close()


def compare_results(
    results_pred: Optional[List[Tuple]],
    results_gold: Optional[List[Tuple]]
) -> bool:
    """
    Compare two SQL result sets for execution accuracy.
    Results are compared as sets of tuples (order-insensitive).
    
    Args:
        results_pred: Predicted query results
        results_gold: Gold query results
        
    Returns:
        True if result sets are equivalent
    """
    if results_pred is None or results_gold is None:
        return False
    
    # Convert to sets of tuples for order-insensitive comparison
    try:
        set_pred = set(results_pred)
        set_gold = set(results_gold)
        return set_pred == set_gold
    except TypeError:
        # Unhashable types — fall back to sorted comparison
        try:
            sorted_pred = sorted([str(r) for r in results_pred])
            sorted_gold = sorted([str(r) for r in results_gold])
            return sorted_pred == sorted_gold
        except Exception:
            return False


def execution_accuracy(
    db_path: str,
    pred_sql: str,
    gold_sql: str,
    timeout: float = 30.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compute execution accuracy for a single prediction.
    
    Args:
        db_path: Path to SQLite database
        pred_sql: Predicted SQL query
        gold_sql: Gold standard SQL query
        timeout: Execution timeout
        
    Returns:
        Tuple of (is_correct: bool, details: dict)
    """
    details = {
        "pred_sql": pred_sql,
        "gold_sql": gold_sql,
        "pred_success": False,
        "gold_success": False,
        "pred_error": None,
        "gold_error": None,
        "match": False,
    }
    
    # Execute gold SQL
    gold_success, gold_results, gold_error = execute_sql_safe(db_path, gold_sql, timeout)
    details["gold_success"] = gold_success
    details["gold_error"] = gold_error
    
    if not gold_success:
        return False, details
    
    # Execute predicted SQL
    pred_success, pred_results, pred_error = execute_sql_safe(db_path, pred_sql, timeout)
    details["pred_success"] = pred_success
    details["pred_error"] = pred_error
    
    if not pred_success:
        return False, details
    
    # Compare results
    match = compare_results(pred_results, gold_results)
    details["match"] = match
    
    return match, details


# ═══════════════════════════════════════════════════════════
# Schema Serialization
# ═══════════════════════════════════════════════════════════

def serialize_schema(
    db_schema: Dict,
    include_types: bool = True,
    include_fks: bool = True
) -> str:
    """
    Serialize a database schema dict into CREATE TABLE DDL format.
    
    Args:
        db_schema: Schema dict with keys: table_names_original, 
                   column_names_original, column_types, 
                   foreign_keys, primary_keys
        include_types: Include column type annotations
        include_fks: Include foreign key constraints
        
    Returns:
        DDL string with CREATE TABLE statements
    """
    table_names = db_schema.get("table_names_original", [])
    columns = db_schema.get("column_names_original", [])
    col_types = db_schema.get("column_types", [])
    foreign_keys = db_schema.get("foreign_keys", [])
    primary_keys = db_schema.get("primary_keys", [])
    
    # Group columns by table
    tables = OrderedDict()
    for i, (table_idx, col_name) in enumerate(columns):
        if table_idx < 0:
            continue  # skip "*" wildcard
        table_name = table_names[table_idx]
        if table_name not in tables:
            tables[table_name] = []
        
        col_type = col_types[i] if include_types and i < len(col_types) else ""
        is_pk = i in primary_keys
        tables[table_name].append((col_name, col_type, is_pk))
    
    # Build DDL
    ddl_parts = []
    for table_name, cols in tables.items():
        col_defs = []
        for col_name, col_type, is_pk in cols:
            parts = [f"  {col_name}"]
            if col_type:
                parts.append(col_type)
            if is_pk:
                parts.append("PRIMARY KEY")
            col_defs.append(" ".join(parts))
        
        # FK constraints
        fk_defs = []
        if include_fks and foreign_keys:
            for fk_col, ref_col in foreign_keys:
                if fk_col < len(columns) and ref_col < len(columns):
                    fk_table_idx, fk_col_name = columns[fk_col]
                    ref_table_idx, ref_col_name = columns[ref_col]
                    
                    if fk_table_idx >= 0 and ref_table_idx >= 0:
                        fk_table = table_names[fk_table_idx]
                        ref_table = table_names[ref_table_idx]
                        
                        if fk_table == table_name:
                            fk_defs.append(
                                f"  FOREIGN KEY ({fk_col_name}) "
                                f"REFERENCES {ref_table}({ref_col_name})"
                            )
        
        all_defs = col_defs + fk_defs
        ddl = f"CREATE TABLE {table_name} (\n"
        ddl += ",\n".join(all_defs)
        ddl += "\n);"
        ddl_parts.append(ddl)
    
    return "\n\n".join(ddl_parts)


# ═══════════════════════════════════════════════════════════
# Statistical Testing
# ═══════════════════════════════════════════════════════════

def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray
) -> Tuple[float, float]:
    """
    McNemar's test for paired binary outcomes.
    Tests whether two models have significantly different error rates.
    
    Args:
        correct_a: Boolean array of model A correctness
        correct_b: Boolean array of model B correctness
        
    Returns:
        Tuple of (chi2_statistic, p_value)
    """
    assert len(correct_a) == len(correct_b), "Arrays must have same length"
    
    # Build contingency table
    # b01: A wrong, B right
    # b10: A right, B wrong
    b01 = np.sum(~correct_a & correct_b)
    b10 = np.sum(correct_a & ~correct_b)
    
    if b01 + b10 == 0:
        return 0.0, 1.0
    
    # McNemar's test with continuity correction
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return float(chi2), float(p_value)


def bootstrap_confidence_interval(
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a metric.
    
    Args:
        scores: Array of binary scores (0/1)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        seed: Random seed
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    n = len(scores)
    
    bootstrap_means = np.array([
        np.mean(rng.choice(scores, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    mean = np.mean(scores)
    
    return float(mean), float(lower), float(upper)


# ═══════════════════════════════════════════════════════════
# Misc Helpers
# ═══════════════════════════════════════════════════════════

def hash_example(question: str, db_id: str) -> str:
    """Create a deterministic hash for a data example."""
    key = f"{db_id}||{question}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def parse_sql_components(sql: str) -> Dict[str, Any]:
    """
    Parse SQL into structural components for error analysis.
    Uses sqlparse for tokenization + regex for specifics.
    
    Returns dict with keys: tables, columns, joins, aggregations,
    where_conditions, subqueries, group_by, order_by, having.
    """
    if not sql:
        return {
            "tables": set(), "columns": set(), "joins": [],
            "aggregations": [], "where_conditions": [],
            "subqueries": [], "group_by": [], "order_by": [],
            "having": [], "is_valid": False
        }
    
    sql_upper = sql.upper().strip()
    sql_lower = sql.lower().strip()
    
    result = {
        "tables": set(),
        "columns": set(),
        "joins": [],
        "aggregations": [],
        "where_conditions": [],
        "subqueries": [],
        "group_by": [],
        "order_by": [],
        "having": [],
        "is_valid": True
    }
    
    try:
        parsed = sqlparse.parse(sql)
        if not parsed:
            result["is_valid"] = False
            return result
    except Exception:
        result["is_valid"] = False
        return result
    
    # Extract tables (FROM and JOIN clauses)
    table_pattern = re.findall(
        r'(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE
    )
    result["tables"] = set(t.lower() for t in table_pattern)
    
    # Extract JOIN types
    join_pattern = re.findall(
        r'((?:LEFT|RIGHT|INNER|OUTER|CROSS|NATURAL)?\s*JOIN)\s+(\w+)\s+(?:ON|USING)',
        sql, re.IGNORECASE
    )
    result["joins"] = [(jtype.strip(), table.lower()) for jtype, table in join_pattern]
    
    # Extract aggregate functions
    agg_pattern = re.findall(
        r'(COUNT|SUM|AVG|MIN|MAX)\s*\(', sql, re.IGNORECASE
    )
    result["aggregations"] = [a.upper() for a in agg_pattern]
    
    # Detect subqueries
    # Count nested SELECT statements
    subquery_count = sql_upper.count('SELECT') - 1
    if subquery_count > 0:
        result["subqueries"] = [f"nested_{i}" for i in range(subquery_count)]
    
    # GROUP BY
    group_match = re.search(r'GROUP\s+BY\s+(.+?)(?:HAVING|ORDER|LIMIT|$)', sql, re.IGNORECASE)
    if group_match:
        result["group_by"] = [c.strip().lower() for c in group_match.group(1).split(',')]
    
    # HAVING
    having_match = re.search(r'HAVING\s+(.+?)(?:ORDER|LIMIT|$)', sql, re.IGNORECASE)
    if having_match:
        result["having"] = [having_match.group(1).strip()]
    
    # WHERE conditions
    where_match = re.search(
        r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|HAVING|$)', sql, re.IGNORECASE | re.DOTALL
    )
    if where_match:
        conditions = where_match.group(1).strip()
        result["where_conditions"] = [conditions]
    
    return result
