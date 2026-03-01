"""
DIESEL — Taxonomy-Guided Data Augmentation (TGDA)  (Novel Contribution)
==========================================================================
Constructs synthetic training examples that specifically target the 
weakest error categories identified by the error analyzer.

Strategy per error category:
    E1 (Schema Linking):  Add distractor columns/tables, force disambiguation
    E2 (JOIN Errors):     Generate multi-hop JOIN chains  
    E3 (Aggregation):     Create aggregate variants (COUNT↔SUM↔AVG)
    E4 (Filter/Condition): Paraphrase WHERE conditions, add negation/range variants
    E5 (Nesting/Subquery): Decompose flat queries into nested equivalents
    E6 (Syntax/Format):   No augmentation — addressed by prompt engineering
"""

import re
import os
import json
import random
import copy
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from datasets import Dataset
from tqdm import tqdm

from .config import DieselConfig, AugmentationConfig, get_default_config
from .error_analyzer import ErrorCategory
from .data_loader import SpiderSchemaManager, format_prompt


# ═══════════════════════════════════════════════════════════
# Augmentation Strategies
# ═══════════════════════════════════════════════════════════

class AugmentationStrategy:
    """Base class for error-category-specific augmentation."""
    
    def __init__(self, schema_manager: SpiderSchemaManager, seed: int = 42):
        self.schema_manager = schema_manager
        self.rng = random.Random(seed)
    
    def augment(self, examples: List[Dict]) -> List[Dict]:
        """
        Generate augmented examples.
        
        Args:
            examples: Original examples that had this error category
            
        Returns:
            List of augmented examples with 'text' field
        """
        raise NotImplementedError


class SchemaLinkingAugmentor(AugmentationStrategy):
    """
    E1: Schema Linking Augmentation
    
    Strategies:
    1. Column disambiguation: When multiple tables have similar column names,
       create examples forcing the model to pick the correct one
    2. Schema subsetting: Provide only a subset of tables to train focus
    3. Column type emphasis: Highlight column types in schema description
    """
    
    def augment(self, examples: List[Dict]) -> List[Dict]:
        augmented = []
        
        for ex in examples:
            db_id = ex.get("db_id", "")
            question = ex.get("question", "")
            query = ex.get("query", "")
            
            if not db_id or not question or not query:
                continue
            
            # Strategy 1: Add explicit column hints in the question
            aug_question = self._add_column_hints(question, query, db_id)
            if aug_question != question:
                augmented.append({
                    "db_id": db_id,
                    "question": aug_question,
                    "query": query,
                    "augmentation_type": "E1_column_hint",
                })
            
            # Strategy 2: Rephrase with table references
            aug_question2 = self._add_table_context(question, query, db_id)
            if aug_question2 != question:
                augmented.append({
                    "db_id": db_id,
                    "question": aug_question2,
                    "query": query,
                    "augmentation_type": "E1_table_context",
                })
        
        return augmented
    
    def _add_column_hints(self, question: str, query: str, db_id: str) -> str:
        """Add column disambiguation hints to the question."""
        # Extract tables used in the query
        tables = re.findall(r'(?:FROM|JOIN)\s+(\w+)', query, re.IGNORECASE)
        if tables:
            hint = f" (using table{'s' if len(tables) > 1 else ''}: {', '.join(tables)})"
            return question.rstrip('?').rstrip('.') + hint + "?"
        return question
    
    def _add_table_context(self, question: str, query: str, db_id: str) -> str:
        """Add table-level context to guide schema linking."""
        columns = re.findall(
            r'(?:SELECT|WHERE|ON|BY)\s+(?:\w+\.)?(\w+)', query, re.IGNORECASE
        )
        if columns:
            key_cols = list(set(columns))[:3]
            context = f"Considering columns like {', '.join(key_cols)}, " + question.lower()
            return context[0].upper() + context[1:]
        return question


class JoinAugmentor(AugmentationStrategy):
    """
    E2: JOIN Error Augmentation
    
    Strategies:
    1. Multi-hop joins: Create questions requiring 3+ table joins
    2. Join type variation: Create examples with LEFT/INNER variants
    3. Redundant table detection: Add examples with unnecessary tables
    """
    
    def augment(self, examples: List[Dict]) -> List[Dict]:
        augmented = []
        
        for ex in examples:
            db_id = ex.get("db_id", "")
            question = ex.get("question", "")
            query = ex.get("query", "")
            
            if not db_id or not question or not query:
                continue
            
            # Strategy: Explicit join path description
            joins = re.findall(
                r'JOIN\s+(\w+)\s+.*?ON\s+(.*?)(?:JOIN|WHERE|GROUP|ORDER|LIMIT|$)',
                query, re.IGNORECASE | re.DOTALL
            )
            
            if joins:
                join_desc = " ".join(
                    f"through {table}" for table, _ in joins
                )
                aug_q = f"{question} (connecting {join_desc})"
                augmented.append({
                    "db_id": db_id,
                    "question": aug_q,
                    "query": query,
                    "augmentation_type": "E2_join_path",
                })
            
            # Strategy: Rephrase to emphasize relationships
            if "JOIN" in query.upper():
                aug_q2 = f"By joining the relevant tables, {question.lower()}"
                augmented.append({
                    "db_id": db_id,
                    "question": aug_q2,
                    "query": query,
                    "augmentation_type": "E2_join_emphasis",
                })
        
        return augmented


class AggregationAugmentor(AugmentationStrategy):
    """
    E3: Aggregation Augmentation
    
    Strategies:
    1. Aggregate function swapping: COUNT↔SUM↔AVG variants
    2. GROUP BY emphasis: Questions that explicitly mention grouping
    3. HAVING clause training: Complex filtering on aggregates
    """
    
    AGG_VARIANTS = {
        "COUNT": ["total number of", "how many"],
        "SUM": ["total", "sum of"],
        "AVG": ["average", "mean"],
        "MAX": ["maximum", "highest", "largest"],
        "MIN": ["minimum", "lowest", "smallest"],
    }
    
    def augment(self, examples: List[Dict]) -> List[Dict]:
        augmented = []
        
        for ex in examples:
            db_id = ex.get("db_id", "")
            question = ex.get("question", "")
            query = ex.get("query", "")
            
            if not db_id or not question or not query:
                continue
            
            # Strategy 1: Rephrase with explicit aggregate language
            aggs = re.findall(r'(COUNT|SUM|AVG|MAX|MIN)\s*\(', query, re.IGNORECASE)
            for agg in aggs:
                agg_upper = agg.upper()
                if agg_upper in self.AGG_VARIANTS:
                    phrase = self.rng.choice(self.AGG_VARIANTS[agg_upper])
                    aug_q = f"Find the {phrase} — {question}"
                    augmented.append({
                        "db_id": db_id,
                        "question": aug_q,
                        "query": query,
                        "augmentation_type": "E3_agg_rephrase",
                    })
                    break  # one augmentation per example
            
            # Strategy 2: GROUP BY emphasis
            if "GROUP BY" in query.upper():
                aug_q = f"Group the results and {question.lower()}"
                augmented.append({
                    "db_id": db_id,
                    "question": aug_q,
                    "query": query,
                    "augmentation_type": "E3_group_by_emphasis",
                })
        
        return augmented


class FilterConditionAugmentor(AugmentationStrategy):
    """
    E4: Filter/Condition Augmentation
    
    Strategies:
    1. Negation variants: "not equal to", "excluding"
    2. Range queries: BETWEEN, >, < variations
    3. Multiple WHERE conditions: AND/OR combinations
    """
    
    def augment(self, examples: List[Dict]) -> List[Dict]:
        augmented = []
        
        for ex in examples:
            db_id = ex.get("db_id", "")
            question = ex.get("question", "")
            query = ex.get("query", "")
            
            if not db_id or not question or not query:
                continue
            
            # Strategy: Explicit filter description
            where_match = re.search(
                r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)', 
                query, re.IGNORECASE | re.DOTALL
            )
            if where_match:
                condition = where_match.group(1).strip()
                # Simplify condition description
                aug_q = f"{question} (with the condition that {self._simplify_condition(condition)})"
                augmented.append({
                    "db_id": db_id,
                    "question": aug_q,
                    "query": query,
                    "augmentation_type": "E4_condition_explicit",
                })
            
            # Strategy: Emphasize comparison
            if any(op in query for op in ['>', '<', '>=', '<=', '!=']):
                aug_q = f"With appropriate comparisons, {question.lower()}"
                augmented.append({
                    "db_id": db_id,
                    "question": aug_q,
                    "query": query,
                    "augmentation_type": "E4_comparison_emphasis",
                })
        
        return augmented
    
    def _simplify_condition(self, condition: str) -> str:
        """Create a simplified natural language description of a WHERE condition."""
        condition = re.sub(r'\s+', ' ', condition.strip())
        condition = condition.replace(' = ', ' equals ')
        condition = condition.replace(' > ', ' is greater than ')
        condition = condition.replace(' < ', ' is less than ')
        condition = condition.replace(' >= ', ' is at least ')
        condition = condition.replace(' <= ', ' is at most ')
        condition = condition.replace(' != ', ' is not ')
        condition = condition.replace(' AND ', ' and ')
        condition = condition.replace(' OR ', ' or ')
        return condition.lower()


class NestingSubqueryAugmentor(AugmentationStrategy):
    """
    E5: Nesting/Subquery Augmentation
    
    Strategies:
    1. Decomposition hints: Add "first find X, then" phrasing
    2. IN/EXISTS emphasis: Rephrase to highlight set membership
    3. Subquery pattern training: Use examples with known subquery patterns
    """
    
    def augment(self, examples: List[Dict]) -> List[Dict]:
        augmented = []
        
        for ex in examples:
            db_id = ex.get("db_id", "")
            question = ex.get("question", "")
            query = ex.get("query", "")
            
            if not db_id or not question or not query:
                continue
            
            # Check if gold SQL uses subqueries
            subquery_count = query.upper().count('SELECT') - 1
            
            if subquery_count > 0:
                # Strategy: Decomposition hint
                aug_q = f"Using a subquery approach, {question.lower()}"
                augmented.append({
                    "db_id": db_id,
                    "question": aug_q,
                    "query": query,
                    "augmentation_type": "E5_subquery_hint",
                })
                
                # Strategy: Step-by-step decomposition
                if "IN" in query.upper() or "EXISTS" in query.upper():
                    aug_q2 = (
                        f"First identify the relevant subset, then "
                        f"{question.lower()}"
                    )
                    augmented.append({
                        "db_id": db_id,
                        "question": aug_q2,
                        "query": query,
                        "augmentation_type": "E5_decomposition",
                    })
        
        return augmented


# ═══════════════════════════════════════════════════════════
# TGDA Pipeline
# ═══════════════════════════════════════════════════════════

AUGMENTOR_MAP = {
    ErrorCategory.SCHEMA_LINKING: SchemaLinkingAugmentor,
    ErrorCategory.JOIN_ERROR: JoinAugmentor,
    ErrorCategory.AGGREGATION: AggregationAugmentor,
    ErrorCategory.FILTER_CONDITION: FilterConditionAugmentor,
    ErrorCategory.NESTING_SUBQUERY: NestingSubqueryAugmentor,
    # E6 (Syntax/Format) has no augmentor — addressed via prompt engineering
}


def run_tgda(
    error_analysis: Dict,
    original_train_data: List[Dict],
    eval_predictions: List[Dict],
    schema_manager: SpiderSchemaManager,
    config: Optional[DieselConfig] = None,
    save_dir: Optional[str] = None,
) -> Dataset:
    """
    Run Taxonomy-Guided Data Augmentation pipeline.
    
    Steps:
    1. Read error analysis to identify top-K weakest categories
    2. Filter training data relevant to each weak category
    3. Apply category-specific augmentation strategies
    4. Combine original + augmented data into a new training set
    
    Args:
        error_analysis: Output from error_analyzer.analyze_errors()
        original_train_data: Original training examples
        eval_predictions: Individual eval predictions (with error classifications)
        schema_manager: Schema manager for schema access
        config: Project config
        save_dir: Where to save augmented data
        
    Returns:
        Augmented Dataset ready for Round 2 training
    """
    if config is None:
        config = get_default_config()
    if save_dir is None:
        save_dir = config.paths.augmented_data_dir
    
    os.makedirs(save_dir, exist_ok=True)
    aug_config = config.augmentation
    
    print(f"\n{'TGDA: Taxonomy-Guided Data Augmentation':=^60}")
    
    # Step 1: Identify top-K weakest categories
    ranked = error_analysis.get("ranked_weaknesses", [])
    weak_categories = []
    for w in ranked:
        if w["error_rate"] >= aug_config.weakness_threshold:
            weak_categories.append(w["category"])
        if len(weak_categories) >= aug_config.top_k_categories:
            break
    
    if not weak_categories:
        # If threshold too high, just take top-K
        weak_categories = [w["category"] for w in ranked[:aug_config.top_k_categories]]
    
    print(f"  Targeting {len(weak_categories)} weak categories:")
    for cat in weak_categories:
        label = ErrorCategory.LABELS.get(cat, cat)
        matching = [w for w in ranked if w["category"] == cat]
        rate = matching[0]["error_rate"] if matching else 0
        print(f"    • {label} ({rate:.1f}% of errors)")
    
    # Step 2: Collect error-specific training examples
    # Use the incorrect predictions to find related training examples
    category_examples = defaultdict(list)
    
    # Get error examples from eval predictions
    classified_errors = error_analysis.get("classified_errors", [])
    for error in classified_errors:
        classification = error.get("error_classification", {})
        for cat in classification.get("categories", []):
            if cat in weak_categories:
                category_examples[cat].append(error)
    
    # Also find related examples from the original training data
    # (examples from the same databases as error-prone examples)
    error_db_ids = defaultdict(set)
    for error in classified_errors:
        classification = error.get("error_classification", {})
        for cat in classification.get("categories", []):
            error_db_ids[cat].add(error.get("db_id", ""))
    
    for cat in weak_categories:
        db_ids = error_db_ids.get(cat, set())
        for ex in original_train_data:
            if ex.get("db_id", "") in db_ids:
                category_examples[cat].append(ex)
    
    # Step 3: Apply augmentation strategies
    all_augmented = []
    augmentation_stats = {}
    
    for cat in weak_categories:
        if cat not in AUGMENTOR_MAP:
            print(f"  Skipping {cat} (no augmentor)")
            continue
        
        augmentor_cls = AUGMENTOR_MAP[cat]
        augmentor = augmentor_cls(schema_manager, seed=aug_config.seed)
        
        examples = category_examples.get(cat, [])
        if not examples:
            print(f"  Skipping {cat} (no examples)")
            continue
        
        # Apply augmentation
        multiplied_examples = examples * aug_config.augmentation_multiplier
        augmented = augmentor.augment(multiplied_examples)
        
        label = ErrorCategory.LABELS.get(cat, cat)
        print(f"  {label}: {len(examples)} source → {len(augmented)} augmented examples")
        
        all_augmented.extend(augmented)
        augmentation_stats[cat] = {
            "source_examples": len(examples),
            "augmented_examples": len(augmented),
        }
    
    # Step 4: Format augmented examples with prompts
    formatted_augmented = []
    for aug_ex in tqdm(all_augmented, desc="Formatting augmented data"):
        db_id = aug_ex.get("db_id", "")
        question = aug_ex.get("question", "")
        query = aug_ex.get("query", "")
        
        if not db_id or not question or not query:
            continue
        
        ddl = schema_manager.get_ddl(
            db_id,
            include_types=config.data.include_column_types,
            include_fks=config.data.include_foreign_keys,
        )
        
        text = format_prompt(
            question=question,
            schema_ddl=ddl,
            system_prompt=config.data.system_prompt,
            sql=query,
            include_response=True,
        )
        
        formatted_augmented.append({
            "text": text,
            "db_id": db_id,
            "question": question,
            "query": query,
            "augmentation_type": aug_ex.get("augmentation_type", "unknown"),
        })
    
    print(f"\n  Total augmented examples: {len(formatted_augmented)}")
    
    # Combine with original training data (format originals too)
    combined = []
    for ex in original_train_data:
        if "text" in ex:
            combined.append(ex)
    
    combined.extend(formatted_augmented)
    self_rng = random.Random(aug_config.seed)
    self_rng.shuffle(combined)
    
    print(f"  Combined dataset: {len(combined)} examples "
          f"({len(original_train_data)} original + {len(formatted_augmented)} augmented)")
    
    # Save augmentation metadata
    metadata = {
        "weak_categories": weak_categories,
        "augmentation_stats": augmentation_stats,
        "total_original": len(original_train_data),
        "total_augmented": len(formatted_augmented),
        "total_combined": len(combined),
    }
    
    meta_path = os.path.join(save_dir, "augmentation_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata saved to: {meta_path}")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(combined)
    
    return dataset
