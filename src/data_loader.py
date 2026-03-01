"""
DIESEL — Data Loader
======================
Spider dataset loading, schema serialization, 
prompt formatting, and train/val splitting.
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

from .config import DieselConfig, get_default_config
from .utils import serialize_schema


# ═══════════════════════════════════════════════════════════
# Spider Schema Management
# ═══════════════════════════════════════════════════════════

class SpiderSchemaManager:
    """
    Manages Spider database schemas. Builds a db_id→schema mapping 
    and provides serialized DDL for each database.
    """
    
    def __init__(self, schemas: List[Dict]):
        """
        Args:
            schemas: List of schema dicts from Spider dataset,
                     each with db_id, table_names_original, 
                     column_names_original, column_types,
                     foreign_keys, primary_keys.
        """
        self.schema_map: Dict[str, Dict] = {}
        self._ddl_cache: Dict[str, str] = {}
        
        for schema in schemas:
            db_id = schema["db_id"]
            self.schema_map[db_id] = schema
    
    def get_ddl(
        self, 
        db_id: str, 
        include_types: bool = True,
        include_fks: bool = True
    ) -> str:
        """Get serialized DDL for a database."""
        cache_key = f"{db_id}_{include_types}_{include_fks}"
        if cache_key not in self._ddl_cache:
            schema = self.schema_map.get(db_id)
            if schema is None:
                return f"-- Unknown database: {db_id}"
            self._ddl_cache[cache_key] = serialize_schema(
                schema, include_types=include_types, include_fks=include_fks
            )
        return self._ddl_cache[cache_key]
    
    def get_tables(self, db_id: str) -> List[str]:
        """Get table names for a database."""
        schema = self.schema_map.get(db_id, {})
        return schema.get("table_names_original", [])
    
    def get_columns(self, db_id: str) -> List[Tuple[int, str]]:
        """Get columns as (table_idx, column_name) pairs."""
        schema = self.schema_map.get(db_id, {})
        return schema.get("column_names_original", [])
    
    @property
    def db_ids(self) -> List[str]:
        return list(self.schema_map.keys())


# ═══════════════════════════════════════════════════════════
# Prompt Formatting
# ═══════════════════════════════════════════════════════════

def format_prompt(
    question: str, 
    schema_ddl: str,
    system_prompt: str,
    sql: Optional[str] = None,
    include_response: bool = True
) -> str:
    """
    Format a training/inference prompt using the Llama 3.1 chat template.
    
    Args:
        question: Natural language question
        schema_ddl: Serialized database schema (CREATE TABLE DDL)
        system_prompt: System instruction
        sql: Gold SQL query (None for inference)
        include_response: Whether to include the assistant response
        
    Returns:
        Formatted prompt string
    """
    user_content = f"""Given the following database schema:

{schema_ddl}

Question: {question}"""
    
    # Llama 3.1 Instruct chat format
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
    )
    
    if include_response and sql is not None:
        prompt += (
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{sql}<|eot_id|>"
        )
    elif include_response:
        prompt += (
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    
    return prompt


def format_example(
    example: Dict,
    schema_manager: SpiderSchemaManager,
    config: DieselConfig
) -> Dict:
    """
    Format a single Spider example into a training-ready dict.
    
    Args:
        example: Spider dataset example
        schema_manager: Schema manager instance
        config: Project config
        
    Returns:
        Dict with 'text' (full prompt) and metadata fields
    """
    db_id = example["db_id"]
    question = example["question"]
    sql = example["query"]
    
    ddl = schema_manager.get_ddl(
        db_id,
        include_types=config.data.include_column_types,
        include_fks=config.data.include_foreign_keys,
    )
    
    text = format_prompt(
        question=question,
        schema_ddl=ddl,
        system_prompt=config.data.system_prompt,
        sql=sql,
        include_response=True
    )
    
    return {
        "text": text,
        "db_id": db_id,
        "question": question,
        "query": sql,
    }


# ═══════════════════════════════════════════════════════════
# Dataset Loading
# ═══════════════════════════════════════════════════════════

def load_spider_dataset(
    config: Optional[DieselConfig] = None,
    cache_dir: Optional[str] = None
) -> Tuple[DatasetDict, SpiderSchemaManager]:
    """
    Load the Spider dataset from HuggingFace + build schema manager.
    
    Returns:
        Tuple of (dataset_dict, schema_manager)
        dataset_dict has 'train', 'validation' splits
    """
    if config is None:
        config = get_default_config()
    
    _cache = cache_dir or config.paths.cache_dir
    
    print("Loading Spider dataset from HuggingFace...")
    raw = load_dataset(config.data.dataset_name, cache_dir=_cache)
    
    # Build schema manager from training schemas
    # Spider stores schemas in the train split examples
    all_schemas = {}
    for split_name in raw.keys():
        for ex in raw[split_name]:
            db_id = ex["db_id"]
            if db_id not in all_schemas:
                all_schemas[db_id] = {
                    "db_id": db_id,
                    "table_names_original": ex.get("table_names_original", []),
                    "column_names_original": ex.get("column_names_original", []),
                    "column_types": ex.get("column_types", []),
                    "foreign_keys": ex.get("foreign_keys", []),
                    "primary_keys": ex.get("primary_keys", []),
                }
    
    schema_manager = SpiderSchemaManager(list(all_schemas.values()))
    print(f"  Loaded {len(schema_manager.db_ids)} database schemas")
    
    # Handle splits
    # Spider typically has 'train' and 'validation' 
    if "validation" in raw:
        train_data = raw["train"]
        val_data = raw["validation"]
    else:
        # Split manually
        print(f"  Splitting train set ({config.data.val_split_ratio:.0%} validation)...")
        split = raw["train"].train_test_split(
            test_size=config.data.val_split_ratio,
            seed=config.training.seed
        )
        train_data = split["train"]
        val_data = split["test"]
    
    # Subsample if configured
    if config.data.max_train_samples and len(train_data) > config.data.max_train_samples:
        train_data = train_data.select(range(config.data.max_train_samples))
    if config.data.max_eval_samples and len(val_data) > config.data.max_eval_samples:
        val_data = val_data.select(range(config.data.max_eval_samples))
    
    print(f"  Train: {len(train_data)} examples")
    print(f"  Validation: {len(val_data)} examples")
    
    dataset = DatasetDict({
        "train": train_data,
        "validation": val_data
    })
    
    return dataset, schema_manager


def prepare_training_data(
    dataset: DatasetDict,
    schema_manager: SpiderSchemaManager,
    config: Optional[DieselConfig] = None,
    split: str = "train"
) -> Dataset:
    """
    Format raw Spider examples into training-ready text prompts.
    
    Args:
        dataset: Spider DatasetDict
        schema_manager: Schema manager
        config: Project config
        split: Which split to prepare ('train' or 'validation')
        
    Returns:
        HuggingFace Dataset with 'text' column
    """
    if config is None:
        config = get_default_config()
    
    data = dataset[split]
    
    formatted = []
    skipped = 0
    for example in tqdm(data, desc=f"Formatting {split}"):
        try:
            f = format_example(example, schema_manager, config)
            # Skip if prompt exceeds max length (rough char estimate)
            if len(f["text"]) > config.training.max_seq_length * 4:
                skipped += 1
                continue
            formatted.append(f)
        except Exception as e:
            skipped += 1
            continue
    
    if skipped > 0:
        print(f"  Skipped {skipped} examples (too long or formatting errors)")
    
    return Dataset.from_list(formatted)


def get_spider_db_path(db_id: str, spider_db_dir: str) -> str:
    """
    Get the SQLite database file path for a Spider db_id.
    
    Args:
        db_id: Database identifier (e.g., 'concert_singer')
        spider_db_dir: Root directory containing Spider databases
        
    Returns:
        Path to the .sqlite file
    """
    return os.path.join(spider_db_dir, db_id, f"{db_id}.sqlite")


# ═══════════════════════════════════════════════════════════
# Data Statistics
# ═══════════════════════════════════════════════════════════

def compute_dataset_statistics(
    dataset: DatasetDict,
    schema_manager: SpiderSchemaManager
) -> Dict:
    """
    Compute comprehensive statistics about the Spider dataset for EDA.
    
    Returns:
        Dict with statistics about queries, schemas, difficulty, etc.
    """
    stats = {
        "num_train": len(dataset["train"]),
        "num_val": len(dataset["validation"]),
        "num_databases": len(schema_manager.db_ids),
        "query_lengths": [],
        "question_lengths": [],
        "tables_per_db": [],
        "cols_per_db": [],
        "difficulty_distribution": {},
        "query_keywords": {},
    }
    
    sql_keywords = ["JOIN", "GROUP BY", "ORDER BY", "HAVING",
                    "UNION", "INTERSECT", "EXCEPT", "LIKE",
                    "BETWEEN", "IN", "EXISTS", "NOT"]
    
    for split in ["train", "validation"]:
        for ex in dataset[split]:
            query = ex.get("query", "")
            question = ex.get("question", "")
            
            stats["query_lengths"].append(len(query))
            stats["question_lengths"].append(len(question))
            
            # Count SQL keywords
            query_upper = query.upper()
            for kw in sql_keywords:
                if kw in query_upper:
                    stats["query_keywords"][kw] = \
                        stats["query_keywords"].get(kw, 0) + 1
    
    for db_id in schema_manager.db_ids:
        tables = schema_manager.get_tables(db_id)
        cols = schema_manager.get_columns(db_id)
        stats["tables_per_db"].append(len(tables))
        stats["cols_per_db"].append(len([c for c in cols if c[0] >= 0]))
    
    return stats
