"""
DIESEL — Interactive Inference
=================================
Compare base model vs Round 1 vs Round 2 fine-tuned models 
on user-provided questions.
"""

import argparse
import os
from typing import Optional

import torch

from .config import DieselConfig, get_default_config
from .data_loader import SpiderSchemaManager, load_spider_dataset
from .model_loader import load_finetuned_model, load_base_model, load_tokenizer, get_generation_config
from .evaluate import generate_sql


def interactive_inference(
    adapter_path: Optional[str] = None,
    config: Optional[DieselConfig] = None,
):
    """
    Run interactive SQL generation.
    
    Args:
        adapter_path: Path to LoRA adapter (None = base model)
        config: Project config
    """
    if config is None:
        config = get_default_config()
    
    # Load model
    if adapter_path:
        print(f"Loading fine-tuned model from {adapter_path}...")
        model, tokenizer = load_finetuned_model(adapter_path, config)
    else:
        print("Loading base model (no fine-tuning)...")
        model = load_base_model(config)
        tokenizer = load_tokenizer(config)
        model.eval()
    
    # Load schemas
    print("Loading Spider schemas...")
    dataset, schema_manager = load_spider_dataset(config)
    
    print("\n" + "=" * 60)
    print("DIESEL Interactive SQL Generator")
    print("=" * 60)
    print(f"Model: {'Fine-tuned' if adapter_path else 'Base'}")
    print(f"Available databases: {len(schema_manager.db_ids)}")
    print("Type 'quit' to exit, 'dbs' to list databases")
    print("=" * 60)
    
    while True:
        try:
            # Get database
            db_id = input("\nDatabase ID (or 'dbs'/'quit'): ").strip()
            
            if db_id.lower() == 'quit':
                break
            
            if db_id.lower() == 'dbs':
                print("\nAvailable databases:")
                for i, db in enumerate(sorted(schema_manager.db_ids)):
                    tables = schema_manager.get_tables(db)
                    print(f"  {db} ({len(tables)} tables)")
                continue
            
            if db_id not in schema_manager.schema_map:
                print(f"Unknown database: {db_id}")
                continue
            
            # Show schema
            ddl = schema_manager.get_ddl(db_id)
            print(f"\nSchema for {db_id}:")
            print(ddl[:500] + ("..." if len(ddl) > 500 else ""))
            
            # Get question
            question = input("\nQuestion: ").strip()
            if not question:
                continue
            
            # Generate SQL
            print("\nGenerating SQL...")
            sql = generate_sql(
                model, tokenizer, question, ddl,
                config.data.system_prompt
            )
            
            print(f"\nGenerated SQL:\n  {sql}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_inference(
    questions: list,
    db_ids: list,
    adapter_path: Optional[str] = None,
    config: Optional[DieselConfig] = None,
) -> list:
    """
    Run batch inference on multiple questions.
    
    Args:
        questions: List of NL questions
        db_ids: List of database IDs
        adapter_path: Path to LoRA adapter
        config: Project config
        
    Returns:
        List of generated SQL queries
    """
    if config is None:
        config = get_default_config()
    
    if adapter_path:
        model, tokenizer = load_finetuned_model(adapter_path, config)
    else:
        model = load_base_model(config)
        tokenizer = load_tokenizer(config)
        model.eval()
    
    _, schema_manager = load_spider_dataset(config)
    
    results = []
    for q, db_id in zip(questions, db_ids):
        ddl = schema_manager.get_ddl(db_id)
        sql = generate_sql(
            model, tokenizer, q, ddl, config.data.system_prompt
        )
        results.append(sql)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIESEL Interactive Inference")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter")
    args = parser.parse_args()
    
    interactive_inference(adapter_path=args.adapter)
