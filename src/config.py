"""
DIESEL — Centralized Configuration
===================================
All hyperparameters, paths, and constants in one place
for reproducibility across experiments.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os
from pathlib import Path


@dataclass
class ModelConfig:
    """Base model and quantization settings."""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    trust_remote_code: bool = True
    torch_dtype: str = "float16"  # compute dtype for QLoRA
    
    # BitsAndBytes 4-bit quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """Low-Rank Adaptation parameters."""
    r: int = 16                  # LoRA rank
    lora_alpha: int = 32         # scaling factor
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass 
class TrainingConfig:
    """Training hyperparameters."""
    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Batching
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    # Effective batch size = 4 * 4 = 16
    
    # Duration
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 = use epochs
    
    # Sequence
    max_seq_length: int = 1024
    
    # Memory optimization
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False  # L4 supports bf16, but fp16 is safer
    optim: str = "paged_adamw_32bit"
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Reproducibility
    seed: int = 42
    
    # WandB
    report_to: str = "wandb"
    run_name: Optional[str] = None


@dataclass
class DataConfig:
    """Dataset and preprocessing settings."""
    dataset_name: str = "xlangai/spider"
    val_split_ratio: float = 0.1
    max_train_samples: Optional[int] = None  # None = all
    max_eval_samples: Optional[int] = None
    
    # Schema serialization
    include_foreign_keys: bool = True
    include_column_types: bool = True
    
    # Prompt template
    system_prompt: str = (
        "You are an expert SQL assistant. Given a database schema and a "
        "natural language question, generate the correct SQL query. "
        "Output ONLY the SQL query, nothing else."
    )


@dataclass
class ErrorTaxonomy:
    """Error category definitions for the 6-category taxonomy."""
    categories: dict = field(default_factory=lambda: {
        "E1_SCHEMA_LINKING": "Wrong table/column selected or hallucinated columns",
        "E2_JOIN_ERROR": "Missing joins, wrong join type, or unnecessary tables",
        "E3_AGGREGATION": "Wrong aggregate function, missing GROUP BY, wrong HAVING",
        "E4_FILTER_CONDITION": "Wrong WHERE predicate, comparison operator, or values",
        "E5_NESTING_SUBQUERY": "Missing/incorrect subquery, wrong IN/EXISTS usage",
        "E6_SYNTAX_FORMAT": "Unparseable SQL, extra text, or dialect errors",
    })


@dataclass
class AugmentationConfig:
    """Taxonomy-Guided Data Augmentation settings."""
    # How many augmented examples to generate per weak category
    augmentation_multiplier: int = 3
    # Minimum error rate (%) to trigger augmentation for a category
    weakness_threshold: float = 15.0
    # Top-K weakest categories to augment
    top_k_categories: int = 3
    # Random seed for augmentation
    seed: int = 42


@dataclass
class PathConfig:
    """Output and cache directories."""
    project_root: str = field(default_factory=lambda: os.getcwd())
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    
    # Derived paths
    @property
    def round1_dir(self) -> str:
        return os.path.join(self.output_dir, "round1")
    
    @property
    def round2_dir(self) -> str:
        return os.path.join(self.output_dir, "round2")
    
    @property
    def eval_dir(self) -> str:
        return os.path.join(self.output_dir, "evaluation")
    
    @property
    def error_analysis_dir(self) -> str:
        return os.path.join(self.output_dir, "error_analysis")
    
    @property
    def augmented_data_dir(self) -> str:
        return os.path.join(self.output_dir, "augmented_data")
    
    @property
    def figures_dir(self) -> str:
        return os.path.join(self.output_dir, "figures")
    
    def create_dirs(self):
        """Create all output directories."""
        for d in [self.output_dir, self.round1_dir, self.round2_dir,
                  self.eval_dir, self.error_analysis_dir,
                  self.augmented_data_dir, self.figures_dir, self.cache_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


@dataclass
class DieselConfig:
    """Master configuration aggregating all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    taxonomy: ErrorTaxonomy = field(default_factory=ErrorTaxonomy)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    def __post_init__(self):
        self.paths.create_dirs()


def get_config(**overrides) -> DieselConfig:
    """
    Factory function to get config with optional overrides.
    
    Usage:
        cfg = get_config()  # defaults
        cfg = get_config(training=TrainingConfig(num_train_epochs=1))
    """
    config = DieselConfig(**overrides)
    return config


# Quick-access singleton for notebooks
DEFAULT_CONFIG = None

def get_default_config() -> DieselConfig:
    global DEFAULT_CONFIG
    if DEFAULT_CONFIG is None:
        DEFAULT_CONFIG = get_config()
    return DEFAULT_CONFIG
