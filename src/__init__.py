# DIESEL: Diagnosing Inaccuracies in Efficient SQL via Error-Taxonomy-Guided Learning
"""
Core library for the DIESEL research project.

Modules:
    config          — Centralized hyperparameters and paths
    data_loader     — Spider dataset loading, schema serialization, prompt formatting
    model_loader    — QLoRA model and tokenizer setup
    train           — SFTTrainer-based fine-tuning
    evaluate        — Execution-accuracy evaluation
    error_analyzer  — 6-category SQL error taxonomy classifier
    augmentor       — Taxonomy-Guided Data Augmentation (TGDA)
    inference       — Interactive inference
    utils           — Helper functions
"""

# Lazy imports — individual modules are imported where needed
# (avoids cascading failures if GPU dependencies aren't installed yet)
