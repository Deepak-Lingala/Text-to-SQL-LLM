# DIESEL: Diagnosing Inaccuracies in Efficient SQL via Error-Taxonomy-Guided Learning

> **A novel approach to improve small LLM Text-to-SQL performance through error taxonomy profiling and taxonomy-guided data augmentation.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-Llama%203.1%208B-yellow.svg)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

---

## Abstract

Text-to-SQL systems powered by large language models (LLMs) have seen rapid progress, yet most advances focus on large proprietary models (GPT-4, Claude) and inference-time error correction. We present **DIESEL**, a training-time approach that systematically diagnoses **why** a fine-tuned small LLM fails on specific SQL queries, then uses that diagnosis to construct **targeted training data** that addresses the weakest error categories. 

We fine-tune **Llama-3.1-8B-Instruct** with QLoRA on the Spider benchmark and introduce three contributions:

1. **A 6-category SQL error taxonomy** applied to profile error distributions before and after fine-tuning, revealing which error types persist despite supervised training.
2. **Taxonomy-Guided Data Augmentation (TGDA)** — a method that constructs synthetic training examples targeting the residual error categories for a second round of fine-tuning.
3. **A Difficulty × Error-Type analysis matrix** that pinpoints exactly where small models fail, providing actionable guidance for the community.

All experiments run on a single NVIDIA L4 GPU (24 GB), demonstrating that meaningful research can be conducted on commodity cloud hardware.

---

## Research Contributions

| Contribution | What's Novel |
|---|---|
| **Error Taxonomy for Fine-Tuned Small LLMs** | First systematic study of how error distributions shift from base → fine-tuned in an 8B-parameter model |
| **Taxonomy-Guided Data Augmentation** | Training-time correction (vs. existing inference-time approaches like DIN-SQL, TS-SQL, ReFoRCE) |
| **Difficulty × Error Matrix** | 2D breakdown revealing failure modes that aggregate metrics miss |

---

## Project Structure

```
DIESEL/
├── src/                          # Core library
│   ├── config.py                 # Centralized hyperparameters
│   ├── data_loader.py            # Spider dataset loading & prompt formatting
│   ├── model_loader.py           # QLoRA model setup
│   ├── train.py                  # SFTTrainer fine-tuning
│   ├── evaluate.py               # Execution accuracy evaluation
│   ├── error_analyzer.py         # 6-category error taxonomy (Novel)
│   ├── augmentor.py              # TGDA augmentation (Novel)
│   ├── inference.py              # Interactive inference
│   └── utils.py                  # SQL utilities & statistical tests
├── notebooks/                    # Colab-ready experiment scripts
│   ├── 01_eda.py                 # Exploratory data analysis
│   ├── 02_train_round1.py        # Round 1 QLoRA fine-tuning
│   ├── 03_error_analysis.py      # Error taxonomy analysis (Novel)
│   ├── 04_augment_and_train_round2.py  # TGDA + Round 2 training (Novel)
│   └── 05_final_evaluation.py    # Three-way comparison & figures
├── paper/
│   └── manuscript.md             # Full research manuscript
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Environment Setup (Google Colab)

```python
# In Colab: Runtime → Change runtime type → L4 GPU
!pip install -q torch transformers peft trl bitsandbytes
!pip install -q accelerate datasets sqlparse sqlglot
!pip install -q wandb matplotlib seaborn scipy scikit-learn python-dotenv tqdm

# HuggingFace login (model is gated)
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")
```

### 2. Run the Pipeline

```bash
# Step 1: Exploratory Data Analysis
python notebooks/01_eda.py

# Step 2: Round 1 Fine-Tuning
python notebooks/02_train_round1.py

# Step 3: Error Analysis (produces error taxonomy profile)
python notebooks/03_error_analysis.py \
    --adapter outputs/round1/final_adapter \
    --spider_db_dir /path/to/spider/database

# Step 4: TGDA Augmentation + Round 2 Training
python notebooks/04_augment_and_train_round2.py \
    --error_analysis outputs/error_analysis/error_analysis_round1_finetuned.json

# Step 5: Final Three-Way Evaluation
python notebooks/05_final_evaluation.py \
    --round1_adapter outputs/round1/final_adapter \
    --round2_adapter outputs/round2/final_adapter \
    --spider_db_dir /path/to/spider/database
```

### 3. Spider Database Setup

```bash
# Download Spider databases
wget https://drive.google.com/uc?id=1iRDVHLr6a7DRyG4fdEUN8BFxv_AAGDVB -O spider.zip
unzip spider.zip -d spider_databases/
```

---

## Error Taxonomy

| Code | Category | Description | Example |
|------|----------|-------------|---------|
| E1 | Schema Linking | Wrong table/column, hallucinated columns | Using `student.name` instead of `student.sname` |
| E2 | JOIN Errors | Missing/wrong join, unnecessary tables | Missing `JOIN courses ON ...` |
| E3 | Aggregation | Wrong aggregate, missing GROUP BY | `COUNT(*)` instead of `SUM(credits)` |
| E4 | Filter/Condition | Wrong WHERE, operators, values | `age > 20` instead of `age >= 21` |
| E5 | Nesting/Subquery | Missing subquery, wrong IN/EXISTS | Flat query where nested required |
| E6 | Syntax/Format | Unparseable SQL, extra text | Model outputs explanation instead of SQL |

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| Quantization | 4-bit NF4 (double quantization) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-4 (Round 1), 1e-4 (Round 2) |
| Epochs | 3 (Round 1), 2 (Round 2) |
| Batch size | 4 × 4 gradient accumulation = 16 |
| Max sequence length | 1024 |
| Optimizer | Paged AdamW 32-bit |

---

## Citation

```bibtex
@article{diesel2026,
  title={DIESEL: Diagnosing Inaccuracies in Efficient SQL via Error-Taxonomy-Guided Learning},
  author={[Your Name]},
  year={2026},
  note={Under review}
}
```

---

## License

This project is licensed under the MIT License.
