# Time-Series Foundation Model Fine-Tuning for Long-Term Glucose Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-purple)](https://github.com/huggingface/peft)

> **Fine-tuning IBM Granite TinyTimeMixer (TTM-R2) and Transformer-based models for long-term continuous glucose forecasting using LoRA, dual-head training, and personalised user embeddings.**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Key Features](#key-features)
- [Installation](#installation)
- [Data Format](#data-format)
- [Usage](#usage)
  - [Base TTM Pipeline](#1-base-ttm-pipeline)
  - [Optimised TTM Pipeline](#2-optimised-ttm-pipeline)
  - [TTM Pipeline with User Embeddings](#3-ttm-pipeline-with-user-embeddings)
  - [Transformer Baseline Comparison](#4-transformer-baseline-comparison)
- [Pipeline Variants](#pipeline-variants)
- [Loss Function](#loss-function)
- [Evaluation Metrics](#evaluation-metrics)
- [Configuration Reference](#configuration-reference)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [Licence](#licence)

---

## Overview

Long-term blood glucose prediction is a clinically critical task for diabetes management, particularly in the context of closed-loop insulin delivery systems. This repository provides a full end-to-end deep learning pipeline for **long-horizon glucose forecasting** (up to 720 time-steps / 6 hours at 5-minute CGM resolution) using:

1. **IBM Granite TinyTimeMixer (TTM-R2)** — a time-series foundation model pre-trained on large-scale temporal data, fine-tuned using **Parameter-Efficient Fine-Tuning (PEFT)** via Low-Rank Adaptation (LoRA).
2. **Dual-Head Training Strategy** — a two-pass curriculum: first a univariate pass on raw glucose values, followed by a multivariate pass incorporating clinical covariates (insulin-on-board, meal data, activity, time features).
3. **Personalised User Embeddings** — an optional per-subject residual embedding injected into the model's hidden state to adapt predictions to individual metabolic profiles.
4. **Transformer Baseline Suite** — TimesNet, Informer, and PatchTST models via the [Darts](https://github.com/unit8co/darts) library for comparative benchmarking.

---

## Architecture

```
                      ┌─────────────────────────────────────────────────┐
                      │          TTM Fine-Tuning Pipeline                │
                      │                                                   │
  CGM Input           │  ┌──────────────────────────────────────────┐   │
  (LBORRES) ─────────►│  │    IBM Granite TTM-R2 (Backbone)          │   │
                      │  │    [LoRA adapted: r=4/32, α=8/32]        │   │
  Covariates ────────►│  │    context_length: 512/1024/1536         │   │
  (NETIOB, MLDOSE,    │  │    prediction_length: 96/192/336/720     │   │
   SEX, PUMP, etc.)   │  └──────────────┬───────────────────────────┘   │
                      │                 │ backbone_hidden_state          │
  User ID ───────────►│  ┌──────────────▼───────────────────────────┐   │
  (User Embedding)    │  │    DualHeadTTM Wrapper                    │   │
                      │  │  ┌─────────────┐  ┌──────────────────┐   │   │
                      │  │  │ Uni Head    │  │ Multi Head       │   │   │
                      │  │  │ (Pass 1)    │  │ (Pass 2)         │   │   │
                      │  │  │LayerNorm    │  │LayerNorm         │   │   │
                      │  │  │Dropout(0.1) │  │Dropout(0.1)      │   │   │
                      │  │  │Linear→1     │  │Linear→1          │   │   │
                      │  │  └─────────────┘  └──────────────────┘   │   │
                      │  └──────────────────────────────────────────┘   │
                      │                 │                                │
                      │     Predicted Glucose Sequence [B, T, 1]        │
                      └─────────────────────────────────────────────────┘
```

### Two-Pass Training Curriculum

| Pass | Head Used | Dataset | Purpose |
|------|-----------|---------|---------|
| Pass 1 | `uni_head` | Univariate (glucose only) | Learn temporal glucose dynamics |
| Pass 2 | `multi_head` | Multivariate (glucose + covariates) | Incorporate clinical context |

---

## Repository Structure

```
├── TTM_Gluco_Finetuning_Pipeline.py                 # Base TTM pipeline (v1.4)
├── TTM_Gluco_Finetuning_Pipeline_Optimised.py       # GPU-optimised pipeline (v2.0)
├── TTM_Gluco_Finetuning_Pipeline_with_user_embedding.py  # Personalised pipeline (v3.1)
├── transformer_based_glucose_forecasting.py          # Transformer baseline (TimesNet/Informer/PatchTST)
├── earlyStopping_class.py                            # Custom early stopping callback
├── glucose_forecast_data_agg.py                      # Data aggregation and preprocessing
├── glucose_forecast_utils.py                         # Shared utility functions
├── optimal_finetuning_lr.py                          # Automatic learning rate finder (OptimalLRFinder)
└── .gitignore
```

---

## Key Features

- **Foundation Model Fine-Tuning**: Leverages `ibm-granite/granite-timeseries-ttm-r2` with PEFT/LoRA for parameter-efficient adaptation to clinical CGM data.
- **Dual-Head Architecture**: Separate univariate and multivariate prediction heads with a two-pass training curriculum, enabling progressive domain adaptation.
- **Personalised User Embeddings** (v3.1): Per-subject `nn.Embedding` layer projected and residually fused into the backbone hidden state, capturing individual metabolic variability.
- **Clinically-Informed Loss Function**: Composite loss combining Huber loss (β=27.0) with a Clarke Error Grid–inspired large-error penalty (threshold: 10–12 mg/dL), weighted 0.7:0.3.
- **Automatic Learning Rate Discovery**: `OptimalLRFinder` utility with exponential sweep and configurable stop factor.
- **Auto Batch-Size Optimisation**: Probes GPU memory to find the largest feasible batch size at runtime.
- **Memory-Efficient Training**: Non-blocking CUDA transfers, `cudnn.benchmark`, fused AdamW (`adamw_torch_fused`), pin memory, persistent workers, and gradient accumulation.
- **Comprehensive Callbacks**: `LossLoggerCallback`, `MemoryMonitorCallback`, `EarlyStoppingCallback` with training/validation loss curve export.
- **Per-User Evaluation Plots**: Automatically generates individual glucose forecast plots per subject (USUBJID).
- **Transformer Baseline Suite**: TimesNet, Informer, and PatchTST via Darts for comparative benchmarking.

---

## Installation

### Requirements

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate
pip install tsfm-public  # IBM TSFM public toolkit
pip install scikit-learn pandas numpy matplotlib psutil

# For transformer baseline comparisons
pip install darts
```

### Recommended Environment

```bash
conda create -n glucose-ttm python=3.10
conda activate glucose-ttm
pip install -r requirements.txt  # (see above)
```

> **GPU**: A CUDA-capable GPU is strongly recommended. The codebase is configured for CUDA by default and will raise errors if CUDA is unavailable unless the device assignment is modified.

---

## Data Format

The pipeline expects a CSV file with the following minimum schema:

| Column | Type | Description |
|--------|------|-------------|
| `USUBJID` | str/int | Unique subject identifier |
| `LBDTC` | datetime | Timestamp of CGM reading |
| `LBORRES` | float | CGM glucose value (mg/dL or mmol/L) |
| `NETIOB` | float | Net insulin-on-board |
| `MLDOSE` | float | Meal dose |
| `MLCAT` | str | Meal category |
| `SEX` | int | Binary sex covariate |
| `PUMP` | int | Insulin pump flag |
| `ACTARMCD` | str | Treatment arm code |
| `DAY_NIGHT` | int | Day/night indicator |
| `SIN_H`, `COS_H` | float | Cyclical hour encodings |
| `SIN_DOW`, `COS_DOW` | float | Cyclical day-of-week encodings |
| `EXCINTSY`, `SNKBEFEX` | int | Exercise/snack covariates |
| `PLNEXDUR` | float | Planned exercise duration |
| `RESQCARB` | float | Rescue carbohydrates |
| `LBORRES_ZSCORE_24` | float | 24-hour glucose Z-score |
| `NETIOB_ZSCORE_24` | float | 24-hour IOB Z-score |

Data must be sorted by `USUBJID` and `LBDTC`. Missing values are handled via forward/backward fill for multivariate features.

---

## Usage

### 1. Base TTM Pipeline

```python
from TTM_Gluco_Finetuning_Pipeline import run_finetuning_inference

# Configure inside the function:
# DATA_FILE = "glucose_forecast_data.csv"
# CONTEXT_LENGTH = 512
# PREDICTION_LENGTH = 96

run_finetuning_inference()
```

### 2. Optimised TTM Pipeline

```python
from TTM_Gluco_Finetuning_Pipeline_Optimised import OptimizedTTMGlucosePipeline, OptimizedGlucoseDataset

pipeline = OptimizedTTMGlucosePipeline(
    train_uni=train_uni_dataset,
    val_uni=val_uni_dataset,
    train_multi=train_multi_dataset,
    val_multi=val_multi_dataset,
    save_dir="./outputs/v2",
    context_length=512,
    prediction_length=96,
    batch_size=64,
    feature_cols=feature_cols
)

pipeline.run_pass1(epochs=4, lr=5e-4)
pipeline.run_pass2(epochs=3, lr=5e-5)

pipeline.run_comprehensive_evaluation(
    test_set=test_uni_dataset,
    use_aux=False,
    title="Univariate Test",
    save_plot_dir="./outputs/v2/plots/uni",
    save_metrics_csv="./outputs/v2/metrics/uni_metrics.csv"
)
```

### 3. TTM Pipeline with User Embeddings

```python
from TTM_Gluco_Finetuning_Pipeline_with_user_embedding import (
    OptimizedTTMGlucosePipeline, OptimizedGlucoseDataset
)

# Build user index mapping first
all_user_ids = df['USUBJID'].unique()
user_id_to_index = {uid: i for i, uid in enumerate(all_user_ids)}
n_users = len(all_user_ids)

train_uni = OptimizedGlucoseDataset(
    df_train, CONTEXT_LENGTH, PREDICTION_LENGTH,
    user_id_to_index=user_id_to_index
)

pipeline = OptimizedTTMGlucosePipeline(
    train_uni=train_uni, val_uni=val_uni,
    train_multi=train_multi, val_multi=val_multi,
    save_dir="./outputs/v3",
    context_length=512, prediction_length=96,
    batch_size=64, n_users=n_users
)

pipeline.run_pass1(epochs=4, lr=5e-4)
pipeline.run_pass2(epochs=3, lr=5e-5)
```

### 4. Transformer Baseline Comparison

```python
# Configure data file path inside transformer_based_glucose_forecasting.py:
# df = pd.read_csv("glucose_netiob_data.csv")
# Runs TimesNet, Informer, and PatchTST sequentially for USUBJID='1014'

python transformer_based_glucose_forecasting.py
```

---

## Pipeline Variants

| Feature | v1.4 (Base) | v2.0 (Optimised) | v3.1 (User Embedding) |
|---------|-------------|-------------------|-----------------------|
| LoRA rank / alpha | 4 / 8 | 32 / 32 | 32 / 32 |
| LoRA init | default | PiSSA | PiSSA |
| LR scheduling | OneCycleLR | Cosine + OneCycleLR | Cosine + OneCycleLR |
| Auto LR finder | ✗ | ✓ | ✓ |
| Auto batch size | ✗ | ✓ | ✓ |
| User embeddings | ✗ | ✗ | ✓ (dim=32) |
| Early stopping | optional | ✓ (patience=3) | ✓ (patience=3) |
| R² metric | ✗ | ✓ | ✓ |
| Fused AdamW | ✗ | ✓ | ✓ |
| CuDNN benchmark | ✗ | ✓ | ✓ |

---

## Loss Function

The training objective is a clinically-motivated composite loss:

\[
\mathcal{L} = 0.7 \cdot \mathcal{L}_{\text{Huber}}(\beta=27) + 0.3 \cdot \mathcal{L}_{\text{Clarke}}
\]

Where:

- \(\mathcal{L}_{\text{Huber}}\) is the smooth L1 (Huber) loss with `β = 27.0`, tuned to the physiological glucose range.
- \(\mathcal{L}_{\text{Clarke}}\) is the mean absolute error penalised only on predictions with absolute error ≥ 10–12 mg/dL, inspired by the Clarke Error Grid Zone A/B boundary.

This formulation prioritises stable optimisation whilst explicitly penalising clinically dangerous large-deviation predictions.

---

## Evaluation Metrics

All pipelines report the following metrics at test time:

| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error (%) |
| R² | Coefficient of Determination (v2.0+ only) |

Per-user forecast plots (`.png`) and aggregated metrics (`.csv`) are automatically saved to the configured output directories.

---

## Configuration Reference

Key hyperparameters across all pipeline variants:

```python
CONTEXT_LENGTH   = 512     # Options: 512, 768, 1024, 1536
PREDICTION_LENGTH = 96     # Options: 96, 192, 336, 720
BATCH_SIZE       = 64
LR_PASS1         = 5e-4    # Univariate pass learning rate
LR_PASS2         = 5e-5    # Multivariate pass learning rate
EPOCHS_PASS1     = 4
EPOCHS_PASS2     = 3
LORA_RANK        = 4       # v1.4: 4 | v2.0+: 32
LORA_ALPHA       = 8       # v1.4: 8 | v2.0+: 32
GRAD_ACCUM_STEPS = 4
USER_EMB_DIM     = 32      # v3.1 only
```

---

## Acknowledgements

- **IBM Research** for the [Granite TinyTimeMixer (TTM-R2)](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) foundation model.
- **Hugging Face** for the `transformers`, `peft`, and `accelerate` libraries.
- **IBM TSFM** for the `tsfm_public` toolkit.
- **Unit8** for the [Darts](https://github.com/unit8co/darts) time-series library used in the baseline comparisons.
- Clarke Error Grid methodology for guiding clinically-meaningful loss design.

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{solanke2026ttmglucose,
  author       = {Solanke, Abiodun},
  title        = {Time-Series Foundation Model Fine-Tuning for Long-Term Glucose Prediction},
  year         = {2026},
  url          = {https://github.com/spyderweb-abdul/Time-series-Foundation-Model-for-Long-Term-Glucose-Prediction},
  note         = {GitHub repository}
}
```

---

## Licence

This project is licensed under the MIT Licence. See [LICENSE](LICENSE) for full details.
