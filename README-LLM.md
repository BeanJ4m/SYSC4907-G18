# LLM-Driven Model Optimization Engine

## Overview

This project provides an **LLM-assisted optimization pipeline** for improving machine learning models during or after training. It integrates with **Ollama-hosted local LLMs** to:

- Analyze model architecture
- Analyze training hyperparameters
- Generate safe improvement recommendations
- Support **Centralized** and **Federated Learning (FL)** workflows
- Enable **mid-training adaptive optimization**

The system is designed for **safe, constraint-aware automated tuning**, especially for:

- Network intrusion detection models  
- Federated learning environments  
- TabTransformer or DNN architectures  

---

## Core Features

- Local LLM inference via Ollama  
- Constraint-safe architecture tuning  
- FL-aware optimization logic  
- Mid-training dynamic config updates  
- JSON-validated LLM outputs  
- Fully automated analysis pipeline  
- CLI + programmatic integration  

---

## Architecture

Training Pipeline ->
Results + Config Snapshot->
LLM Prompt Generator->
Ollama Local LLM->
JSON Recommendation Parser->
Improved Config Generator

---

## Requirements

### System
- Linux or macOS
- Python 3.9+
- Ollama

### Python Packages
#### requests
#### torch

---

## Installation

### Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### Install Python Dependencies
```bash
pip install requests torch
```

---

## Usage Model

This module is **not intended to be executed directly**.

The only supported integration method is importing and calling:

```python
from llm import llm_mid_training_update
```

All internal logic, model setup, Ollama management, prompt generation, and safety enforcement are handled automatically.

Users should **not** call any other function or class inside `llm.py`.

---

## Mid-Training Integration

### Import
```python
from llm import llm_mid_training_update
```

### Example Usage
```python
new_config = llm_mid_training_update(
    model_path="model.pth",
    config_path="config.json",
    results_snapshot=round_metrics,
    round_idx=current_round,
    trigger_round=20
)
```

---

## Behavior

When triggered, the system will automatically:

- Verify Ollama installation  
- Start Ollama server if needed  
- Pull required LLM model if missing  
- Analyze training metrics  
- Generate safe architecture and/or hyperparameter recommendations  
- Return an updated configuration dictionary  

If no safe update is possible, the function returns:

```python
None
```

---

## Output Files

Generated automatically during analysis:

```
centralized_output/
 ├ llm_architecture_analysis.txt
 ├ llm_hyperparameter_analysis.txt
 ├ config_comparison.txt

config_v2_improved.json
```

---

## Supported Training Modes

| Mode | Description |
|---|---|
| Centralized TT | Standard training optimization |
| Centralized DNN| Standard training optimization |
| Federated DNN | FedAvg-safe tuning |
| Federated TabTransformer | Net2Net widening compatible |

---

## Safety Guarantees

The system enforces:

- Parameter range limits  
- Change magnitude limits  
- Federated learning stability constraints  
- Training cost limits  
- Single-parameter FL hyperparameter tuning when required  

Unsafe or invalid LLM outputs are automatically discarded.

---

## Example Workflow

```
Train Model
    ↓
Save Metrics Snapshot
    ↓
Call llm_mid_training_update()
    ↓
Receive Updated Config (or None)
    ↓
Continue Training
```

---

## Design Goals

- Safe automated optimization  
- Local LLM inference (no cloud dependency)  
- Federated learning aware optimization logic  
- Deterministic JSON-based updates  
- Drop-in training pipeline integration  
