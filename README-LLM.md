
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
Results + Config Snapshot ->  
LLM Prompt Generator ->  
Ollama Local LLM ->  
JSON Recommendation Parser ->  
Improved Config Generator  

---

## Requirements

### System
- Linux or macOS
- Python 3.9+
- Ollama

### Python Packages
- requests  
- torch  

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
| Centralized DNN | Standard training optimization |
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

---

# LLM Module File Reference and Prompt Documentation

## Overview

This document describes the internal structure of `llm.py`, including:

- File responsibilities
- Public API entry point
- Internal subsystems
- Prompt generation logic
- Function reference with internal hyperlinks
- Prompt design philosophy

---

## Public API

### Primary Entry Point

**Only Supported External Call**

```
from llm import llm_mid_training_update
```

All other functions and classes are internal and subject to change.

---

## File Responsibilities

`llm.py` is responsible for:

- Managing Ollama lifecycle (install, start, model pull)
- Generating constraint-safe LLM prompts
- Running architecture and hyperparameter analysis
- Parsing strict JSON outputs
- Generating safe config updates
- Supporting centralized and federated learning workflows
- Enabling mid-training optimization injection

---

## Internal System Architecture

```
Training Loop
    ↓
llm_mid_training_update()
    ↓
Ollama Validation + Startup
    ↓
ModelImprover Initialization
    ↓
Prompt Generation
    ↓
LLM Query
    ↓
JSON Parsing + Validation
    ↓
Safe Config Update
```

---

## Function Reference

### Environment and Ollama Management

#### [`is_ollama_installed`](./llm.py#L35)
Checks if Ollama binary exists in system PATH.

#### [`install_ollama`](./llm.py#L52)
Installs Ollama automatically (Linux supported, others manual).

#### [`ensure_ollama_running`](./llm.py#L92)
Starts Ollama server if not already running.

#### [`check_model_exists`](./llm.py#L125)
Verifies model availability in Ollama registry.

#### [`pull_model`](./llm.py#L150)
Downloads model into Ollama runtime.

#### [`ask_llm`](./llm.py#L190)
Sends prompt to Ollama API and returns the response.

---

### Model Wrapper

#### [`Net`](./llm.py#L197)
Lightweight PyTorch wrapper for compatibility with configs.

---

### Core Engine

#### [`ModelImprover`](./llm.py#L222)
Central analysis engine.

Responsibilities:

- Prompt construction
- Metric extraction
- LLM querying
- JSON validation
- Config mutation
- Result persistence

---

## Prompt Generation Functions

---

### Centralized Training Prompts

#### [`create_architecture_cent_prompt`](./llm.py#L259)

Purpose:
- Optimize neural network structure
- Enforce parameter bounds
- Prevent over-expansion
- Respect dataset constraints

Key Constraints:
- Hidden layer size limits
- Dropout range enforcement
- Total parameter ceiling
- Change magnitude rules

---

#### [`create_hyperparameter_cent_prompt`](./llm.py#L349)

Purpose:
- Optimize convergence speed
- Prevent unstable training
- Control overfitting risk

Key Constraints:
- Learning rate bounds
- Batch size limits
- Epoch limits
- Training cost caps

---

### Federated Learning Prompts

#### [`create_hyperparameter_fl_prompt`](./llm.py#L470)

Purpose:
- Optimize FedAvg convergence
- Control client drift
- Stabilize aggregation

Focus Areas:
- Client accuracy variance
- Global vs local gap
- Communication efficiency
- Update stability

---

#### [`create_architecture_fl_prompt`](./llm.py#L579)

Purpose:
- Safe FL architecture evolution
- Prevent client instability
- Control communication overhead

Rules:
- Single layer change max
- Strict parameter ceilings
- Heterogeneity-aware decisions

---

## JSON Parsing Layer

### [`parse_json_from_response`](./llm.py#L676)

Responsibilities:

- Remove markdown wrappers
- Extract valid JSON blocks
- Prioritize final JSON object
- Validate structure integrity

---

## Analysis Execution

### [`analyze_architecture`](./llm.py#L719)

Runs:
- Architecture prompt generation
- LLM inference
- JSON extraction
- Recommendation validation

---

### [`analyze_hyperparameters`](./llm.py#L756)

Runs:
- Hyperparameter prompt generation
- LLM inference
- JSON extraction
- Recommendation validation

---

## Config Mutation

### [`generate_improved_config`](./llm.py#L801)

Rules:

Centralized:
- Architecture + hyperparameters allowed

Federated:
- Hyperparameters only (server-safe)

---

## Persistence Layer

### [`save_results`](./llm.py#L835)

Outputs:

```
centralized_output/
 ├ llm_architecture_analysis.txt
 ├ llm_hyperparameter_analysis.txt
 ├ config_comparison.txt

config_v2_improved.json
```

---

## Pipeline Orchestration

### [`run_full_analysis`](./llm.py#L888)

Full sequence:

1. Architecture analysis
2. Hyperparameter analysis
3. Config generation
4. Result persistence

---

## Training Integration Hook

### [`llm_mid_training_update`](./llm.py#L968)

Primary runtime integration.

Behavior:

- Triggers only at configured training round
- Performs full safety validation
- Returns safe config update or None

---

## CLI Entry Point

### [`main`](./llm.py#L1034)

Supports standalone execution for debugging and batch analysis.

Not intended for production training integration.

---

## Prompt Design Philosophy

All prompts enforce:

### Hard Constraints
- Absolute parameter limits
- Safe numeric ranges
- Architecture invariants

### Soft Constraints
- Performance-aware scaling
- Conservative tuning near convergence
- Cost-aware training changes

### Safety Rejection Rules

Outputs are rejected if:

- Multiple conflicting changes suggested
- Values outside safe bounds
- Structural violations detected

---

## Federated Learning Safety Model

Special Handling Includes:

- Client heterogeneity awareness
- Communication cost sensitivity
- Single-parameter change rule
- Aggregation stability prioritization

---

## Internal Stability Guarantees

The system is designed to:

- Never blindly trust LLM output
- Enforce deterministic JSON extraction
- Validate before config mutation
- Fail safe (return None) instead of unsafe update

---

## Intended Usage Pattern

```
Training Loop
    ↓
Collect Metrics Snapshot
    ↓
Call llm_mid_training_update()
    ↓
Apply Safe Config Patch
    ↓
Continue Training
```

---

## Non-Goals

This system intentionally does NOT:

- Perform gradient-level tuning
- Modify optimizer logic
- Perform client-specific tuning in FL
- Allow unconstrained architecture mutation

---

## Versioning Philosophy

Stable:
- llm_mid_training_update()

Unstable / Internal:
- All other functions
- Prompt formats
- Constraint thresholds
- Output schemas (internal only)

---

## Future Expansion Areas

Potential future additions:

- Adaptive trigger round selection
- Multi-stage prompt chaining
- Client clustering aware FL tuning
- Confidence-weighted recommendation merging
- Prompt self-verification loops

---

