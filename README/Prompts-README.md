## Prompt Design Rationale

This section explains **why these prompt types were chosen**, **why they are structured this way**, and **why these specific configuration parameters are the ones we allow the LLM to modify**.

---

## Core Design Philosophy

The prompt system is designed around three principles:

1. **Constrained Intelligence**
   - The LLM is allowed to reason
   - The LLM is NOT allowed to freely redesign systems
   - Hard boundaries prevent unsafe or unstable suggestions

2. **Metric-Driven Decision Making**
   - All prompts are tied to real training performance
   - No hallucinated optimization targets
   - No speculative architecture redesign

3. **Operational Safety**
   - Prevent training collapse
   - Prevent runaway compute cost
   - Prevent FL instability amplification
   - Ensure deterministic post-processing via JSON

---

## Why Role-Based Expert Prompts

Each prompt starts with a **domain-expert role**:

| Prompt Type | Role |
|---|---|
| Architecture (Centralized) | Neural Network Architect |
| Hyperparameters (Centralized) | Training Convergence Expert |
| Hyperparameters (FL) | Federated Systems Optimizer |
| Architecture (FL) | Federated Neural Architect |

### Why This Matters

This improves:

- Constraint adherence
- Structured reasoning
- Reduced generic LLM advice
- Domain-consistent recommendations

---

## Why We Chose These Configurations to Modify

### Architecture Parameters

#### HIDDEN1_SIZE
Controls early feature compression / representation width.

Why Modifiable:
- Direct impact on model capacity
- Low risk if bounded
- Strong effect on classification boundary quality

---

#### HIDDEN2_SIZE
Controls deeper abstraction expansion or refinement.

Why Modifiable:
- Controls second-stage representation learning
- Enables expansion–compression patterns
- Influences separability in final layers

---

#### DROPOUT_RATE
Controls generalization and overfitting resistance.

Why Modifiable:
- Safe continuous parameter
- Low catastrophic failure risk
- High regularization impact

---

### Hyperparameters (Centralized)

#### LEARNING_RATE
Primary convergence control knob.

Why Modifiable:
- Largest impact on training stability
- Needed for plateau escape or oscillation reduction

---

#### BATCH_SIZE
Controls gradient noise vs stability.

Why Modifiable:
- Impacts convergence smoothness
- Impacts GPU efficiency
- Impacts generalization behavior

---

#### EPOCHS
Controls local optimization depth.

Why Modifiable:
- Direct control over training completeness
- Risk bounded via strict caps

---

#### ROUNDS
Controls total training duration and convergence window.

Why Modifiable:
- Useful when undertraining detected
- Prevents premature stopping

---

## Why Some Parameters Are Locked

### INPUT_SIZE
Fixed by dataset schema.

Changing this breaks:
- Model weight compatibility
- Data pipeline alignment
- Feature engineering assumptions

---

### OUTPUT_SIZE
Fixed by classification target space.

Changing this breaks:
- Loss function assumptions
- Evaluation metrics
- Dataset label mapping

---

### Optimizer Type
Locked to Adam.

Reason:
- Changing optimizer mid-training destabilizes convergence
- Requires retuning LR schedules
- Breaks training continuity assumptions

---

### Loss Function
Locked to CrossEntropy.

Reason:
- Matched to classification objective
- Changing requires dataset relabeling logic

---

## Why Strict Constraint Blocks Exist

The LLM is not trusted to:

- Respect training cost automatically
- Understand FL stability automatically
- Avoid architecture explosion automatically

Therefore prompts include:

### Absolute Limits
Hard numeric safety bounds.

### Change Magnitude Rules
Prevents destabilizing late-stage training.

### Forbidden Actions
Prevents invalid or unsupported modifications.

---

## Why JSON Output Is Mandatory

Benefits:

- Deterministic parsing
- Safe automated config patching
- Easy validation layer
- Prevents prompt injection through text output

---

## Why Centralized and Federated Prompts Are Different

### Centralized Training

Focus:
- Pure convergence optimization
- Architecture capacity tuning
- Regularization balancing

---

### Federated Training

Focus:
- Client heterogeneity stability
- Communication cost control
- Aggregation robustness
- Drift minimization

---

## Why Federated Prompts Limit Changes

FL systems are fragile because:

- Clients have non-IID data
- Gradient updates are asynchronous
- Overfitting one client hurts global model

Therefore:

- Single parameter change rules
- Conservative architecture scaling
- Stability-driven decision logic

---

## Why Performance-Based Change Scaling Exists

We scale allowed change size by final accuracy because:

| Performance | Risk Tolerance |
|---|---|
| >99% | Very Low |
| 95–99% | Moderate |
| <95% | Higher |

This prevents:

- Over-optimization near convergence
- Catastrophic late-stage training collapse

---

## Why We Chose Prompt + Hard Constraint Hybrid

Pure LLM optimization is dangerous.

Pure rule-based optimization is rigid.

Hybrid approach provides:

- LLM reasoning for context
- Hard rules for safety enforcement

---

## Why These Prompts Are Considered “Best Fit”

They balance:

- Safety
- Flexibility
- Interpretability
- Automation compatibility
- FL-aware system design
- Deterministic downstream integration

---

## Future Prompt Evolution Strategy

Potential improvements include:

- Self-verification prompts
- Multi-step reasoning chains
- Confidence scoring outputs
- Multi-model voting
- Dataset-aware adaptive constraint scaling

---

## Summary

These prompts were designed to:

- Allow intelligent optimization
- Prevent unsafe architecture drift
- Preserve training stability
- Respect real-world deployment constraints
- Enable automated safe mid-training optimization
- Work across both centralized and federated training regimes

---
