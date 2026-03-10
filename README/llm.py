#!/usr/bin/env python3
"""
LLM-Based Model Improvement Analyzer
Auto-installs Ollama and model, then analyzes trained model
WITH STRICT CONSTRAINTS to prevent bad suggestions
"""

import json
import pickle
import re
import subprocess
import sys
import os
from pathlib import Path
import time
import requests
import platform




# Hardcoded project paths (DOES NOT AUTO-DETECT FEEL FREE TO CHANGE)

BASE_DIR = Path("/teamspace/studios/this_studio")
DATA_DIR = BASE_DIR / "data"
ITERATION_MODELS_DIR = BASE_DIR / "iteration_models"
OUTPUT_DIR = BASE_DIR / "centralized_output"
CONFIG_PATH = BASE_DIR / "config.json"

# ==========================================================
#  Ollama installation + startup
# ==========================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MODEL_NAME = "gemma2"


def is_ollama_installed():
    """Check if ollama binary is installed."""
    try:
        result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
        return result.returncode == 0
           
    except Exception:
        return False

#  PUBLIC API FOR NOTEBOOK INTEGRATION

def _get_torch():
    import torch
    return torch


def install_ollama():
    """Install Ollama binary."""
    print(" Installing Ollama...")
    
    system = platform.system()
    try:
        if system == "Linux":
            print("   Downloading and installing Ollama for Linux...")
            install_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
            result = subprocess.run(
                install_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print(" Ollama installed successfully")
                return True
            else:
                print(f" Installation failed: {result.stderr}")
                return False
                
        elif system == "Darwin":  # macOS
            print("   Please install Ollama manually:")
            print("   $ brew install ollama")
            print("   Or download from: https://ollama.com/download")
            return False
            
        else:
            print(f" Unsupported platform: {system}")
            print("   Please install Ollama manually from: https://ollama.com/download")
            return False
            
    except Exception as e:
        print(f" Installation error: {e}")
        return False


def ensure_ollama_running():
    """Start Ollama if not already running."""
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=2)
        if r.ok:
            print(" Ollama server already running.")
            return True
    except Exception:
        pass

    print(" Starting Ollama server...")
    log_file = "ollama_server.log"
    
    try:
        with open(log_file, "ab", buffering=0) as log:
            subprocess.Popen(
                ["ollama", "serve"], 
                stdout=log, 
                stderr=log, 
                start_new_session=True
            )
        
        for i in range(60):
            try:
                r = requests.get(OLLAMA_TAGS_URL, timeout=2)
                if r.ok:
                    print(" Ollama API is ready.")
                    return True
            except Exception:
                time.sleep(2)
        
        raise RuntimeError(" Ollama did not start in time. Check ollama_server.log")
        
    except FileNotFoundError:
        print(" Ollama binary not found in PATH")
        return False
    except Exception as e:
        print(f" Failed to start Ollama: {e}")
        return False


def check_model_exists(model_name=MODEL_NAME):
    """Check if model is already pulled."""
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=5)
        if r.ok:
            models = r.json().get("models", [])
            for model in models:
                if model_name in model.get("name", ""):
                    print(f" Model '{model_name}' is available.")
                    return True
        print(f"  Model '{model_name}' not found. Will pull...")
        return False
    except Exception as e:
        print(f"  Could not check models: {e}")
        return False


def pull_model(model_name=MODEL_NAME):
    """Pull the model if not available."""
    print(f" Pulling model '{model_name}'...")
    print("   This may take 2-5 minutes on first run...")
    
    try:
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        try:
            if process.stdout:
                for line in process.stdout:
                    print(f"   {line.strip()}")
        except Exception:
            pass
        
        process.wait()
        
        if process.returncode == 0:
            print(f" Model '{model_name}' ready.")
            return True
        else:
            print(f" Failed to pull model (exit code: {process.returncode})")
            return False
            
    except FileNotFoundError:
        print(" Ollama binary not found. Please install Ollama first.")
        return False
    except Exception as e:
        print(f" Error pulling model: {e}")
        return False


def ask_llm(prompt, model=MODEL_NAME):
    """Query Ollama API."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["response"]
    except Exception as e:
        raise Exception(f"Ollama query failed: {e}")

class Net:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, dropout_rate):
        torch = _get_torch()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden1_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden1_size, hidden2_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden2_size, output_size),
        )

    def load_state_dict(self, state):
        self.model.load_state_dict(state)

    def eval(self):
        self.model.eval()


# ==========================================================
#  Model Improvement Analyzer
# ==========================================================
class ModelImprover:
    """Analyzes training dynamics and generates improvement recommendations using LLM."""

    def __init__(self, model_path, config_path=str(CONFIG_PATH), results_path=None):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.results_path = Path(results_path) if results_path else None
        self.OUTPUT_DIR = OUTPUT_DIR

        # Load config
        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        # IMPORTANT:
        # We do NOT load or build the model.
        # LLM analysis is metric-driven, not weight-driven.
        self.model = None

        # Load results (optional)
        if self.results_path and self.results_path.exists():
            with open(self.results_path, "rb") as f:
                self.results = pickle.load(f)
        else:
            self.results = None
    
    def generate_llm_response(self, prompt):
        """Generate response from Ollama."""
        return ask_llm(prompt, model=MODEL_NAME)
    
    def create_architecture_cent_prompt(self):
        """Create prompt for architecture analysis with STRICT constraints."""
        
        # Get performance metrics
        if self.results and 'Centralized' in self.results:
            final_accuracy = self.results['Centralized']['Accuracy'][-1] * 100
            final_precision = self.results['Centralized']['Precision'][-1] * 100
            final_recall = self.results['Centralized']['Recall'][-1] * 100
            final_f1 = self.results['Centralized']['F1_Score'][-1] * 100
        else:
            final_accuracy = "N/A"
            final_precision = "N/A"
            final_recall = "N/A"
            final_f1 = "N/A"
        
        # Calculate current network size
        current_params = self.config['HIDDEN1_SIZE'] * self.config['HIDDEN2_SIZE']
        
        # Determine if performance is already excellent
        performance_status = "EXCELLENT - Use CONSERVATIVE changes only" if isinstance(final_accuracy, float) and final_accuracy > 99 else "Room for improvement"
        
        prompt = f"""You are a neural network architect specializing in classification tasks.

CURRENT ARCHITECTURE:
- Input: {self.config['INPUT_SIZE']} features (FIXED - cannot change)
- Hidden Layer 1: {self.config['HIDDEN1_SIZE']} neurons
- Hidden Layer 2: {self.config['HIDDEN2_SIZE']} neurons
- Output: {self.config['OUTPUT_SIZE']} classes (FIXED - cannot change)
- Activation: ReLU
- Dropout: {self.config['DROPOUT_RATE']}
- Current Network Size: {current_params} parameters

TASK: {self.config['OUTPUT_SIZE']}-class network intrusion detection

CURRENT PERFORMANCE:
- Accuracy: {final_accuracy}%
- Precision: {final_precision}%
- Recall: {final_recall}%
- F1 Score: {final_f1}%
- STATUS: {performance_status}

===== CRITICAL CONSTRAINTS - MUST OBEY =====

1. ABSOLUTE LIMITS (VIOLATION = INVALID):
   - HIDDEN1_SIZE: MUST be between 32 and 256 (current: {self.config['HIDDEN1_SIZE']})
   - HIDDEN2_SIZE: MUST be between 64 and 512 (current: {self.config['HIDDEN2_SIZE']})
   - DROPOUT_RATE: MUST be between 0.1 and 0.5 (current: {self.config['DROPOUT_RATE']})
   - Total parameters: MUST NOT exceed 200,000 (current: {current_params})

2. CHANGE MAGNITUDE RULES:
   - If accuracy > 99%: Changes MUST be ≤ 10% of current value
   - If accuracy > 95%: Changes MUST be ≤ 25% of current value
   - If accuracy < 95%: Changes can be up to 50% of current value

3. ARCHITECTURE PRINCIPLES:
   - HIDDEN1_SIZE should generally be ≤ INPUT_SIZE (avoid information bottleneck)
   - HIDDEN2_SIZE can be larger than HIDDEN1_SIZE (expand then compress pattern OK)
   - Avoid creating very deep/narrow networks (numerical instability)
   - Network should not be unnecessarily large for the task

4. WHAT YOU CANNOT CHANGE:
   - INPUT_SIZE (fixed by dataset)
   - OUTPUT_SIZE (fixed by number of classes)
   - Cannot add additional layers (code limitation)
   - Cannot change activation function

===== ANALYSIS TASK =====

Based on CURRENT performance of {final_accuracy}%, analyze if architecture is optimal.

RESPOND IN THIS EXACT FORMAT:

VERDICT: [KEEP or MODIFY]

REASONING:
[2-3 sentences explaining your decision based on current performance and architecture]

RECOMMENDATIONS:
- HIDDEN1_SIZE: {self.config['HIDDEN1_SIZE']} → [new value] (Reason: [brief explanation])
- HIDDEN2_SIZE: {self.config['HIDDEN2_SIZE']} → [new value] (Reason: [brief explanation])
- DROPOUT_RATE: {self.config['DROPOUT_RATE']} → [new value] (Reason: [brief explanation])

CONSTRAINT COMPLIANCE CHECK:
- HIDDEN1_SIZE in range [32, 256]: [YES/NO]
- HIDDEN2_SIZE in range [64, 512]: [YES/NO]
- DROPOUT_RATE in range [0.1, 0.5]: [YES/NO]
- Total params < 200,000: [YES/NO]
- Change magnitude appropriate: [YES/NO]

JSON_OUTPUT:
{{"HIDDEN1_SIZE": [integer between 32-256], "HIDDEN2_SIZE": [integer between 64-512], "DROPOUT_RATE": [float between 0.1-0.5]}}

CRITICAL: Do NOT include markdown code fences (```). Output raw JSON only.
CRITICAL: All values MUST be within the specified ranges or your output is INVALID.
"""
        return prompt
  
    def create_hyperparameter_cent_prompt(self):
        """Create prompt for hyperparameter analysis with STRICT constraints."""
        
        # Get training progression
        if self.results and 'Centralized' in self.results:
            accuracies = self.results['Centralized']['Accuracy']
            
            # Sample key rounds
            acc_round_1 = accuracies[0] * 100 if len(accuracies) > 0 else "N/A"
            acc_round_10 = accuracies[9] * 100 if len(accuracies) > 9 else "N/A"
            acc_round_20 = accuracies[19] * 100 if len(accuracies) > 19 else "N/A"
            acc_round_30 = accuracies[29] * 100 if len(accuracies) > 29 else "N/A"
            acc_round_40 = accuracies[-1] * 100 if len(accuracies) > 0 else "N/A"
            
            # Analyze trend
            if len(accuracies) >= 2:
                improvement = (accuracies[-1] - accuracies[0]) * 100
                trend = f"improving (+{improvement:.2f}%)" if improvement > 0 else f"declining ({improvement:.2f}%)"
            else:
                trend = "unknown"
        else:
            acc_round_1 = acc_round_10 = acc_round_20 = acc_round_30 = acc_round_40 = "N/A"
            trend = "unknown"
        
        # Determine performance status
        performance_status = "EXCELLENT - Use CONSERVATIVE changes only" if isinstance(acc_round_40, float) and acc_round_40 > 99 else "Room for improvement"
        
        prompt = f"""You are a deep learning optimization expert specializing in training convergence.

CURRENT HYPERPARAMETERS:
- Learning Rate: {self.config['LEARNING_RATE']}
- Batch Size: {self.config['BATCH_SIZE']}
- Epochs per round: {self.config['EPOCHS']}
- Total rounds: {self.config['ROUNDS']}
- Optimizer: Adam (fixed)
- Loss: CrossEntropyLoss (fixed)

TRAINING PROGRESSION:
- Round 1: {acc_round_1}% accuracy
- Round 10: {acc_round_10}% accuracy
- Round 20: {acc_round_20}% accuracy
- Round 30: {acc_round_30}% accuracy
- Round 40: {acc_round_40}% accuracy
- Overall trend: {trend}
- STATUS: {performance_status}

===== CRITICAL CONSTRAINTS - MUST OBEY =====

1. ABSOLUTE LIMITS (VIOLATION = INVALID):
   - LEARNING_RATE: MUST be between 1e-05 and 1e-03 (current: {self.config['LEARNING_RATE']})
   - BATCH_SIZE: MUST be between 128 and 512 (current: {self.config['BATCH_SIZE']})
   - EPOCHS: MUST be between 1 and 5 (current: {self.config['EPOCHS']})
   - ROUNDS: Keep at {self.config['ROUNDS']} for consistency (or suggest KEEP)

2. CHANGE MAGNITUDE RULES:
   - If final accuracy > 99%: Changes MUST be ≤ 20% of current value
   - If final accuracy > 95%: Changes MUST be ≤ 30% of current value
   - If final accuracy < 95%: Changes can be up to 50% of current value

3. BAD COMBINATIONS TO AVOID:
   - NEVER: Large batch (>1024) + Tiny LR (<5e-05) = Poor convergence
   - NEVER: Very small batch (<64) + Large LR (>1e-03) = Unstable training
   - NEVER: Too many epochs (>5) = Overfitting risk

4. TRAINING TIME CONSTRAINT:
   - Total training time should not increase more than 2x
   - Formula: BATCH_SIZE × EPOCHS affects time
   - Doubling EPOCHS = ~2x training time

5. WHAT YOU CANNOT CHANGE:
   - Optimizer (must use Adam)
   - Loss function (must use CrossEntropyLoss)

===== ANALYSIS TASK =====

Based on training progression showing final accuracy of {acc_round_40}%, analyze if hyperparameters are optimal.

RESPOND IN THIS EXACT FORMAT:

VERDICT: [OPTIMAL or SUBOPTIMAL]

REASONING:
[2-3 sentences explaining convergence behavior and why changes are/aren't needed]

RECOMMENDATIONS:
- LEARNING_RATE: {self.config['LEARNING_RATE']} → [new value] (Reason: [brief explanation])
- BATCH_SIZE: {self.config['BATCH_SIZE']} → [new value] (Reason: [brief explanation])
- EPOCHS: {self.config['EPOCHS']} → [new value] (Reason: [brief explanation])
- ROUNDS: {self.config['ROUNDS']} → [KEEP or new value] (Reason: [brief explanation])

CONSTRAINT COMPLIANCE CHECK:
- LEARNING_RATE in range [1e-05, 1e-03]: [YES/NO]
- BATCH_SIZE in range [128, 512]: [YES/NO]
- EPOCHS in range [1, 5]: [YES/NO]
- No bad combinations: [YES/NO]
- Training time reasonable: [YES/NO]

JSON_OUTPUT:
{{"LEARNING_RATE": [float between 1e-05 and 1e-03], "BATCH_SIZE": [integer between 128-512], "EPOCHS": [integer between 1-5], "ROUNDS": [integer or {self.config['ROUNDS']}]}}

CRITICAL: Do NOT include markdown code fences (```). Output raw JSON only.
CRITICAL: All values MUST be within the specified ranges or your output is INVALID.
CRITICAL: Use scientific notation for LEARNING_RATE (e.g., 5e-05, not 0.00005).
"""
        return prompt
    
    def create_hyperparameter_fl_prompt(self):
        """
        Create prompt for FEDERATED LEARNING hyperparameter analysis.
        Focuses on coordination dynamics, not client-side optimization.
        """

        # Expected FL metrics structure:
        # self.results["Federated"] = {
        #   "global_accuracy": [...],
        #   "mean_client_accuracy": [...],
        #   "std_client_accuracy": [...],
        #   "min_client_accuracy": [...],
        #   "max_client_accuracy": [...],
        #   "participation_rate": [...]
        # }

        results = self.results or {}
        fl = results.get("Federated", {})


        global_acc = fl.get("global_accuracy", [])
        mean_acc   = fl.get("mean_client_accuracy", [])
        std_acc    = fl.get("std_client_accuracy", [])
        min_acc    = fl.get("min_client_accuracy", [])
        max_acc    = fl.get("max_client_accuracy", [])

        def pct(x):
            return round(x * 100, 2)

        global_final = pct(global_acc[-1]) if global_acc else "N/A"
        mean_final   = pct(mean_acc[-1]) if mean_acc else "N/A"
        std_final    = round(std_acc[-1], 2) if std_acc else "N/A"
        min_final    = pct(min_acc[-1]) if min_acc else "N/A"
        max_final    = pct(max_acc[-1]) if max_acc else "N/A"

        if len(global_acc) >= 3:
            delta = pct(global_acc[-1] - global_acc[-3])
            trend = f"improving (+{delta}%)" if delta > 0 else f"stagnant ({delta}%)"
        else:
            trend = "unknown"

        stability = "STABLE" if isinstance(std_final, float) and std_final < 5 else "HETEROGENEOUS"

        prompt = f"""
You are a FEDERATED LEARNING systems optimization expert.

This system uses standard FedAvg with server-controlled hyperparameters.
Clients use identical optimizers and hyperparameters.

===== CURRENT FL CONFIGURATION =====
- Total clients: {self.config['NUM_CLIENTS']}
- Communication rounds: {self.config['ROUNDS']}
- Local epochs per client: {self.config['EPOCHS']}
- Learning rate (client-side): {self.config['LEARNING_RATE']}
- Aggregation: FedAvg (fixed)

===== GLOBAL PERFORMANCE =====
- Final global accuracy: {global_final}%
- Accuracy trend (recent rounds): {trend}

===== CLIENT DISTRIBUTION (LAST ROUND) =====
- Mean client accuracy: {mean_final}%
- Std deviation: {std_final}
- Min client accuracy: {min_final}%
- Max client accuracy: {max_final}%

SYSTEM STABILITY ASSESSMENT: {stability}

===== CRITICAL CONSTRAINTS (MUST OBEY) =====

1. ABSOLUTE LIMITS (VIOLATION = INVALID):
- local_epochs (EPOCHS): MUST be between 1 and 5 (current: {self.config['EPOCHS']})
- ROUNDS: MUST be between 5 and 50 (current: {self.config['ROUNDS']})
- LEARNING_RATE: MUST be between 1e-05 and 1e-03 (current: {self.config['LEARNING_RATE']})

2. CHANGE MAGNITUDE RULES:
- You may change AT MOST ONE parameter
- Learning rate changes MUST be ≤ 30% of current value
- If std deviation ≥ 5:
  - Do NOT increase local_epochs
  - Prefer LEARNING_RATE reduction or KEEP
- If std deviation < 5:
  - local_epochs may change by ±1
- Do NOT increase total training cost by more than 2× overall

3. FORBIDDEN ACTIONS:
- Do NOT change batch size
- Do NOT change model architecture
- Do NOT change optimizer type
- Do NOT change aggregation algorithm (FedAvg only)

===== ANALYSIS TASK =====

Determine whether FL performance is limited by:
(A) insufficient local optimization
(B) unstable aggregation due to client heterogeneity
(C) unnecessary communication rounds

Recommend the SAFEST possible adjustment.

===== RESPONSE FORMAT (STRICT) =====

VERDICT: [STABLE or SUBOPTIMAL]

REASONING:
[2–3 sentences focused on FL convergence, aggregation stability, and client variance]

RECOMMENDATIONS:
- local_epochs (EPOCHS): {self.config['EPOCHS']} → [new value or KEEP]
- ROUNDS: {self.config['ROUNDS']} → [new value or KEEP]
- LEARNING_RATE: {self.config['LEARNING_RATE']} → [new value or KEEP]

CONSTRAINT CHECK:
- local_epochs in [1,5]: [YES/NO]
- ROUNDS in [5,50]: [YES/NO]
- LEARNING_RATE in [1e-05, 1e-03]: [YES/NO]
- Only one parameter changed: [YES/NO]
- FL-safe update: [YES/NO]

JSON_OUTPUT:
{{
  "EPOCHS": [integer 1–5 or {self.config['EPOCHS']}],
  "ROUNDS": [integer 5–50 or {self.config['ROUNDS']}],
  "LEARNING_RATE": [float between 1e-05 and 1e-03]
}}

CRITICAL:
- Output RAW JSON only
- NO markdown
- Use scientific notation for LEARNING_RATE
- Invalid or multiple changes invalidate the response
"""
        return prompt
    def create_architecture_fl_prompt(self):
        """
        Create FL-safe architecture optimization prompt with STRICT constraints.
        """

        # Extract FL metrics
        results = self.results or {}
        fl = results.get("Federated", {})


        global_acc = fl.get("global_accuracy", [])
        mean_acc   = fl.get("mean_client_accuracy", [])
        std_acc    = fl.get("std_client_accuracy", [])

        final_global_acc = global_acc[-1] * 100 if global_acc else "N/A"
        final_std = std_acc[-1] if std_acc else "N/A"

        # Current architecture
        h1 = self.config["HIDDEN1_SIZE"]
        h2 = self.config["HIDDEN2_SIZE"]
        input_size = self.config["INPUT_SIZE"]
        output_size = self.config["OUTPUT_SIZE"]

        # Approximate parameter count (FC network assumption)
        current_params = (
            input_size * h1 +
            h1 * h2 +
            h2 * output_size
        )

        stability = (
            "STABLE" if isinstance(final_std, float) and final_std < 5
            else "UNSTABLE"
        )

        prompt = f"""
    You are a FEDERATED LEARNING neural network architect.

    This system uses synchronous FedAvg.
    Any architecture change will be applied SERVER-SIDE and broadcast to ALL clients.
    Architecture changes are EXPENSIVE and MUST be conservative.

    ===== CURRENT ARCHITECTURE =====
    - Input features: {input_size} (FIXED)
    - Hidden Layer 1: {h1}
    - Hidden Layer 2: {h2}
    - Output classes: {output_size} (FIXED)
    - Activation: ReLU (FIXED)
    - Estimated parameters: {current_params}

    ===== FEDERATED PERFORMANCE =====
    - Final global accuracy: {final_global_acc}%
    - Client accuracy std (last round): {final_std}
    - FL stability: {stability}

    ===== CRITICAL CONSTRAINTS (MUST OBEY) =====

    1. ABSOLUTE LIMITS (VIOLATION = INVALID):
    - HIDDEN1_SIZE: MUST be between 32 and 256 (current: {h1})
    - HIDDEN2_SIZE: MUST be between 64 and 512 (current: {h2})
    - Total parameters MUST NOT exceed 150,000
    - INPUT_SIZE and OUTPUT_SIZE are FIXED

    2. FL-SPECIFIC RULES:
    - You may change AT MOST ONE hidden layer
    - If client std ≥ 5:
    - DO NOT increase model size
    - Prefer reducing HIDDEN sizes or KEEP
    - If client std < 5:
    - Model size increase MUST be ≤ 15%
    - Prefer SIMPLER models if accuracy ≥ 97%

    3. FORBIDDEN ACTIONS:
    - Do NOT add layers
    - Do NOT change activation functions
    - Do NOT change optimizer or aggregation
    - Do NOT suggest frequent architecture changes

    ===== ANALYSIS TASK =====

    Determine whether FL performance is limited by:
    (A) under-capacity model
    (B) excessive model complexity for heterogeneous clients
    (C) architecture is already optimal

    Recommend the SAFEST architecture adjustment.

    ===== RESPONSE FORMAT (STRICT) =====

    VERDICT: [KEEP or MODIFY]

    REASONING:
    [2–3 sentences focused on FL stability, client heterogeneity, and capacity]

    RECOMMENDATIONS:
    - HIDDEN1_SIZE: {h1} → [new value or KEEP]
    - HIDDEN2_SIZE: {h2} → [new value or KEEP]

    CONSTRAINT CHECK:
    - HIDDEN1_SIZE in [32,256]: [YES/NO]
    - HIDDEN2_SIZE in [64,512]: [YES/NO]
    - Total params ≤ 150k: [YES/NO]
    - Only one layer changed: [YES/NO]
    - FL-safe update: [YES/NO]

    JSON_OUTPUT:
    {{
    "HIDDEN1_SIZE": [integer between 32–256],
    "HIDDEN2_SIZE": [integer between 64–512]
    }}

    CRITICAL:
    - Output RAW JSON only
    - NO markdown
    - Invalid or aggressive changes invalidate the response
    """

        return prompt

    
    def parse_json_from_response1(self, response):
        """Extract and validate JSON from LLM response."""
        # Remove markdown code fences if present
        response = re.sub(r'```json\\s*', '', response)
        response = re.sub(r'```\\s*', '', response)
        response = response.strip()
        
        # Try to find JSON block
        json_pattern = r'\\{[^}]+\\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            for match in matches:
                try:
                    cleaned = match.replace('\\n', ' ')
                    parsed = json.loads(cleaned)
                    
                    # Validate it has expected keys and reasonable values
                    if parsed and isinstance(parsed, dict) and len(parsed) > 0:
                        return parsed
                except json.JSONDecodeError:
                    continue
        
        return None
    

    def parse_json_from_response(self, response):
        """
        Extract and validate JSON from LLM response.
        Tweak: Uses non-greedy regex and prioritizes the last JSON object found.
        """
        # 1. Remove markdown code fences (```json...``` and ```...```)
        # This non-greedy substitution captures content inside the fences and replaces the 
        # whole block with just the content, handling multi-line JSON.
        response = re.sub(r'```json\s*(.*?)\s*```', r'\1', response, flags=re.DOTALL)
        response = re.sub(r'```\s*', '', response)
        response = response.strip()
        
        # 2. Find all potential JSON blocks using a non-greedy pattern
        # r'\{.*?\}' finds the smallest string that starts with { and ends with }
        # re.DOTALL allows '.' to match newlines
        json_pattern = r'\{.*?\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            # Prioritize the last-occurring JSON object, as this is often the canonical output.
            for match in reversed(matches):
                try:
                    # Replace newlines within the match with spaces for safer parsing
                    cleaned = match.replace('\n', ' ').strip()
                    
                    # Attempt to parse the cleaned string into a JSON object
                    parsed = json.loads(cleaned)
                    
                    # Validate: must be a non-empty dictionary
                    if parsed and isinstance(parsed, dict) and len(parsed) > 0:
                        return parsed # Success: return the valid JSON object
                except json.JSONDecodeError:
                    # If parsing fails, continue to the next match
                    continue
        
        return None
    
    def analyze_architecture(self):
        """Run architecture analysis with LLM."""
        print("\\n" + "="*80)
        print("ARCHITECTURE ANALYSIS (Prompt 1)")
        print("="*80 + "\\n")
        if self.config.get("FL", False):
            prompt = self.create_architecture_fl_prompt()
            print(" Using FEDERATED LEARNING hyperparameter prompt")
        else:
            prompt = self.create_architecture_cent_prompt()
            print(" Using CENTRALIZED hyperparameter prompt")
        
        print(" Generating architecture recommendations...")
        print(" This may take 30-60 seconds...\\n")
        
        try:
            response = self.generate_llm_response(prompt)
            
            print("--- LLM Response ---")
            print(response)
            print("-------------------\\n")
            
            # Parse JSON
            recommendations = self.parse_json_from_response(response)
            
            if recommendations:
                print(" Parsed recommendations:")
                print(json.dumps(recommendations, indent=2))
            else:
                print("  Could not automatically parse JSON, will save full response")
            
            return recommendations, response
            
        except Exception as e:
            print(f" Error during architecture analysis: {e}")
            return None, str(e)
    
    def analyze_hyperparameters(self):
        """Run hyperparameter analysis with LLM (mode-aware)."""
        print("\n" + "="*80)
        print("HYPERPARAMETER ANALYSIS (Prompt 2)")
        print("="*80 + "\n")

        if self.config.get("FL", False):
            prompt = self.create_hyperparameter_fl_prompt()
            print(" Using FEDERATED LEARNING hyperparameter prompt")
        else:
            prompt = self.create_hyperparameter_cent_prompt()
            print(" Using CENTRALIZED hyperparameter prompt")

        print(" Generating hyperparameter recommendations...")
        print(" This may take 30-60 seconds...\n")

        try:
            response = self.generate_llm_response(prompt)

            print("--- LLM Response ---")
            print(response)
            print("-------------------\n")

            recommendations = self.parse_json_from_response(response)

            if recommendations:
                print(" Parsed recommendations:")
                print(json.dumps(recommendations, indent=2))
            else:
                print("  Could not automatically parse JSON, will save full response")

            return recommendations, response

        except Exception as e:
            print(f" Error during hyperparameter analysis: {e}")
            return None, str(e)

            
        
    
    def generate_improved_config(self, arch_recommendations, hyperparam_recommendations):
        new_config = self.config.copy()

        if not self.config.get("FL", False):
            # Centralized
            if arch_recommendations:
                for key in ['HIDDEN1_SIZE', 'HIDDEN2_SIZE', 'DROPOUT_RATE']:
                    if key in arch_recommendations:
                        new_config[key] = arch_recommendations[key]

            if hyperparam_recommendations:
                for key in ['LEARNING_RATE', 'BATCH_SIZE', 'EPOCHS', 'ROUNDS']:
                    if key in hyperparam_recommendations:
                        new_config[key] = hyperparam_recommendations[key]
        else:
            # Federated — ONLY server-side safe params
            if hyperparam_recommendations:
                for key in ['EPOCHS', 'ROUNDS']:
                    if key in hyperparam_recommendations:
                        new_config[key] = hyperparam_recommendations[key]

        return new_config

    
    def save_results(self, arch_response, hyperparam_response, new_config):
        """Save all analysis results."""
        output_dir = Path(self.OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        
        # Save full responses
        with open(output_dir / "llm_architecture_analysis.txt", 'w') as f:
            f.write("ARCHITECTURE ANALYSIS\\n")
            f.write("="*80 + "\\n\\n")
            f.write(arch_response)
        
        with open(output_dir / "llm_hyperparameter_analysis.txt", 'w') as f:
            f.write("HYPERPARAMETER ANALYSIS\\n")
            f.write("="*80 + "\\n\\n")
            f.write(hyperparam_response)
        
        # Save new config
        with open("config_v2_improved.json", 'w') as f:
            json.dump(new_config, indent=2, fp=f)
        
        # Save comparison
        with open(output_dir / "config_comparison.txt", 'w') as f:
            f.write("CONFIGURATION COMPARISON\\n")
            f.write("="*80 + "\\n\\n")
            f.write("ORIGINAL vs IMPROVED\\n\\n")
            
            for key in sorted(new_config.keys()):
                old_val = self.config.get(key, "N/A")
                new_val = new_config.get(key, "N/A")
                
                if old_val != new_val:
                    f.write(f"  {key}:\\n")
                    f.write(f"    Old: {old_val}\\n")
                    f.write(f"    New: {new_val}\\n\\n")
                else:
                    f.write(f"✓  {key}: {old_val} (unchanged)\\n")
        
        print(f"\\n💾 Results saved:")
        print(f"   - {output_dir}/llm_architecture_analysis.txt")
        print(f"   - {output_dir}/llm_hyperparameter_analysis.txt")
        print(f"   - config_v2_improved.json")
        print(f"   - {output_dir}/config_comparison.txt")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\\n" + "="*80)
        print(" LLM-BASED MODEL IMPROVEMENT ANALYZER")
        print("="*80)
        print(f"\\nAnalyzing model: {self.model_path}")
        print(f"Using config: {self.config_path}")
        print(f"Results from: {self.results_path}\\n")
        
        # Run both analyses
        arch_recommendations, arch_response = self.analyze_architecture()
        hyperparam_recommendations, hyperparam_response = self.analyze_hyperparameters()
        
        # Generate improved config
        print("\\n" + "="*80)
        print("GENERATING IMPROVED CONFIGURATION")
        print("="*80 + "\\n")
        
        if arch_recommendations or hyperparam_recommendations:
            new_config = self.generate_improved_config(
                arch_recommendations, 
                hyperparam_recommendations
            )
            
            print(" Successfully generated improved configuration\\n")
            print("Changes made:")
            
            changes_made = False
            for key in sorted(new_config.keys()):
                old_val = self.config.get(key)
                new_val = new_config.get(key)
                
                if old_val != new_val:
                    print(f"  • {key}: {old_val} → {new_val}")
                    changes_made = True
            
            if not changes_made:
                print("  • No changes recommended (model already optimal)")
            
            # Save everything
            self.save_results(arch_response, hyperparam_response, new_config)
            
            print("\\n" + "="*80)
            print(" ANALYSIS COMPLETE")
            print("="*80)
            print("\\nNext steps:")
            print("1. Review the recommendations in centralized_output/")
            print("2. Compare config.json vs config_v2_improved.json")
            print("3. automated_optimization.py will validate and use this config")
            print("="*80 + "\\n")
            
            return new_config
        else:
            print("  Could not parse recommendations from LLM")
            print("Saving full responses for manual review...")
            
            # Still save the responses
            output_dir = Path(self.OUTPUT_DIR)
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / "llm_architecture_analysis.txt", 'w') as f:
                f.write(arch_response)
            
            with open(output_dir / "llm_hyperparameter_analysis.txt", 'w') as f:
                f.write(hyperparam_response)
            
            print(f" Responses saved to {output_dir}/")
            print("   Please review manually and update config.json")
            
            return None
def llm_mid_training_update(*,model_path: str,config_path: str,results_snapshot: dict,round_idx: int,trigger_round: int,allow_architecture_change: bool = True,):
    if round_idx != trigger_round:
        return None  # continue training normally
    print("\n LLM mid-training checkpoint triggered")
    if not is_ollama_installed():
        print(" Ollama not installed, installing...")
        install_ollama()
    if not ensure_ollama_running():
        raise RuntimeError("Ollama server not running")
    if not check_model_exists(MODEL_NAME):
        pull_model(MODEL_NAME)
    improver = ModelImprover(
        model_path=model_path,
        config_path=config_path,
        results_path=None,
    )
    if improver.config.get("FL", False):
        improver.results = {
            "Federated": results_snapshot
        }
    else:
        improver.results = {
            "Centralized": results_snapshot
        }
    
    arch_cfg = None
    arch_response = ""
    if allow_architecture_change:
        arch_cfg, arch_response = improver.analyze_architecture()
        if not arch_cfg:
            print("Architecture analysis failed or unsafe — ignoring architecture changes")
            arch_cfg = None
    else:
        print("Architecture analysis DISABLED (safe default)")
    hyper_cfg, hyper_response = improver.analyze_hyperparameters()
    if not hyper_cfg:
        return None
    new_config = improver.generate_improved_config(
        arch_recommendations=arch_cfg,
        hyperparam_recommendations=hyper_cfg,
    )
    print("\n LLM Suggested Updates:")
    for k in sorted(new_config.keys()):
        old = improver.config.get(k)
        new = new_config.get(k)
        if old != new:
            print(f"  • {k}: {old} → {new}")
    return new_config
