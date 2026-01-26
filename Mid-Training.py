#!/usr/bin/env python3
"""
Mid-Training LLM Optimization
Trains rounds 1-10, gets LLM suggestions, continues rounds 11-40 with improved config
FIXED: Correct paths and metrics for LLM analysis
"""

# %%
# Standard Library and Third-Party Imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict, Counter
import json
import csv
import math
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import subprocess
import sys

# --- Configuration ---
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("WARNING: 'config.json' not found. Using default simulated constants.")
    config = {
        "NUM_CLIENTS": 5,
        "ROUNDS": 40,
        "BATCH_SIZE": 256,
        "LEARNING_RATE": 0.00005,
        "EPOCHS": 2,
        "DATA_GROUPS": 160,
        "BATCH_ROUND": 14,
        "PATH": "midtraining_output",
        "INPUT_SIZE": 98,
        "HIDDEN1_SIZE": 64,
        "HIDDEN2_SIZE": 128,
        "OUTPUT_SIZE": 10,
        "DROPOUT_RATE": 0.2,
        "G": 0
    }

# Access constants
NUM_CLIENTS = config["NUM_CLIENTS"]
ROUNDS = config["ROUNDS"]
BATCH_SIZE = config["BATCH_SIZE"]
LEARNING_RATE = config["LEARNING_RATE"]
EPOCHS = config["EPOCHS"]
DATA_GROUPS = config["DATA_GROUPS"]
BATCH_ROUND = config["BATCH_ROUND"]
SIZE_ROUND = int(BATCH_ROUND * BATCH_SIZE * NUM_CLIENTS)
PATH = config["PATH"]
INPUT_SIZE = config["INPUT_SIZE"]
HIDDEN1_SIZE = config["HIDDEN1_SIZE"]
HIDDEN2_SIZE = config["HIDDEN2_SIZE"]
OUTPUT_SIZE = config["OUTPUT_SIZE"]
DROPOUT_RATE = config["DROPOUT_RATE"]

# Mid-training settings
STAGE1_ROUNDS = 10  # Train first 10 rounds
STAGE2_START = 11   # Continue from round 11
LLM_SCRIPT = "llm.py"

# For metrics calculation
CLASS_COUNT = OUTPUT_SIZE

# Label names for confusion matrix
LABEL_NAMES_8 = ['Normal', 'DDoS_UDP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_HTTP', 'Password', 'Vulnerability_scanner', 'SQL_injection']
LABEL_NAMES_2 = ['Normal', 'Attack']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"\\n{'='*80}")
print(f" MID-TRAINING LLM OPTIMIZATION")
print(f"{'='*80}")
print(f"Device: {DEVICE}")
print(f"Initial Config: ROUNDS={ROUNDS}, EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}")
print(f"Output Path: {PATH}")
print(f"\\nStrategy:")
print(f"   Stage 1: Rounds 1-{STAGE1_ROUNDS} (Initial config)")
print(f"   LLM Analysis at Round {STAGE1_ROUNDS}")
print(f"   Stage 2: Rounds {STAGE2_START}-{ROUNDS} (Improved config)")
print(f"{'='*80}\\n")

# --- Output directory for predictions ---
OUTPUT_DIR = os.path.join(PATH, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PATH, exist_ok=True)

# --- Dataset Preparations ---

# %%
TrafficData = {}
TrafficData['Dataset'] = {}

sets_names = ['30', '100', '70', '50', 'testing']
for DATA_NUM in sets_names:
    try:
        TrafficData['Dataset'][DATA_NUM] = pd.read_csv(
            f'2_Dataset_4_Attack_{DATA_NUM}_normal.csv', 
            low_memory=False, 
            quoting=csv.QUOTE_NONE, 
            on_bad_lines='skip'
        )
        print(f"Loaded {DATA_NUM}: {TrafficData['Dataset'][DATA_NUM].shape}")
    except FileNotFoundError:
        print(f"ERROR: Data file 2_Dataset_4_Attack_{DATA_NUM}_normal.csv not found.")
        sys.exit(1)

for DATA_NUM in TrafficData['Dataset']:
    TrafficData['Dataset'][DATA_NUM] = TrafficData['Dataset'][DATA_NUM].sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

# %%
TrafficData['Split'] = {}
sets_training = ['30', '100', '70', '50']
for DATA_NUM in sets_training:
    TrafficData['Split'][DATA_NUM] = np.array_split(TrafficData['Dataset'][DATA_NUM], DATA_GROUPS)

# Combine splits sequentially
TrafficData['Combined'] = pd.concat([
    TrafficData['Split']['30'][0], 
    TrafficData['Split']['100'][0], 
    TrafficData['Split']['70'][0], 
    TrafficData['Split']['50'][0]
]).reset_index(drop=True)

for GROUP in range(1, DATA_GROUPS):
    TrafficData['Combined'] = pd.concat([
        TrafficData['Combined'], 
        TrafficData['Split']['30'][GROUP], 
        TrafficData['Split']['100'][GROUP], 
        TrafficData['Split']['70'][GROUP], 
        TrafficData['Split']['50'][GROUP]
    ]).reset_index(drop=True)
print(f"Combined Training Data Shape: {TrafficData['Combined'].shape}")

# %%
TrafficData['Train'] = {}
TrafficData['Train']['X'] = TrafficData['Combined'].iloc[:, 0:-1]
TrafficData['Train']['y'] = TrafficData['Combined'].iloc[:, -1]

TrafficData['Test'] = {}
TrafficData['Test']['X'] = TrafficData['Dataset']['testing'].iloc[:, 0:-1]
TrafficData['Test']['y'] = TrafficData['Dataset']['testing'].iloc[:, -1]

# Feature Scaling
scaler = MinMaxScaler()
model_scaler = scaler.fit(TrafficData['Train']['X'])
TrafficData['Train']['X'] = model_scaler.transform(TrafficData['Train']['X'])
TrafficData['Test']['X'] = model_scaler.transform(TrafficData['Test']['X'])

# Convert to NumPy arrays
TrafficData['Train']['X'] = np.array(TrafficData['Train']['X'])
TrafficData['Train']['y'] = np.array(TrafficData['Train']['y'])
TrafficData['Test']['X'] = np.array(TrafficData['Test']['X'])
TrafficData['Test']['y'] = np.array(TrafficData['Test']['y'])

# %%
# Split the combined training data into sequential rounds
TrafficData['ROUNDS'] = {}
total_train_samples = len(TrafficData['Train']['X'])

for ROUND in range(1, ROUNDS + 1):
    start_idx = (ROUND - 1) * SIZE_ROUND
    end_idx = ROUND * SIZE_ROUND
    
    if start_idx >= total_train_samples:
        print(f"WARNING: Not enough data for Round {ROUND}. Breaking.")
        ROUNDS = ROUND - 1
        break

    TrafficData['ROUNDS'][ROUND] = {}
    TrafficData['ROUNDS'][ROUND]['X'] = TrafficData['Train']['X'][start_idx:end_idx]
    TrafficData['ROUNDS'][ROUND]['y'] = TrafficData['Train']['y'][start_idx:end_idx]

del TrafficData['Train']

# %%
# Custom Dataset Class
class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.from_numpy(X_data).float()
        self.y_data = torch.from_numpy(y_data).long()
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__(self):
        return len(self.X_data)

# Create Datasets
TrafficData['trainsets'] = {}
for ROUND in range(1, ROUNDS + 1):
    TrafficData['trainsets'][ROUND] = ClassifierDataset(
        TrafficData['ROUNDS'][ROUND]['X'], 
        TrafficData['ROUNDS'][ROUND]['y']
    )
TrafficData['testset'] = ClassifierDataset(TrafficData['Test']['X'], TrafficData['Test']['y'])

del TrafficData['ROUNDS']

# %%
# Centralized DataLoaders
Centralized_Dataloaders = {}
for ROUND in range(1, ROUNDS + 1):
    round_dataset = TrafficData['trainsets'][ROUND]
    Centralized_Dataloaders[ROUND] = DataLoader(
        round_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
    )

Centralized_Dataloaders['Test'] = DataLoader(
    TrafficData['testset'], 
    batch_size=BATCH_SIZE, 
    shuffle=False
)
print(f"Created {ROUNDS} Centralized Training DataLoaders and 1 Test DataLoader.\\n")

del TrafficData

# %%
# --- Neural Network Definition ---

class Net(nn.Module):
    def __init__(self,
                 input_size=INPUT_SIZE,
                 hidden1_size=HIDDEN1_SIZE,
                 hidden2_size=HIDDEN2_SIZE,
                 output_size=OUTPUT_SIZE,
                 dropout_rate=DROPOUT_RATE):
        super(Net, self).__init__()

        self.layer_1 = nn.Linear(input_size, hidden1_size)
        self.layer_2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer_out = nn.Linear(hidden2_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_out(x)
        return x


# %%
# --- Metrics Functions ---

def calculate_accuracy(actual_values, predicted_values):
    """Fraction of correct predictions."""
    correct_predictions = sum(a == p for a, p in zip(actual_values, predicted_values))
    total_samples = len(actual_values)
    return correct_predictions / total_samples


def calculate_weighted_precision(actual_values, predicted_values, n_classes):
    """Weighted Precision by actual distribution."""
    total_samples = len(actual_values)
    
    tp = [0] * n_classes
    fp = [0] * n_classes
    actual_count = [0] * n_classes

    for a, p in zip(actual_values, predicted_values):
        actual_count[a] += 1
        if a == p:
            tp[a] += 1
        else:
            fp[p] += 1

    precision_sum = 0.0
    for i in range(n_classes):
        denominator = tp[i] + fp[i]
        precision_i = tp[i] / denominator if denominator > 0 else 0.0
        weight_i = actual_count[i] / total_samples
        precision_sum += precision_i * weight_i
    
    return precision_sum


def calculate_weighted_recall(actual_values, predicted_values, n_classes):
    """Weighted Recall by actual distribution."""
    total_samples = len(actual_values)
    
    tp = [0] * n_classes
    fn = [0] * n_classes
    actual_count = [0] * n_classes

    for a, p in zip(actual_values, predicted_values):
        actual_count[a] += 1
        if a == p:
            tp[a] += 1
        else:
            fn[a] += 1

    recall_sum = 0.0
    for i in range(n_classes):
        denominator = tp[i] + fn[i]
        recall_i = tp[i] / denominator if denominator > 0 else 0.0
        weight_i = actual_count[i] / total_samples
        recall_sum += recall_i * weight_i

    return recall_sum


def calculate_weighted_f1(actual_values, predicted_values, n_classes):
    """Computes the weighted F1 score."""
    total_samples = len(actual_values)
    
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
    
    for a, p in zip(actual_values, predicted_values):
        if a == p:
            tp[a] += 1
        else:
            fp[p] += 1
            fn[a] += 1

    f1_sum = 0.0
    
    for i in range(n_classes):
        precision_i = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        recall_i = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        
        if precision_i + recall_i > 0:
            f1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i)
        else:
            f1_i = 0.0
        
        weight_i = (tp[i] + fn[i]) / total_samples if total_samples > 0 else 0.0
        f1_sum += f1_i * weight_i
    
    return f1_sum


# %%
# --- Training and Testing Functions ---

def test(net, testloader):
    """Evaluate the model on the test dataset and return predictions."""
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    
    prediction_matrix = []
    actual_matrix = []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            prediction_matrix.append(predicted.cpu().tolist())
            actual_matrix.append(labels.cpu().tolist())
            
    loss /= total
    accuracy = correct / total
    
    return loss, accuracy, prediction_matrix, actual_matrix


def centralized_training_midtraining(
    net, 
    train_dataloaders, 
    test_dataloader, 
    total_rounds: int,
    stage1_rounds: int,
    epochs_per_round: int,
    learning_rate: float,
    output_dir: str,
    path: str
):
    """
    Mid-training optimization:
    - Stage 1: Train rounds 1-10
    - LLM Analysis
    - Stage 2: Continue rounds 11-40 with improved config
    """
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    results = {
        "round_train_loss": [],
        "round_train_acc": [],
        "round_test_loss": [],
        "round_test_acc": [],
    }

    net.to(DEVICE)
    print(f"{'='*80}")
    print(f"🚀 STAGE 1: Training Rounds 1-{stage1_rounds}")
    print(f"{'='*80}\\n")
    
    # ============================================
    # STAGE 1: Initial Training (Rounds 1-10)
    # ============================================
    for current_round in range(1, stage1_rounds + 1):
        print(f"Round {current_round}/{stage1_rounds} (Stage 1)...", end=" ")
        
        trainloader = train_dataloaders[current_round]
        
        # Training Phase
        net.train()
        total_correct, total_examples, round_loss_sum = 0, 0, 0.0
        
        for epoch in range(epochs_per_round):
            for images, labels in trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                round_loss_sum += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_examples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        round_train_loss = round_loss_sum / total_examples
        round_train_acc = total_correct / total_examples
        
        results["round_train_loss"].append(round_train_loss)
        results["round_train_acc"].append(round_train_acc)

        # Evaluation Phase
        round_test_loss, round_test_acc, predictions, actuals = test(net, test_dataloader)
        
        results["round_test_loss"].append(round_test_loss)
        results["round_test_acc"].append(round_test_acc)
        
        # Save predictions
        with open(f"{output_dir}/Global_{current_round}_pred", 'wb') as f:
            pickle.dump(predictions, f)
        with open(f"{output_dir}/Global_{current_round}_actual", 'wb') as f:
            pickle.dump(actuals, f)
        
        print(f"Train: {round_train_acc:.4f} | Test: {round_test_acc:.4f}")
    
    # Save Stage 1 checkpoint
    checkpoint_path = f"{path}/checkpoint_stage1_round{stage1_rounds}.pth"
    torch.save(net.state_dict(), checkpoint_path)
    print(f"\\n💾 Checkpoint saved: {checkpoint_path}")
    
    # Calculate full metrics for Stage 1
    print("Calculating Stage 1 metrics for LLM...")
    stage1_results = {
        "Centralized": {
            "Accuracy": results["round_test_acc"].copy(),
            "Precision": [],
            "Recall": [],
            "F1_Score": [],
        }
    }

    for round_num in range(1, stage1_rounds + 1):
        try:
            with open(f"{output_dir}/Global_{round_num}_actual", 'rb') as f:
                actual = [item for sublist in pickle.load(f) for item in sublist]
            
            with open(f"{output_dir}/Global_{round_num}_pred", 'rb') as f:
                predicted = [item for sublist in pickle.load(f) for item in sublist]
            
            stage1_results["Centralized"]["Precision"].append(
                calculate_weighted_precision(actual, predicted, CLASS_COUNT)
            )
            stage1_results["Centralized"]["Recall"].append(
                calculate_weighted_recall(actual, predicted, CLASS_COUNT)
            )
            stage1_results["Centralized"]["F1_Score"].append(
                calculate_weighted_f1(actual, predicted, CLASS_COUNT)
            )
        except Exception as e:
            print(f"   Warning: Could not calculate metrics for round {round_num}")
            stage1_results["Centralized"]["Precision"].append(0.0)
            stage1_results["Centralized"]["Recall"].append(0.0)
            stage1_results["Centralized"]["F1_Score"].append(0.0)

    stage1_results_path = f"{output_dir}/results.pkl"
    with open(stage1_results_path, "wb") as f:
        pickle.dump(stage1_results, f)
    
    print(f"Results saved: {stage1_results_path}")
    print(f"\\nStage 1 Performance:")
    print(f"   Final Test Accuracy:  {stage1_results['Centralized']['Accuracy'][-1]:.4f}")
    print(f"   Final Test Precision: {stage1_results['Centralized']['Precision'][-1]:.4f}")
    print(f"   Final Test Recall:    {stage1_results['Centralized']['Recall'][-1]:.4f}")
    print(f"   Final Test F1 Score:  {stage1_results['Centralized']['F1_Score'][-1]:.4f}")
    
    # ============================================
    # LLM ANALYSIS
    # ============================================
    print(f"\\n{'='*80}")
    print(f" LLM ANALYSIS")
    print(f"{'='*80}\\n")
    
    print("Calling LLM for optimization suggestions...")
    print(f"   Model checkpoint: {checkpoint_path}")
    print(f"   Results file: {stage1_results_path}")
    print(f"   Config file: config.json\\n")
    
    try:
        result = subprocess.run(
            ["python", LLM_SCRIPT,
             "--model", checkpoint_path,
             "--results", stage1_results_path],
            timeout=600
        )
        
        if result.returncode == 0:
            print("\\n LLM analysis completed")
        else:
            print(f"\\n LLM returned code {result.returncode}")
    except subprocess.TimeoutExpired:
        print("\\n LLM analysis timed out")
    except Exception as e:
        print(f"\\n LLM error: {e}")
    
    # Load improved config
    improved_config = None
    if os.path.exists("config_v2_improved.json"):
        with open("config_v2_improved.json", 'r') as f:
            improved_config = json.load(f)
        
        print(f"\\n Loaded improved config from LLM\\n")
        print(f" Config Changes:")
        
        # Show changes
        changes = []
        if improved_config['LEARNING_RATE'] != learning_rate:
            changes.append(f"   Learning Rate: {learning_rate:.6f} → {improved_config['LEARNING_RATE']:.6f}")
        if improved_config['BATCH_SIZE'] != BATCH_SIZE:
            changes.append(f"   Batch Size: {BATCH_SIZE} → {improved_config['BATCH_SIZE']}")
        if improved_config['EPOCHS'] != epochs_per_round:
            changes.append(f"   Epochs per Round: {epochs_per_round} → {improved_config['EPOCHS']}")
        if improved_config['DROPOUT_RATE'] != DROPOUT_RATE:
            changes.append(f"   Dropout Rate: {DROPOUT_RATE} → {improved_config['DROPOUT_RATE']}")
        
        if changes:
            for change in changes:
                print(change)
        else:
            print("   No changes recommended (config already optimal)")
        
        # Update training parameters
        new_lr = improved_config['LEARNING_RATE']
        new_epochs = improved_config['EPOCHS']
        new_dropout = improved_config.get('DROPOUT_RATE', DROPOUT_RATE)
        
        # Create new optimizer with updated LR
        optimizer = optim.Adam(net.parameters(), lr=new_lr)
        
        # Note: We can't change dropout mid-training without recreating the model
        # So we'll note it but only apply LR and Epochs changes
        if new_dropout != DROPOUT_RATE:
            print(f"\\n     Note: Dropout change ({DROPOUT_RATE} → {new_dropout}) cannot be applied mid-training")
            print(f"          Architecture changes require model recreation")
        
        print(f"\\n Applied new learning rate and epochs for Stage 2")
    else:
        print(f"\\n No improved config found (config_v2_improved.json missing)")
        print(f"   Continuing with original config")
        new_lr = learning_rate
        new_epochs = epochs_per_round
    
    # ============================================
    # STAGE 2: Continue Training (Rounds 11-40)
    # ============================================
    print(f"\\n{'='*80}")
    print(f" STAGE 2: Training Rounds {stage1_rounds+1}-{total_rounds}")
    print(f"{'='*80}\\n")
    
    for current_round in range(stage1_rounds + 1, total_rounds + 1):
        print(f"Round {current_round}/{total_rounds} (Stage 2)...", end=" ")
        
        trainloader = train_dataloaders[current_round]
        
        # Training Phase with NEW config
        net.train()
        total_correct, total_examples, round_loss_sum = 0, 0, 0.0
        
        for epoch in range(new_epochs):  # Use new epochs
            for images, labels in trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()  # Uses new optimizer with new LR
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                round_loss_sum += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_examples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        round_train_loss = round_loss_sum / total_examples
        round_train_acc = total_correct / total_examples
        
        results["round_train_loss"].append(round_train_loss)
        results["round_train_acc"].append(round_train_acc)

        # Evaluation Phase
        round_test_loss, round_test_acc, predictions, actuals = test(net, test_dataloader)
        
        results["round_test_loss"].append(round_test_loss)
        results["round_test_acc"].append(round_test_acc)
        
        # Save predictions
        with open(f"{output_dir}/Global_{current_round}_pred", 'wb') as f:
            pickle.dump(predictions, f)
        with open(f"{output_dir}/Global_{current_round}_actual", 'wb') as f:
            pickle.dump(actuals, f)
        
        print(f"Train: {round_train_acc:.4f} | Test: {round_test_acc:.4f}")

    return results


# %%
# --- Plotting Functions ---

def plot_training_curves(history, output_dir, stage1_rounds):
    """Plot training curves with stage division marker."""
    rounds = range(1, len(history["round_train_loss"]) + 1)

    # Plot Loss Curves
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, history["round_train_loss"], 'b-o', label="Train Loss", linewidth=2, markersize=4)
    plt.plot(rounds, history["round_test_loss"], 'r-s', label="Test Loss", linewidth=2, markersize=4)
    plt.axvline(x=stage1_rounds, color='gold', linestyle='--', linewidth=2.5, 
                label=f'🤖 LLM Optimization (Round {stage1_rounds})', alpha=0.8)
    plt.xlabel("Round", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training & Testing Loss Per Round (Mid-Training Optimization)", fontsize=16)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_curve.png", dpi=300)
    plt.close()

    # Plot Accuracy Curves
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, history["round_train_acc"], 'b-o', label="Train Accuracy", linewidth=2, markersize=4)
    plt.plot(rounds, history["round_test_acc"], 'r-s', label="Test Accuracy", linewidth=2, markersize=4)
    plt.axvline(x=stage1_rounds, color='gold', linestyle='--', linewidth=2.5, 
                label=f' LLM Optimization (Round {stage1_rounds})', alpha=0.8)
    plt.xlabel("Round", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Training & Testing Accuracy Per Round (Mid-Training Optimization)", fontsize=16)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_curve.png", dpi=300)
    plt.close()
    
    print(f" Saved training curves to {output_dir}/")


def plot_metrics_over_rounds(results, output_dir, class_count, stage1_rounds):
    """Plot metrics with stage division marker."""
    rounds_range = range(1, len(results['Accuracy']) + 1)
    
    # All metrics in one plot
    plt.figure(figsize=(12, 7))
    plt.plot(rounds_range, results['Accuracy'], 'ro-', label='Accuracy', linewidth=2, markersize=5)
    plt.plot(rounds_range, results['Precision'], 'b*-', label='Precision', linewidth=2, markersize=5)
    plt.plot(rounds_range, results['Recall'], 'gs-', label='Recall', linewidth=2, markersize=5)
    plt.plot(rounds_range, results['F1_Score'], 'mh-', label='F1 Score', linewidth=2, markersize=5)
    plt.axvline(x=stage1_rounds, color='gold', linestyle='--', linewidth=2.5, 
                label=f' LLM Optimization', alpha=0.8)
    plt.xlabel('Training Round', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('All Metrics Per Round (Mid-Training Optimization)', fontsize=16)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_all.png", dpi=300)
    plt.close()
    
    print(f" Saved metric plots to {output_dir}/")


def plot_confusion_matrix(actual, predicted, class_count, output_dir, title="Confusion Matrix"):
    """Plot confusion matrix."""
    if class_count == 2:
        label_names = ['Normal', 'Attack']
    elif class_count == 8:
        label_names = ['Normal', 'DDoS_UDP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_HTTP', 'Password', 'Vulnerability_scanner', 'SQL_injection']
    else:
        label_names = [f'Class_{i}' for i in range(class_count)]
    
    label_numbers = list(range(class_count))
    
    cm = confusion_matrix(actual, predicted, labels=label_numbers)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=label_names, 
        yticklabels=label_names
    )
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("Actual Labels", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    plt.close()
    
    print(f" Saved confusion matrix to {output_dir}/")


# %%
# --- Main Execution ---

if __name__ == "__main__":
    # Instantiate the model
    central_net = Net().to(DEVICE)
    print(" Initialized Neural Network\\n")

    # Run the mid-training optimization
    history = centralized_training_midtraining(
        net=central_net,
        train_dataloaders=Centralized_Dataloaders,
        test_dataloader=Centralized_Dataloaders['Test'],
        total_rounds=ROUNDS,
        stage1_rounds=STAGE1_ROUNDS,
        epochs_per_round=EPOCHS,
        learning_rate=LEARNING_RATE,
        output_dir=OUTPUT_DIR,
        path=PATH
    )

    # Print training history
    print(f"\\n{'='*80}")
    print(" TRAINING HISTORY")
    print(f"{'='*80}\\n")
    df_history = pd.DataFrame(history)
    df_history.index = df_history.index + 1
    df_history.index.name = "Round"
    print(df_history)

    # Save the final model
    final_model_path = f"{PATH}/Final_model_midtraining.pth"
    torch.save(central_net.state_dict(), final_model_path)
    print(f"\\n Final model: {final_model_path}")

    # Plot training curves
    plot_training_curves(history, OUTPUT_DIR, STAGE1_ROUNDS)

    # Load predictions and calculate detailed metrics
    print(f"\\n{'='*80}")
    print(" CALCULATING FINAL METRICS")
    print(f"{'='*80}\\n")
    
    Data = {"Centralized": {"Actual": {}, "Predictions": {}}}
    
    for i in range(1, ROUNDS + 1):
        try:
            with open(f"{OUTPUT_DIR}/Global_{i}_actual", 'rb') as file:
                Actual = pickle.load(file)
            Data["Centralized"]["Actual"][i] = [item for sublist in Actual for item in sublist]
            
            with open(f"{OUTPUT_DIR}/Global_{i}_pred", 'rb') as file:
                Pred = pickle.load(file)
            Data["Centralized"]["Predictions"][i] = [item for sublist in Pred for item in sublist]
        except FileNotFoundError:
            print(f" File not found for round {i}")

    # Calculate metrics
    Results = {"Centralized": {
        "Accuracy": [],
        "Recall": [],
        "Precision": [],
        "F1_Score": []
    }}

    for round_num in range(1, ROUNDS + 1):
        try:
            actual = Data["Centralized"]["Actual"][round_num]
            predicted = Data["Centralized"]["Predictions"][round_num]
            
            Results["Centralized"]["Accuracy"].append(
                calculate_accuracy(actual, predicted)
            )
            Results["Centralized"]["Precision"].append(
                calculate_weighted_precision(actual, predicted, CLASS_COUNT)
            )
            Results["Centralized"]["Recall"].append(
                calculate_weighted_recall(actual, predicted, CLASS_COUNT)
            )
            Results["Centralized"]["F1_Score"].append(
                calculate_weighted_f1(actual, predicted, CLASS_COUNT)
            )
        except KeyError:
            print(f" Missing data for round {round_num}")

    # Print metrics summary
    metrics_df = pd.DataFrame({
        'Round': range(1, ROUNDS + 1),
        'Accuracy': Results["Centralized"]["Accuracy"],
        'Precision': Results["Centralized"]["Precision"],
        'Recall': Results["Centralized"]["Recall"],
        'F1_Score': Results["Centralized"]["F1_Score"]
    })
    print(metrics_df.to_string(index=False))
    
    # Highlight Stage 1 vs Stage 2 performance
    print(f"\\n{'='*80}")
    print(" STAGE COMPARISON")
    print(f"{'='*80}\\n")
    
    stage1_acc = Results["Centralized"]["Accuracy"][:STAGE1_ROUNDS]
    stage2_acc = Results["Centralized"]["Accuracy"][STAGE1_ROUNDS:]
    
    stage1_avg = sum(stage1_acc) / len(stage1_acc)
    stage2_avg = sum(stage2_acc) / len(stage2_acc)
    
    print(f"Stage 1 (Rounds 1-{STAGE1_ROUNDS}):")
    print(f"  Average Accuracy: {stage1_avg:.4f}")
    print(f"  Final Accuracy:   {stage1_acc[-1]:.4f}")
    
    print(f"\\nStage 2 (Rounds {STAGE1_ROUNDS+1}-{ROUNDS}):")
    print(f"  Average Accuracy: {stage2_avg:.4f}")
    print(f"  Final Accuracy:   {stage2_acc[-1]:.4f}")
    
    improvement = stage2_avg - stage1_avg
    improvement_pct = (improvement / stage1_avg) * 100
    
    if improvement > 0:
        print(f"\\n Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    elif improvement < 0:
        print(f"\\n Performance decreased: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    else:
        print(f"\\n  No change in performance")
    
    print(f"{'='*80}\\n")

    # Plot metrics
    plot_metrics_over_rounds(Results["Centralized"], OUTPUT_DIR, CLASS_COUNT, STAGE1_ROUNDS)

    # Plot confusion matrix for final round
    final_round = ROUNDS
    plot_confusion_matrix(
        Data["Centralized"]["Actual"][final_round],
        Data["Centralized"]["Predictions"][final_round],
        CLASS_COUNT,
        OUTPUT_DIR,
        title=f"Confusion Matrix - Round {final_round}"
    )

    # Save final results
    with open(f"{OUTPUT_DIR}/results_final.pkl", "wb") as file:
        pickle.dump(Results, file)

    # Print classification report
    print(f"{'='*80}")
    print(f" CLASSIFICATION REPORT (Final Round {final_round})")
    print(f"{'='*80}\\n")
    print(classification_report(
        Data["Centralized"]["Actual"][final_round],
        Data["Centralized"]["Predictions"][final_round]
    ))

    print(f"{'='*80}")
    print(" COMPLETE")
    print(f"{'='*80}\\n")
    print(f"All outputs saved to:")
    print(f"  📁 {PATH}/")
    print(f"     ├── Final_model_midtraining.pth")
    print(f"     ├── checkpoint_stage1_round{STAGE1_ROUNDS}.pth")
    print(f"     └── results/")
    print(f"         ├── results.pkl (Stage 1)")
    print(f"         ├── results_final.pkl (All rounds)")
    print(f"         ├── Global_*_pred/actual")
    print(f"         ├── accuracy_curve.png")
    print(f"         ├── loss_curve.png")
    print(f"         ├── metrics_all.png")
    print(f"         └── confusion_matrix.png\\n")
