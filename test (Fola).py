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

# --- Configuration (Simulated from config.json) ---
# NOTE: Replace 'config.json' reading with direct values or ensure the file exists.
# For demonstration, I will simulate the constants here based on typical values
# for a network training on your traffic data.
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("WARNING: 'config.json' not found. Using default simulated constants.")
    config = {
        "NUM_CLIENTS": 10,
        "ROUNDS": 5,
        "BATCH_SIZE": 64,
        "LEARNING_RATE": 0.001,
        "EPOCHS": 3,
        "DATA_GROUPS": 4,
        "BATCH_ROUND": 40, # Number of batches per client per round
        "PATH": ".",
        "INPUT_SIZE": 77, # Assuming 77 features from traffic data
        "HIDDEN1_SIZE": 128,
        "HIDDEN2_SIZE": 64,
        "OUTPUT_SIZE": 2, # Assuming binary classification (e.g., normal/attack)
        "DROPOUT_RATE": 0.2
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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Config: ROUNDS={ROUNDS}, EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, SIZE_ROUND={SIZE_ROUND}")


# --- Dataset Preparations (Assuming data files exist) ---

# %%
TrafficData = {}
TrafficData['Dataset']={}
# NOTE: Ensure these files exist in the execution directory
sets_names = ['30','100','70','50','testing']
for DATA_NUM in sets_names:
    try:
        TrafficData['Dataset'][DATA_NUM]=pd.read_csv(f'2_Dataset_4_Attack_{DATA_NUM}_normal.csv', low_memory=False, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
        print(f"Loaded {DATA_NUM}: {TrafficData['Dataset'][DATA_NUM].shape}")
    except FileNotFoundError:
        print(f"ERROR: Data file 2_Dataset_5_Attack_{DATA_NUM}_normal.csv not found.")
        # Exit or provide placeholder data if necessary for a complete run
        import sys
        sys.exit(1) 

for DATA_NUM in TrafficData['Dataset']:
    TrafficData['Dataset'][DATA_NUM]=TrafficData['Dataset'][DATA_NUM].sample(frac=1, random_state=42).reset_index(drop=True)

# %%
TrafficData['Split'] = {}
sets_training = ['30','100','70','50']
for DATA_NUM in sets_training:
    TrafficData['Split'][DATA_NUM] = np.array_split(TrafficData['Dataset'][DATA_NUM], DATA_GROUPS)

# Combine splits sequentially (as done in the original notebook)
TrafficData['Combined'] = pd.concat([TrafficData['Split']['30'][0], TrafficData['Split']['100'][0], TrafficData['Split']['70'][0], TrafficData['Split']['50'][0]]).reset_index(drop=True)
for GROUP in range(1, DATA_GROUPS):
    TrafficData['Combined'] = pd.concat([TrafficData['Combined'], TrafficData['Split']['30'][GROUP], TrafficData['Split']['100'][GROUP], TrafficData['Split']['70'][GROUP], TrafficData['Split']['50'][GROUP]]).reset_index(drop=True)
print(f"Combined Training Data Shape: {TrafficData['Combined'].shape}")

# %%
TrafficData['Train'] = {}
TrafficData['Train']['X'] = TrafficData['Combined'].iloc[:, 0:-1]
TrafficData['Train']['y'] = TrafficData['Combined'].iloc[:, -1]

TrafficData['Test'] = {}
TrafficData['Test']['X']=TrafficData['Dataset']['testing'].iloc[:, 0:-1]
TrafficData['Test']['y']=TrafficData['Dataset']['testing'].iloc[:, -1]

# Feature Scaling
scaler = MinMaxScaler()
model_scaler = scaler.fit(TrafficData['Train']['X'])
TrafficData['Train']['X'] = model_scaler.transform(TrafficData['Train']['X'])
TrafficData['Test']['X'] = model_scaler.transform(TrafficData['Test']['X'])

# Convert to NumPy arrays
TrafficData['Train']['X'], TrafficData['Train']['y']= np.array(TrafficData['Train']['X']), np.array(TrafficData['Train']['y'])
TrafficData['Test']['X'], TrafficData['Test']['y']= np.array(TrafficData['Test']['X']), np.array(TrafficData['Test']['y'])

# %%
# Split the combined training data into sequential rounds of size SIZE_ROUND
TrafficData['ROUNDS']={}
SIZE_Demo = SIZE_ROUND
total_train_samples = len(TrafficData['Train']['X'])

for ROUND in range(1, ROUNDS+1):
    start_idx = (ROUND - 1) * SIZE_ROUND
    end_idx = ROUND * SIZE_ROUND
    
    if start_idx >= total_train_samples:
        print(f"WARNING: Not enough data for Round {ROUND}. Breaking.")
        ROUNDS = ROUND - 1 # Adjust the total ROUNDS
        break

    TrafficData['ROUNDS'][ROUND]={}
    TrafficData['ROUNDS'][ROUND]['X']= TrafficData['Train']['X'][start_idx:end_idx]
    TrafficData['ROUNDS'][ROUND]['y']= TrafficData['Train']['y'][start_idx:end_idx]
    print(f"Round {ROUND} data shape: {TrafficData['ROUNDS'][ROUND]['X'].shape}")

del TrafficData['Train'] # Clean up

# %%
# Custom Dataset Class
class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.from_numpy(X_data).float()
        self.y_data = torch.from_numpy(y_data).long()
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

# Create Datasets
TrafficData['trainsets']={}
for ROUND in range(1, ROUNDS+1):
    TrafficData['trainsets'][ROUND]= ClassifierDataset(TrafficData['ROUNDS'][ROUND]['X'], TrafficData['ROUNDS'][ROUND]['y'])
TrafficData['testset'] = ClassifierDataset(TrafficData['Test']['X'], TrafficData['Test']['y'])

del TrafficData['ROUNDS'] # Clean up

# %%
# Centralized DataLoaders
Centralized_Dataloaders = {}
for ROUND in range(1, ROUNDS + 1):
    # Use the ClassifierDataset created earlier which holds the full round's data
    round_dataset = TrafficData['trainsets'][ROUND]
    
    # Create a single centralized DataLoader for the round
    Centralized_Dataloaders[ROUND] = DataLoader(
        round_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,  # Shuffle the combined data for better training
    )

# The test DataLoader is already centralized
Centralized_Dataloaders['Test'] = DataLoader(TrafficData['testset'], batch_size=BATCH_SIZE, shuffle=False)
print(f"Created {ROUNDS} Centralized Training DataLoaders and 1 Test DataLoader.")

del TrafficData # Clean up

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
# --- Training and Testing Functions ---

def test(net, testloader):
    """Evaluate the model on the test dataset."""
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    
    # Track predictions and actuals for detailed metrics if needed
    prediction_matrix = []
    actual_matrix= []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            prediction_matrix.append(predicted.tolist())
            actual_matrix.append(labels.tolist())
            
    loss /= total
    accuracy = correct / total
    
    return loss, accuracy, prediction_matrix, actual_matrix

def centralized_training_loop(
    net, 
    train_dataloaders, 
    test_dataloader, 
    rounds: int, 
    epochs_per_round: int
):
    """
    Performs centralized training across sequential data rounds and evaluates 
    the model after each round.
    """
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    results = {
        "round_train_loss": [],
        "round_train_acc": [],
        "round_test_loss": [],
        "round_test_acc": [],
    }

    net.to(DEVICE)
    print(f"\n--- Starting Centralized Training on {DEVICE} ---")
    
    # Loop through each sequential data round
    for current_round in range(1, rounds + 1):
        print(f"\n| Starting Training Round {current_round}/{rounds}...")
        
        # Get the centralized DataLoader for the current round
        trainloader = train_dataloaders[current_round]
        
        # --- Training Phase ---
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
                
                # Accumulate metrics
                round_loss_sum += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_examples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        # Calculate average metrics for the round's training data
        round_train_loss = round_loss_sum / total_examples
        round_train_acc = total_correct / total_examples
        
        results["round_train_loss"].append(round_train_loss)
        results["round_train_acc"].append(round_train_acc)
        
        print(f"|   -> Round {current_round} Train Loss: {round_train_loss:.4f} | Train Accuracy: {round_train_acc:.4f}")

        # --- Evaluation Phase on the Test Set ---
        round_test_loss, round_test_acc, _, _ = test(net, test_dataloader)
        
        results["round_test_loss"].append(round_test_loss)
        results["round_test_acc"].append(round_test_acc)
        
        print(f"|   -> Round {current_round} Test Loss: {round_test_loss:.4f} | Test Accuracy: {round_test_acc:.4f}")
        print("-" * 50)

    return results

# %%
# --- Execution ---

# Instantiate the model
central_net = Net().to(DEVICE)
print("Initialized Centralized Neural Network.")

# Run the centralized training loop
history = centralized_training_loop(
    net=central_net,
    train_dataloaders=Centralized_Dataloaders,
    test_dataloader=Centralized_Dataloaders['Test'],
    rounds=ROUNDS,
    epochs_per_round=EPOCHS
)

# Final Summary
print("\n🎉 Final Training History:")
df_history = pd.DataFrame(history)
df_history.index = df_history.index + 1
df_history.index.name = "Round"
print(df_history)

# Save the final centralized model state
# NOTE: Ensure the directory specified by PATH exists
os.makedirs(PATH, exist_ok=True) 
torch.save(central_net.state_dict(), f"{PATH}/Centralized_4_Final_model_Net.pth")
rounds = range(1, len(history["round_train_loss"]) + 1)

## ---- Plot 1: Loss Curves ----
plt.figure(figsize=(8,5))
plt.plot(rounds, history["round_train_loss"], label="Train Loss")
plt.plot(rounds, history["round_test_loss"], label="Test Loss")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("Training & Testing   4 devices Loss Per Round")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_4_curve.png", dpi=300)
plt.show()

# ---- Plot 2: Accuracy Curves ----
plt.figure(figsize=(8,5))
plt.plot(rounds, history["round_train_acc"], label="Train Accuracy")
plt.plot(rounds, history["round_test_acc"], label="Test Accuracy")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Training & Testing  4 devices Accuracy Per Round")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_4_curve.png", dpi=300)
plt.show()


print(f"\nSaved final centralized model to {PATH}/Centralized_Final_model_Net.pth")


# Extract history
