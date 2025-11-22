#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-19T02:28:43.364Z
"""

# <font color='green'>***Installation and Libraries Import***</font>
# ---
# --- 


%pip install flwr==1.5.0
%pip install 'flwr[simulation]==1.5.0'
%pip install --upgrade pip
%pip install torch torchvision matplotlib
%pip install async-timeout

import os
libraries_to_uninstall = [
    "tb-nightly==2.18.0a20240701",
    "tensorboard==2.16.2",
    "tensorboard-data-server==0.7.2",
    "tensorboard-plugin-wit==1.8.1",
    "tensorflow==2.16.2",
    "tensorflow-io-gcs-filesystem==0.37.0",
    "termcolor==2.4.0",
    "terminado==0.18.1",
    "tf-estimator-nightly==2.8.0.dev2021122109",
    "tf_keras-nightly==2.18.0.dev2024070109",
    "tf-nightly==2.18.0.dev20240626"
]
for library in libraries_to_uninstall:
    os.system(f"pip uninstall -y {library}")

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
from torch.utils.data import Dataset, WeightedRandomSampler
import flwr as fl
from flwr.common import Metrics
# from flwr_datasets import FederatedDataset
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
import random
import math
import pickle
import csv
import copy
print(fl.__version__)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# <font color='Brown'>***FL Constants***</font>
# ---
# --- 


# DEVICE
NUM_CLIENTS = 48
ROUNDS = 40
BATCH_SIZE = 64
LEARNING_RATE = 0.0025
EPOCHS = 3
DATA_GROUPS = 120
BATCH_ROUND = 6
SIZE_ROUND = int(BATCH_ROUND * BATCH_SIZE * NUM_CLIENTS)
PATH = '20241215_3Epochs'

# <font color='Light Blue'>***Dataset Preparations***</font>
# ---
# --- 


TrafficData = {}
TrafficData['Dataset']={}
sets_names = ['30','100','70','50','testing']
for  DATA_NUM in sets_names:
    TrafficData['Dataset'][DATA_NUM]=pd.read_csv(f'Datasets/3_Generalization_20241201_{DATA_NUM}.csv', low_memory=False, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    print(DATA_NUM, TrafficData['Dataset'][DATA_NUM].shape)
for DATA_NUM in TrafficData['Dataset']:
    TrafficData['Dataset'][DATA_NUM]=TrafficData['Dataset'][DATA_NUM].sample(frac=1, random_state=42).reset_index(drop=True)

TrafficData['Split'] = {}
sets_training =  ['30','100','70','50']
for DATA_NUM in sets_training:
    TrafficData['Split'][DATA_NUM] = np.array_split(TrafficData['Dataset'][DATA_NUM],DATA_GROUPS)

TrafficData['Combined'] = pd.concat([TrafficData['Split']['30'][0], TrafficData['Split']['100'][0], TrafficData['Split']['70'][0], TrafficData['Split']['50'][0]]).reset_index(drop=True)
for GROUP in range(1, DATA_GROUPS):
    TrafficData['Combined'] = pd.concat([TrafficData['Combined'], TrafficData['Split']['30'][GROUP], TrafficData['Split']['100'][GROUP], TrafficData['Split']['70'][GROUP], TrafficData['Split']['50'][GROUP]]).reset_index(drop=True)
print(TrafficData['Combined'].shape)            

TrafficData['Train'] = {}
TrafficData['Train']['X'] = TrafficData['Combined'].iloc[:, 0:-1]
TrafficData['Train']['y'] = TrafficData['Combined'].iloc[:, -1]
print(TrafficData['Train']['X'].shape)
print(TrafficData['Train']['y'].shape)

TrafficData['Test'] = {}
TrafficData['Test']['X']=TrafficData['Dataset']['testing'].iloc[:, 0:-1]
TrafficData['Test']['y']=TrafficData['Dataset']['testing'].iloc[:, -1]
print(TrafficData['Test']['X'].shape)
print(TrafficData['Test']['y'].shape)

scaler = MinMaxScaler()
model = scaler.fit(TrafficData['Train']['X'])
TrafficData['Train']['X'] = model.transform(TrafficData['Train']['X'])
TrafficData['Test']['X'] = model.transform(TrafficData['Test']['X'])

TrafficData['Train']['X'], TrafficData['Train']['y']= np.array(TrafficData['Train']['X']), np.array(TrafficData['Train']['y'])
print(type(TrafficData['Train']['X']),type(TrafficData['Train']['y']))
print(TrafficData['Train']['X'].shape,TrafficData['Train']['y'].shape)
TrafficData['Test']['X'], TrafficData['Test']['y']= np.array(TrafficData['Test']['X']), np.array(TrafficData['Test']['y'])
print(type(TrafficData['Test']['X']),type(TrafficData['Test']['y']))
print(TrafficData['Test']['X'].shape,TrafficData['Test']['y'].shape)

TrafficData['ROUNDS']={}
for ROUND in range(1, ROUNDS+1):
    TrafficData['ROUNDS'][ROUND]={}

SIZE_Demo = SIZE_ROUND
for ROUND in range(1,ROUNDS+1):
    if ROUND == 1:
        TrafficData['ROUNDS'][ROUND]['X']= TrafficData['Train']['X'][:SIZE_Demo]
        TrafficData['ROUNDS'][ROUND]['y']= TrafficData['Train']['y'][:SIZE_Demo]
    else:
        print((SIZE_Demo - SIZE_ROUND),SIZE_Demo)
        TrafficData['ROUNDS'][ROUND]['X']= TrafficData['Train']['X'][(SIZE_Demo - SIZE_ROUND):SIZE_Demo]
        TrafficData['ROUNDS'][ROUND]['y']= TrafficData['Train']['y'][(SIZE_Demo - SIZE_ROUND):SIZE_Demo]
    SIZE_Demo = SIZE_Demo + SIZE_ROUND
for ROUND in TrafficData['ROUNDS']:
    print("ROUND: ", ROUND, TrafficData['ROUNDS'][ROUND]['X'].shape, TrafficData['ROUNDS'][ROUND]['y'].shape)
print(SIZE_Demo, SIZE_ROUND)

class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__ (self):
        return len(self.X_data)

TrafficData['trainsets']={}
for ROUND in range(1, ROUNDS+1):
    TrafficData['trainsets'][ROUND]= ClassifierDataset(torch.from_numpy(TrafficData['ROUNDS'][ROUND]['X']).float(), torch.from_numpy(TrafficData['ROUNDS'][ROUND]['y']).long())
TrafficData['testset'] = ClassifierDataset(torch.from_numpy(TrafficData['Test']['X']).float(), torch.from_numpy(TrafficData['Test']['y']).long())

def load_train(numberofclients, ROUND):    
    portion_size = int(BATCH_ROUND*BATCH_SIZE)
    num_portions = int(NUM_CLIENTS)
    portion_indices = []
    for i in range(num_portions):
        start_idx = i * portion_size
        end_idx = (i + 1) * portion_size
        portion_indices.append(list(range(start_idx, min(end_idx, SIZE_ROUND))))
    portion_datasets = [Subset(TrafficData['trainsets'][ROUND], indices) for indices in portion_indices]
    portion_loaders = [DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False) for dataset in portion_datasets]             
    return portion_loaders
def load_test(numberofclients):    
    testloader = DataLoader(TrafficData['testset'], batch_size=BATCH_SIZE, shuffle=False)
    return testloader

Dataloaders = {}
for ROUND in range(1, ROUNDS+1):
    Dataloaders[ROUND] = load_train(NUM_CLIENTS, ROUND)
Dataloaders['Test'] = load_test(NUM_CLIENTS)

for i, batch in enumerate(Dataloaders['Test']):
    batch_size = len(batch[0])  # Assuming the first element of the batch is the data
    print(f"Batch {i+1} size: {batch_size}")
    if batch_size != 64:
        print(f"Batch {i+1} does not contain 64 records.")
        break

for i, batch in enumerate(Dataloaders[160][0]):
    batch_size = len(batch[0])  # Assuming the first element of the batch is the data
    print(f"Batch {i+1} size: {batch_size}")
    if batch_size != 64:
        print(f"Batch {i+1} does not contain 64 records.")
        break

from collections import Counter
for CLUSTER in range (1, 9):
    DEVICE_PERCENTAGE = []
    for DEVICE__ in range(0, 12):
        for i, batch in enumerate(Dataloaders[CLUSTER][DEVICE__]):
            _, labels = batch
            class_counts = Counter(labels.numpy())
            total_records = sum(class_counts.values())
            class_0_count = class_counts.get(0, 0)
            percentage_class_0 = (class_0_count / total_records) * 100
            DEVICE_PERCENTAGE.append(percentage_class_0)
            # print(f"Batch {i+1}: {dict(class_counts)}")
            # print(f"Percentage of class 0: {percentage_class_0:.2f}%\n")
    # print(DEVICE_PERCENTAGE)        
    chunk_size = 6
    averages = [sum(DEVICE_PERCENTAGE[i:i + chunk_size]) / chunk_size for i in range(0, len(DEVICE_PERCENTAGE), chunk_size)]
    # print("Averages of every device:")
    # print(averages)
    chunk_size_4 = 4
    averages = [sum(averages[i:i + chunk_size_4]) / chunk_size_4 for i in range(0, len(averages), chunk_size_4)]
    # print("Averages of every 4 devices:")
    print(averages)

del TrafficData

# <font color='Red'>***Neural Network***</font>
# ---
# --- 


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()        
        self.layer_1 = nn.Linear(98, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, 15) 
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.layer_out(x)        
        return x

# Random  = Net()
# for param_tensor in Random.state_dict():
#     print(param_tensor, "\t", Random.state_dict()[param_tensor].size())
# torch.save(Random.state_dict(), "0_Input_Random_model_Net.pth")

def train(net, trainloader, epochs: int, verbose=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    net.train()
    prediction_matrix = []
    actual_matrix= []
    acc_matrix = []
    loss_matrix=[]
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            predictions = torch.max(outputs.data, 1)[1]
            prediction_matrix.append(predictions.tolist())
            actual_matrix.append(labels.tolist())
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        loss_matrix.append(epoch_loss.tolist())
        acc_matrix.append(epoch_acc)
    return prediction_matrix, actual_matrix, acc_matrix, loss_matrix
def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    prediction_matrix = []
    actual_matrix= []
    acc_matrix = []
    loss_matrix=[]
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            prediction_matrix.append(predicted.tolist())
            actual_matrix.append(labels.tolist())
    loss /= len(testloader.dataset)
    accuracy = correct / total
    loss_matrix.append(loss)
    acc_matrix.append(accuracy)    
    print(f"Evaluation: eval loss {loss}, eval accuracy {accuracy}")
    return loss, accuracy, prediction_matrix, actual_matrix, acc_matrix, loss_matrix

prediction_dict= {}
actual_dict= {}
accuracy_dict= {}
loss_dict= {}

# <font color='Brown'>***Federated Learning Classes***</font>
# ---
# --- 


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, FL_Update):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.FL_Update = FL_Update
    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)
    def fit(self, parameters, config):
        local_epochs = config["local_epochs"]
        print(f"[Client {self.cid}, round {self.FL_Update}] fit, config: {config}")
        set_parameters(self.net, parameters)
        _1, _2, _3, _4 = train(self.net, self.trainloader, epochs=local_epochs)
        prediction_dict[f'C{self.cid}R{self.FL_Update}'] = _1
        actual_dict[f'C{self.cid}R{self.FL_Update}'] = _2
        accuracy_dict[f'C{self.cid}R{self.FL_Update}'] = _3
        loss_dict[f'C{self.cid}R{self.FL_Update}'] = _4
        # Save model updates (parameters)
        # update_filename = f'EdgeCooperation/Performance/Results/C{self.cid}R{self.FL_Update}_update.pkl'
        # with open(update_filename, 'wb') as update_outfile:
        #     pickle.dump(get_parameters(self.net), update_outfile)        

        return get_parameters(self.net), len(self.trainloader), {}
    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}")
        print(f"[Client {self.cid}] evaluate, config: {config}")
        return None

def retrieve_and_sort_client_updates(global_model, round_number, client_ids):
    client_updates = {}
    for cid in client_ids:
        update_filename = f'EdgeCooperation/Performance/Results/C{cid}R{round_number}_update.pkl'
        with open(update_filename, 'rb') as update_file:
            client_update = pickle.load(update_file)
            client_updates[cid] = client_update
    client_contributions = {cid: calculate_weight_magnitude(global_model, update) for cid, update in client_updates.items()}
    sorted_clients = sorted(client_contributions.items(), key=lambda x: x[1], reverse=True)
    least_contributing_clients = sorted_clients[-3:]
    return sorted_clients, least_contributing_clients

def calculate_weight_magnitude(global_model, client_update):
    """
    Calculate the L2 norm of the weight difference between the global model and client's updated model.
    
    Args:
    global_model (nn.Module): The global model before client update.
    client_update (list): List of numpy arrays representing client's updated model parameters.

    Returns:
    float: The L2 norm of the weight difference.
    """
    weight_diff = 0.0
    global_parameters = [param.detach().cpu().numpy() for param in global_model.parameters()]
    for global_param, client_param in zip(global_parameters, client_update):
        weight_diff += np.linalg.norm(global_param - client_param)
    return weight_diff


# <font color='Brown'>***Clients Functions***</font>
# ---
# --- 


def General_Client():
    def client_fn(cid: int, Round: int) -> FlowerClient:
        clients_ids_list = TrainingListPerRound[Round]
        if int(cid) in clients_ids_list:
            net = Net().to(DEVICE)
            trainloader = Dataloaders[Round][int(cid)]
            arg_ = Round
            return FlowerClient(cid, net, trainloader, arg_)
        else:
            raise ValueError(f"Client ID {cid} not found in the list for round {Round}")
    return client_fn

# <font color='Brown'>***FL Strategy***</font>
# ---
# --- 


Global_Models = {}
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, additional_argument, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.additional_argument = additional_argument
    def aggregate_fit(self, rnd: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], failures: List[BaseException]) -> Optional[fl.common.NDArrays]:
        aggregated_parameters_tuple = super().aggregate_fit(rnd, results, failures)
        aggregated_parameters, _ = aggregated_parameters_tuple
        if aggregated_parameters is not None:
            # print(f"Saving round {rnd} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_weights: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            Global_Models[self.additional_argument] = Net()
            params_dict = zip(Global_Models[self.additional_argument].state_dict().keys(), aggregated_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            Global_Models[self.additional_argument].load_state_dict(state_dict, strict=True)
            torch.save(Global_Models[self.additional_argument].state_dict(), f"{PATH}/GlobalModel_{self.additional_argument}.pth")
        return aggregated_parameters_tuple

def fit_config(server_round: int):
    """Return training configuration dict for each round.
    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "current_round": server_round,  # The current round of federated learning
        "local_epochs": EPOCHS #1 if rnd < 2 else 2,  # 
    }
    return config

# ***Running the Generalized FL Round***
# ---
# ---


print("Loading Initial Global Model")
Global_Models[0] = Net()
Global_Models[0].load_state_dict(torch.load("0_Input_Random_model_Net.pth"))
Global_Models[0].train()

TrainingListPerRound = {}
for Round in range(1, ROUNDS+1):
    TrainingListPerRound[Round] = []     
    for CLIENT in range (NUM_CLIENTS):
        TrainingListPerRound[Round].append(int(CLIENT))

for Round in range(1, ROUNDS+1):
    print("Starting FL Round: ", Round)
    strategy = SaveModelStrategy(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=0,  # Sample 50% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 10 clients for training
            min_evaluate_clients=0,  # Never sample less than 5 clients for evaluation
            min_available_clients=2,  # Wait until all 10 clients are available
            on_fit_config_fn=fit_config,
            initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Global_Models[Round-1])),
            additional_argument = Round
    )
    # print(f'Current training nodes at round {Round}: ', len(TrainingListPerRound[Round]), ' ', TrainingListPerRound[Round])
    client_fn = General_Client()
    fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, int(Round)),
        num_clients=int(NUM_CLIENTS),
        config=fl.server.ServerConfig(num_rounds=int(1)),
        client_resources={"num_cpus":16, "num_gpus":1}, 
        ray_init_args = {'num_cpus': 16, 'num_gpus': 1},
        strategy=strategy
    )
    print("End of FL Round: ", Round)
    print("Loading Global Model: ", Round)
    Global_Models[Round] = Net()
    Global_Models[Round].load_state_dict(torch.load(f"{PATH}/GlobalModel_{Round}.pth"))
    Global_Models[Round].train()

# <font color='Grey'>***Performance Testing***</font>
# ---
# --- 


import os
import glob

# Define the directory and file pattern
directory = PATH + '/'
pattern = "GlobalModel_*.pth"

# Find all matching files
files = glob.glob(os.path.join(directory, pattern))

# Extract numbers from file names
numbers = []
for file in files:
    base_name = os.path.basename(file)
    num_str = base_name.replace("GlobalModel_", "").replace(".pth", "")
    try:
        numbers.append(int(num_str))
    except ValueError:
        pass

# Determine the maximum number
max_num = max(numbers) if numbers else 0
print(max_num)

# Use the max_num in a loop
for num in range(1, max_num + 1):
    file_path = f"{PATH}/GlobalModel_{num}.pth"
    if os.path.exists(file_path):
        # Load the file or perform any operation you need
        print(f"Loading {file_path}")
    else:
        print(f"File {file_path} does not exist")

pred_test = {}
actual_test = {}
accuracy_test = {}
loss_test = {}
G = 0

for num in range(1, max_num+1):
    model = Net()
    model.load_state_dict(torch.load(f"{PATH}/GlobalModel_{num}.pth"))
    model.eval()
    
    prediction_matrix = []
    actual_matrix= []
    acc_matrix = []
    loss_matrix=[]
    G = G + 1
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in Dataloaders['Test']:
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            prediction_matrix.append(predicted.tolist())
            actual_matrix.append(labels.tolist())
    loss /= len(Dataloaders['Test'].dataset)
    accuracy = correct / total
    loss_matrix.append(loss)
    acc_matrix.append(accuracy) 

    pred_test[f'Global_{G}'] = prediction_matrix
    actual_test[f'Global_{G}'] = actual_matrix
    accuracy_test[f'Global_{G}'] = acc_matrix
    loss_test[f'Global_{G}'] = loss_matrix 

    filename = f'{PATH}/Global_{G}_pred'
    outfile = open(filename,'wb')
    pickle.dump(pred_test[f'Global_{G}'],outfile)
    outfile.close()

    filename = f'{PATH}/Global_{G}_actual'
    outfile = open(filename,'wb')
    pickle.dump(actual_test[f'Global_{G}'],outfile)
    outfile.close()

    filename = f'{PATH}/Global_{G}_accurracy'
    outfile = open(filename,'wb')
    pickle.dump(accuracy_test[f'Global_{G}'],outfile)
    outfile.close()

    filename = f'{PATH}/Global_{G}_loss'
    outfile = open(filename,'wb')
    pickle.dump(loss_test[f'Global_{G}'],outfile)
    outfile.close()