#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler   
from sklearn.model_selection import train_test_split
import random
import math
import pickle
import csv
import copy
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# DEVICE
NUM_CLIENTS = 48
ROUNDS = 40
BATCH_SIZE = 64
LEARNING_RATE = 0.0025
EPOCHS = 1
DATA_GROUPS = 40
BATCH_ROUND = 6
SIZE_ROUND = int(BATCH_ROUND * BATCH_SIZE * NUM_CLIENTS)
PATH = '20241215_3Epochs'

# ***Dataset Preparations***
# ---
base_dir = os.path.dirname(__file__)
TrafficData = {}
TrafficData['Dataset'] = {}
sets_names = ['30', '100', '70', '50', '120']
for DATA_NUM in sets_names:
    csv_path = os.path.join(base_dir, f"2_Dataset_5_Attack_{DATA_NUM}_normal.csv")
    TrafficData['Dataset'][DATA_NUM] = pd.read_csv(
        csv_path,
        low_memory=False,
        quoting=csv.QUOTE_NONE,
        on_bad_lines='skip'
    )
    print(DATA_NUM, TrafficData['Dataset'][DATA_NUM].shape)
for DATA_NUM in TrafficData['Dataset']:
    TrafficData['Dataset'][DATA_NUM] = TrafficData['Dataset'][DATA_NUM].sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

TrafficData['Split'] = {}
sets_training = ['30', '100', '70', '50']
for DATA_NUM in sets_training:
    TrafficData['Split'][DATA_NUM] = np.array_split(
        TrafficData['Dataset'][DATA_NUM], DATA_GROUPS
    )

TrafficData['Combined'] = pd.concat(
    [
        TrafficData['Split']['30'][0],
        TrafficData['Split']['100'][0],
        TrafficData['Split']['70'][0],
        TrafficData['Split']['50'][0],
    ]
).reset_index(drop=True)
for GROUP in range(1, DATA_GROUPS):
    TrafficData['Combined'] = pd.concat(
        [
            TrafficData['Combined'],
            TrafficData['Split']['30'][GROUP],
            TrafficData['Split']['100'][GROUP],
            TrafficData['Split']['70'][GROUP],
            TrafficData['Split']['50'][GROUP],
        ]
    ).reset_index(drop=True)
print(TrafficData['Combined'].shape)

TrafficData['Train'] = {}
TrafficData['Train']['X'] = TrafficData['Combined'].iloc[:, 0:-1]
TrafficData['Train']['y'] = TrafficData['Combined'].iloc[:, -1]
print(TrafficData['Train']['X'].shape)
print(TrafficData['Train']['y'].shape)

TrafficData['Test'] = {}
TrafficData['Test']['X'] = TrafficData['Dataset']['120'].iloc[:, 0:-1]
TrafficData['Test']['y'] = TrafficData['Dataset']['120'].iloc[:, -1]
print(TrafficData['Test']['X'].shape)
print(TrafficData['Test']['y'].shape)

scaler = StandardScaler()
model = scaler.fit(TrafficData['Train']['X'])
TrafficData['Train']['X'] = model.transform(TrafficData['Train']['X'])
TrafficData['Test']['X'] = model.transform(TrafficData['Test']['X'])

TrafficData['Train']['X'], TrafficData['Train']['y'] = np.array(
    TrafficData['Train']['X']
), np.array(TrafficData['Train']['y'])
print(type(TrafficData['Train']['X']), type(TrafficData['Train']['y']))
print(TrafficData['Train']['X'].shape, TrafficData['Train']['y'].shape)
TrafficData['Test']['X'], TrafficData['Test']['y'] = np.array(
    TrafficData['Test']['X']
), np.array(TrafficData['Test']['y'])
print(type(TrafficData['Test']['X']), type(TrafficData['Test']['y']))
print(TrafficData['Test']['X'].shape, TrafficData['Test']['y'].shape)

TrafficData['ROUNDS'] = {}
for ROUND in range(1, ROUNDS + 1):
    TrafficData['ROUNDS'][ROUND] = {}

SIZE_Demo = SIZE_ROUND
for ROUND in range(1, ROUNDS + 1):
    if ROUND == 1:
        TrafficData['ROUNDS'][ROUND]['X'] = TrafficData['Train']['X'][:SIZE_Demo]
        TrafficData['ROUNDS'][ROUND]['y'] = TrafficData['Train']['y'][:SIZE_Demo]
    else:
        print((SIZE_Demo - SIZE_ROUND), SIZE_Demo)
        TrafficData['ROUNDS'][ROUND]['X'] = TrafficData['Train']['X'][
            (SIZE_Demo - SIZE_ROUND) : SIZE_Demo
        ]
        TrafficData['ROUNDS'][ROUND]['y'] = TrafficData['Train']['y'][
            (SIZE_Demo - SIZE_ROUND) : SIZE_Demo
        ]
    SIZE_Demo = SIZE_Demo + SIZE_ROUND
for ROUND in TrafficData['ROUNDS']:
    print(
        ROUND,
        TrafficData['ROUNDS'][ROUND]['X'].shape,
        TrafficData['ROUNDS'][ROUND]['y'].shape,
    )

TrafficData['Train']['X'], TrafficData['Train']['y'] = TrafficData['Train']['X'][
    :SIZE_Demo
], TrafficData['Train']['y'][:SIZE_Demo]
print(TrafficData['Train']['X'].shape, TrafficData['Train']['y'].shape)

TrafficData['Train']['X'] = TrafficData['Train']['X'].astype(np.float32)
TrafficData['Test']['X'] = TrafficData['Test']['X'].astype(np.float32)

class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

TrafficData['trainsets'] = {}
for ROUND in range(1, ROUNDS + 1):
    TrafficData['trainsets'][ROUND] = ClassifierDataset(
        torch.from_numpy(TrafficData['ROUNDS'][ROUND]['X']).float(),
        torch.from_numpy(TrafficData['ROUNDS'][ROUND]['y']).long(),
    )
TrafficData['testset'] = ClassifierDataset(
    torch.from_numpy(TrafficData['Test']['X']).float(),
    torch.from_numpy(TrafficData['Test']['y']).long(),
)

def load_train(numberofclients, ROUND):
    portion_size = int(BATCH_ROUND * BATCH_SIZE)
    num_portions = int(NUM_CLIENTS)
    portion_indices = []
    for i in range(num_portions):
        start_idx = i * portion_size
        end_idx = (i + 1) * portion_size
        portion_indices.append(list(range(start_idx, min(end_idx, SIZE_ROUND))))
    portion_datasets = [
        Subset(TrafficData['trainsets'][ROUND], indices)
        for indices in portion_indices
    ]
    portion_loaders = [
        DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        for dataset in portion_datasets
    ]
    return portion_loaders

def load_test(numberofclients):
    testloader = DataLoader(
        TrafficData['testset'], batch_size=BATCH_SIZE, shuffle=False
    )
    return testloader

Dataloaders = {}
for ROUND in range(1, ROUNDS + 1):
    Dataloaders[ROUND] = load_train(NUM_CLIENTS, ROUND)
Dataloaders['Test'] = load_test(NUM_CLIENTS)


from collections import Counter
for CLUSTER in range(1, 9):
    DEVICE_PERCENTAGE = []
    for DEVICE__ in range(0, 12):
        for i, batch in enumerate(Dataloaders[CLUSTER][DEVICE__]):
            _, labels = batch
            class_counts = Counter(labels.numpy())
            total_records = sum(class_counts.values())
            class_0_count = class_counts.get(0, 0)
            percentage_class_0 = (class_0_count / total_records) * 100
            DEVICE_PERCENTAGE.append(percentage_class_0)
    chunk_size = 6
    averages = [
        sum(DEVICE_PERCENTAGE[i : i + chunk_size]) / chunk_size
        for i in range(0, len(DEVICE_PERCENTAGE), chunk_size)
    ]
    chunk_size_4 = 4
    averages = [
        sum(averages[i : i + chunk_size_4]) / chunk_size_4
        for i in range(0, len(averages), chunk_size_4)
    ]
    print(averages)

del TrafficData
