import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib


def prepare_dataloaders(
    train_csv_path,
    test_csv_path,
    target_column="Attack_type",
    n_splits=40,
    batch_size=128,
    scaler_path="scaler_sequential.pkl",
    random_state=42,
):

    # Load raw data 
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Shuffle the training data 
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Separate features/labels
    y_train = train_df[target_column].values
    X_train = train_df.drop(columns=[target_column]).values

    y_test = test_df[target_column].values
    X_test = test_df.drop(columns=[target_column]).values

    # Scale features 
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if scaler_path is not None:
        joblib.dump(scaler, scaler_path)

    # Split train into n_splits chunks 
    X_splits = np.array_split(X_train_scaled, n_splits)
    y_splits = np.array_split(y_train, n_splits)

    train_loaders = []
    for X_chunk, y_chunk in zip(X_splits, y_splits):
        tensor_x = torch.tensor(X_chunk, dtype=torch.float32)
        tensor_y = torch.tensor(y_chunk, dtype=torch.long)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_loaders.append(loader)

    # Test loader over entire test set 
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loaders, test_loader, y_train
