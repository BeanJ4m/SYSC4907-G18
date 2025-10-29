import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def load_data(batch_size=256):
    X_train = torch.tensor(np.load("X_train.npy"), dtype=torch.float32)
    y_train = torch.tensor(np.load("y_train.npy"), dtype=torch.long)
    X_test  = torch.tensor(np.load("X_test.npy"), dtype=torch.float32)
    y_test  = torch.tensor(np.load("y_test.npy"), dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    num_classes = len(torch.unique(y_train))
    input_size = X_train.shape[1]
    return train_loader, test_loader, num_classes, input_size
