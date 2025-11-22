import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report

class PacketDataset(Dataset):
    def __init__(self, csv, target_column):
        df = pd.read_csv(csv)
        self.y = torch.tensor(df[target_column].values, dtype=torch.long)
        self.X = torch.tensor(df.drop(columns=[target_column]).values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ff_hidden_dim=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(dim, ff_hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class TabTransformer(nn.Module):
    def __init__(
        self,
        num_features=98,
        num_classes=6,
        dim=128,
        depth=4,
        num_heads=8,
        ff_hidden_dim=256,
        dropout=0.1
    ):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.feature_embed = nn.Linear(1, dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_hidden_dim, dropout) for _ in range(depth)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.num_features, f"Expected input with {self.num_features} features, got {tuple(x.shape)}"
        x = x.unsqueeze(-1)
        x = self.feature_embed(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        assert logits.size(1) == self.num_classes, f"Expected {self.num_classes} logits, got {logits.size(1)}"
        return logits

def evaluate_and_compute_metrics(model, dataloader, device, verbose=True):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = float((all_preds == all_labels).mean())
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    if verbose:
        print("\nDetailed per-class report:")
        print(classification_report(all_labels, all_preds, zero_division=0))

    return accuracy, precision, recall, f1

def save_metrics_to_csv(csv_path, accuracy, precision, recall, f1, epoch, batch_size, lr):
    log_file = "training_metrics_log.csv"
    log_entry = pd.DataFrame([{
        "Dataset": csv_path,
        "Epoch": epoch,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_score": round(f1, 4),
        "Batch_Size": batch_size,
        "Learning_Rate": lr
    }])
    log_entry.to_csv(log_file, mode='a', index=False, header=not os.path.exists(log_file))

def train(train_csv_path, test_csv_path, target_column="label", epochs=10, batch_size=32, lr=1e-4):
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train_dataset = PacketDataset(train_csv_path, target_column)
    test_dataset = PacketDataset(test_csv_path, target_column)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_train = train_dataset.y.numpy()

    model = TabTransformer(num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        accuracy, precision, recall, f1 = evaluate_and_compute_metrics(model, test_loader, device, verbose=True)

        print(f"Epoch {epoch+1}/{epochs} | Loss={avg_loss:.4f} | Accuracy={accuracy*100:.2f}% | P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
        save_metrics_to_csv(train_csv_path, accuracy, precision, recall, f1, epoch+1, batch_size, lr)

    return model

if __name__ == "__main__":
    model = train(
        "Data/2_Dataset_1_Attack_30_normal.csv",
        "Data/2_Dataset_1_Attack_120_normal.csv",
        target_column="Attack_type",
        epochs=10,
        batch_size=128,
        lr=1e-4
    )
    torch.save(model.state_dict(), "tabtransformer_model.pt")
