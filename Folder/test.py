import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import pandas as pd
import numpy as np
import os

# Dataset
class Dataset(Dataset):
    def __init__(self, csv, target_column):
        df = pd.read_csv(csv)
        self.y = torch.tensor(df[target_column].values, dtype=torch.long)
        self.X = torch.tensor(df.drop(columns=[target_column]).values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# FeedForward Layer
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



# Transformer Block
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



# TabTransformer Model
class TabTransformer(nn.Module):
    def __init__(
        self,
        num_features=98,
        num_classes=6,
        dim=128,
        depth=4,
        num_heads=8,
        ff_hidden_dim=256,
        dropout=0.22
    ):
        super().__init__()
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
        x = x.unsqueeze(-1)  # (batch, features, 1)
        x = self.feature_embed(x)  # (batch, features, dim)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.mean(dim=1)  # Mean pooling
        return self.classifier(x)


# Evaluation Metrics Function
def evaluate_metrics(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # Compute metrics
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nDetailed per-class report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    return precision, recall, f1


# Save Metrics to CSV
def save_metrics_to_csv(csv_path, accuracy, precision, recall, f1, epochs, batch_size, lr):
    log_file = "training_metrics_log.csv"

    # Prepare data
    log_entry = pd.DataFrame([{
        "Dataset": csv_path,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_score": round(f1, 4),
        "Epochs": epochs,
        "Batch_Size": batch_size,
        "Learning_Rate": lr
    }])

    # Append to existing file or create new one
    if os.path.exists(log_file):
        log_entry.to_csv(log_file, mode='a', index=False, header=False)
    else:
        log_entry.to_csv(log_file, index=False)


# Training Function
def train(csv_path, target_column="label", epochs=10, batch_size=32, lr=1e-4):
    dataset = Dataset(csv_path, target_column)

    # Split dataset 80/20
    n_train = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train
    train_ds, test_ds = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = TabTransformer(num_features=98, num_classes=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    acc = correct / total
    print(f"\nTest Accuracy: {acc*100:.2f}%")

    # Compute additional metrics
    precision, recall, f1 = evaluate_metrics(model, test_loader, device)

    # Save metrics to CSV
    save_metrics_to_csv(csv_path, acc, precision, recall, f1, epochs, batch_size, lr)

    return model


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    model = train("Folder/2_Dataset_5_Attack_30_normal.csv",target_column="Attack_type",epochs=50,batch_size=128,lr=1e-4)
    torch.save(model.state_dict(), "tabtransformer_model.pt")
    
