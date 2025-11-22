import os
import numpy as np
import pandas as pd
import pickle 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils import clip_grad_norm_

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from split_data import prepare_dataloaders

from metric_utils import plot_confusion_matrix_with_names, calculate_accuracy, calculate_weighted_precision, calculate_weighted_recall, calculate_weighted_f1
import matplotlib.pyplot as plt

import joblib

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

def scale_features(dataset, train_idx, scaler_path="scaler.pkl"):
    X_full = dataset.X.numpy()
    y_full = dataset.y.numpy()

    scaler = StandardScaler()
    scaler.fit(X_full[train_idx])

    X_scaled = scaler.transform(X_full)

    # Update dataset tensors with scaled features
    dataset.X = torch.tensor(X_scaled, dtype=torch.float32)
    dataset.y = torch.tensor(y_full, dtype=torch.long)

    # Save for later use (inference or retraining)
    joblib.dump(scaler, scaler_path)

    return dataset, scaler

def stratified_split(labels, test_size=0.2, validation_size=0.1, random_state=42):
    all_idx = np.arange(len(labels))

    # first split: test 
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss1.split(all_idx, labels))

    # second split: train + validation
    relative_val_size = validation_size / (1 - test_size) 
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, validation_idx = next(sss2.split(train_idx, labels[train_idx]))

    return train_idx, validation_idx, test_idx

def compute_metrics(all_preds, all_labels, num_classes, verbose=True):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy  = float(calculate_accuracy(all_labels, all_preds))
    precision = float(calculate_weighted_precision(all_labels, all_preds, num_classes))
    recall    = float(calculate_weighted_recall(all_labels, all_preds, num_classes))
    f1        = float(calculate_weighted_f1(all_labels, all_preds, num_classes))

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    if verbose:
        print("\nDetailed per-class report:")
        print(classification_report(all_labels, all_preds, zero_division=0))
    return accuracy, precision, recall, f1

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy, precision, recall, f1 = compute_metrics(all_preds, all_labels, model.num_classes)

    return np.array(all_labels), np.array(all_preds), accuracy, precision, recall, f1

def plot_confusion_matrix(
    actual,
    predicted,
    num_classes,
    results_dir,
    round_idx
):

    full_label_names = [
        'Normal',
        'DDoS_UDP',
        'DDoS_ICMP',
        'DDoS_TCP',
        'DDoS_HTTP',
        'Password',
        'Vulnerability_scanner',
        'SQL_injection'
    ]

    # Trim to the correct number of classes
    label_names = full_label_names[:num_classes]

    # label_numbers must correspond 0..num_classes-1
    label_numbers = list(range(num_classes))

    plot_confusion_matrix_with_names(
        actual=actual,
        predicted=predicted,
        label_numbers=label_numbers,
        label_names=label_names,
        title=f"Confusion Matrix - Global Model {round_idx}",
    )

    save_path = os.path.join(results_dir, f"CM_Global_{round_idx}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"[Saved] {save_path}")

def evaluate_all_global_models(
    model,
    test_loader,
    device,
    n_splits,
    models_dir,
    results_dir,
    test_csv_path,
    batch_size,
    lr,
):
    os.makedirs(results_dir, exist_ok=True)
    confusion_matrix_dir = "Data2/ConfusionMatrices"
    os.makedirs("Data2/ConfusionMatrices", exist_ok=True)

    print("\n=== Evaluating all saved global models on test set ===")

    for round_idx in range(1, n_splits + 1):
        checkpoint_path = os.path.join(models_dir, f"tabtransformer_split_{round_idx}.pt")
        if not os.path.exists(checkpoint_path):
            print(f"[Round {round_idx}] Checkpoint not found: {checkpoint_path}")
            continue

        # Load model snapshot for this round into the SAME model object
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)

        # Evaluate and collect labels/preds
        test_labels, test_preds, accuracy, precision, recall, f1 = evaluate(model, test_loader, device)

        plot_confusion_matrix(
            actual=test_labels,
            predicted=test_preds,
            num_classes=model.num_classes,
            results_dir=confusion_matrix_dir,
            round_idx=round_idx
        )

        print(
            f"[Evaluation Round {round_idx}] [Test Acc={accuracy:.4f}] [Model File Name: tabtransformer_split_{round_idx}.pt]"
            f"\nP={precision:.4f} R={recall:.4f} F1={f1:.4f}"
        )

        # Save per-round test metrics to CSV (optional but useful)
        save_metrics_to_csv(
            csv_path=test_csv_path,
            split_idx=round_idx,
            stage="test_round",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            batch_size=batch_size,
            lr=lr,
        )

        # Save labels & preds in the format metric_utils expects
        actual_path = os.path.join(results_dir, f"Global_{round_idx}_actual")
        pred_path   = os.path.join(results_dir, f"Global_{round_idx}_pred")

        # metric_utils flattens list-of-lists, so we save [list] not just list
        with open(actual_path, "wb") as f:
            pickle.dump([test_labels.tolist()], f)

        with open(pred_path, "wb") as f:
            pickle.dump([test_preds.tolist()], f)

def save_metrics_to_csv(
    csv_path,
    split_idx,
    stage,
    accuracy,
    precision,
    recall,
    f1,
    batch_size,
    lr,
):
  
    log_file = "Data2/TrainingMetricsLog/training_metrics_log.csv"

    log_entry = pd.DataFrame([{
        "Dataset": csv_path,
        "Split": split_idx,
        "Stage": stage,             
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_score": round(f1, 4),
        "Batch_Size": batch_size,
        "Learning_Rate": lr,
    }])

    log_entry.to_csv(
        log_file,
        mode="a",
        index=False,
        header=not os.path.exists(log_file),
    )

def train(
    train_csv_path="Data2/2_Dataset_5_Attack_30_normal.csv",
    test_csv_path="Data2/2_Dataset_5_Attack_120_normal.csv",
    target_column="label", 
    n_splits=40,
    batch_size=32, 
    lr=1e-4
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train_loaders, test_loader, y_train = prepare_dataloaders(
        train_csv_path=train_csv_path,
        test_csv_path=test_csv_path,
        target_column=target_column,
        n_splits=n_splits,
        batch_size=batch_size,
        scaler_path="scaler_sequential.pkl",
        random_state=42,
    )

    sample_batch_x, _ = next(iter(train_loaders[0]))
    num_features = sample_batch_x.size(1)

    classes = np.unique(y_train)
    num_classes = len(classes)

    model = TabTransformer(
        num_features=num_features,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    models_dir = "Data2/Models"
    results_dir = "Data2/GlobalModelResults"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Training Loop
    for train_loader_idx, train_loader in enumerate(train_loaders, start=1):
        print(f"\n=== Training on split {train_loader_idx}/{n_splits} ===")

        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            batch_preds = logits.argmax(dim=1)
            all_preds.extend(batch_preds.detach().cpu().numpy())
            all_labels.extend(y_batch.detach().cpu().numpy())

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"[Split {train_loader_idx}/{n_splits}] Training Loss: {avg_loss:.4f}")

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        train_accuracy, train_precision, train_recall, train_f1 = compute_metrics(all_preds, all_labels, num_classes)
    
        save_metrics_to_csv(
            csv_path=train_csv_path,      
            split_idx=train_loader_idx,
            stage="train",
            accuracy=train_accuracy,
            precision=train_precision,
            recall=train_recall,
            f1=train_f1,
            batch_size=batch_size,
            lr=lr,
        )

        checkpoint_path = os.path.join(models_dir, f"tabtransformer_split_{train_loader_idx}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model after split {train_loader_idx} -> tabtransformer_split_{train_loader_idx}.pt")

    evaluate_all_global_models(
        model=model,
        test_loader=test_loader,
        device=device,
        n_splits=n_splits,
        models_dir=models_dir,
        results_dir=results_dir,
        test_csv_path=test_csv_path,
        batch_size=batch_size,
        lr=lr,
    )

    print("\n=== Final evaluation on test CSV ===")
    _, _, test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_loader, device)

    # Save metrics to CSV
    save_metrics_to_csv(
        csv_path=test_csv_path,
        split_idx=n_splits,      
        stage="final_test",
        accuracy=test_accuracy,
        precision=test_precision,
        recall=test_recall,
        f1=test_f1,
        batch_size=batch_size,
        lr=lr,
    )

    print(
        f"Test Accuracy: {test_accuracy*100:.2f}% | "
        f"Precision: {test_precision:.4f} | "
        f"Recall: {test_recall:.4f} | "
        f"F1: {test_f1:.4f}"
    )

    return model

if __name__ == "__main__":
    model = train(
        train_csv_path="Data2/2_Dataset_5_Attack_30_normal.csv",
        test_csv_path="Data2/2_Dataset_5_Attack_120_normal.csv",
        target_column="Attack_type",
        n_splits=40,
        batch_size=128,
        lr=3e-5,
    )
    torch.save(model.state_dict(), "Data2/Models/tabtransformer_final.pt")