# ==============================
# IMPORTS + CONFIG
# ==============================

import os, json, glob, pickle
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()

print("Running on:", DEVICE)

with open("config.json", "r") as f:
    CFG = json.load(f)

INPUT_SIZE = CFG["INPUT_SIZE"]
HIDDEN1_SIZE = CFG["HIDDEN1_SIZE"]
HIDDEN2_SIZE = CFG["HIDDEN2_SIZE"]
EMB_DIM = CFG["EMB_DIM"]
MLP_DIM = CFG["MLP_DIM"]
NUM_ATCKS = CFG["NUM_ATCKS"]

DATA_DIR = "./inference_input"
OUTPUT_DIR = "./inference_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 50000
MAX_WORKERS = 4 if not USE_GPU else 2


# ==============================
# LABELS
# ==============================

LABEL_NAMES = [
    "Normal","DDoS_UDP","DDoS_ICMP","DDoS_TCP","DDoS_HTTP","Password",
    "Vulnerability_scanner","SQL_injection","Uploading","Backdoor",
    "Port_Scanning","XSS","Ransomware","MITM","OS_Fingerprinting"
]

label_map = dict(enumerate(LABEL_NAMES))


# ==============================
# MODELS
# ==============================

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(INPUT_SIZE, HIDDEN2_SIZE)
        self.layer_2 = nn.Linear(HIDDEN2_SIZE, HIDDEN1_SIZE)
        self.layer_out = nn.Linear(HIDDEN1_SIZE, NUM_ATCKS + 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_out(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


class TabTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_projection = nn.Linear(1, EMB_DIM)
        self.feature_id_emb = nn.Parameter(torch.randn(INPUT_SIZE, EMB_DIM))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMB_DIM,
            nhead=4,
            dim_feedforward=MLP_DIM,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.output_head = nn.Sequential(
            nn.Linear(EMB_DIM, MLP_DIM),
            nn.ReLU(),
            nn.Linear(MLP_DIM, NUM_ATCKS + 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        emb = self.feature_projection(x)
        emb = emb + self.feature_id_emb.unsqueeze(0)
        z = self.transformer(emb)
        return self.output_head(z.mean(dim=1))


# ==============================
# LOAD MODEL
# ==============================

def load_model():
    model_path = "./GlobalModel_40.pth"
    print("Loading model:", model_path)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint if "state_dict" not in checkpoint else checkpoint["state_dict"]

    # 🔥 AUTO DETECT MODEL TYPE
    if any("transformer" in k for k in state_dict.keys()):
        print("Detected: TabTransformer")
        model = TabTransformer()
    else:
        print("Detected: DNN")
        model = Net()

    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()

    if hasattr(torch, "compile"):
        model = torch.compile(model)

    if USE_GPU:
        model = model.half()

    return model


model = load_model()


# ==============================
# LOAD SCALER
# ==============================

with open("./scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

train_features = scaler.feature_names_in_


# ==============================
# PREPROCESS (THREAD SAFE)
# ==============================

def preprocess_chunk(chunk):
    X = chunk.iloc[:, :-1]
    y = chunk.iloc[:, -1].astype(int).values

    X = X.reindex(columns=train_features, fill_value=0)
    X_scaled = scaler.transform(X)

    return chunk, X_scaled, y


# ==============================
# MAIN EVALUATION
# ==============================

def evaluate():

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    for file_path in csv_files:

        file_name = os.path.basename(file_path).replace(".csv", "")
        print(f"\nProcessing: {file_name}")

        chunks = list(pd.read_csv(file_path, chunksize=CHUNK_SIZE))

        # ----------------------
        # PARALLEL PREPROCESS
        # ----------------------
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            processed = list(executor.map(preprocess_chunk, chunks))

        results = []
        all_preds = []
        all_actuals = []

        # ----------------------
        # SAFE INFERENCE
        # ----------------------
        for i, (chunk, X_scaled, y) in enumerate(processed):

            print(f"  inference chunk {i}")

            X_tensor = torch.tensor(X_scaled)

            if USE_GPU:
                X_tensor = X_tensor.half().to(DEVICE)
            else:
                X_tensor = X_tensor.float().to(DEVICE)

            with torch.no_grad():
                preds = torch.argmax(model(X_tensor), dim=1).cpu().numpy()

            chunk_out = chunk.copy()

            chunk_out["Actual_Label_ID"] = y
            chunk_out["Predicted_Label_ID"] = preds
            chunk_out["Actual_Label_Name"] = [label_map[i] for i in y]
            chunk_out["Predicted_Label_Name"] = [label_map[i] for i in preds]
            chunk_out["Correct"] = (y == preds)

            results.append(chunk_out)
            all_preds.extend(preds)
            all_actuals.extend(y)

        # ----------------------
        # FINAL OUTPUT
        # ----------------------
        final_df = pd.concat(results, ignore_index=True)
        final_df.insert(0, "Row_Index", range(len(final_df)))

        acc = accuracy_score(all_actuals, all_preds)
        print(f"Accuracy: {acc:.4f}")

        out_path = os.path.join(OUTPUT_DIR, f"{file_name}_analysis.csv")
        final_df.to_csv(out_path, index=False)

        print("Saved →", out_path)


# ==============================
# RUN
# ==============================

evaluate()