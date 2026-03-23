#!/usr/bin/env python3
"""
Federated training client for the IoT IDS demo.

Compatible with demo_server.py.

Features
- Flower NumPyClient
- DNN / TabTransformer support
- Optional local system monitoring
- Works with static CSV shards or Pi-generated CSVs
- Handles server-driven hyperparameter updates per round
- Rebuilds the local model if server parameters change shape

Expected CSV format
- Last column is the integer class label
- All preceding columns are numeric features

Recommended deployment on Raspberry Pi
- One CSV shard per client, or a rolling CSV produced by extract.py / ids_linux.py
- Example: client_01_train.csv, client_01_test.csv
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from demo_data_stage import (
    build_client_round_loaders,
    build_eval_loader,
    client_id_to_index,
    prepare_notebook_data,
)

try:
    from system_monitor import SystemMonitor
except Exception:
    SystemMonitor = None  # type: ignore


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ClassifierDataset(Dataset):
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        self.x_data = torch.from_numpy(x_data).float()
        self.y_data = torch.from_numpy(y_data).long()

    def __getitem__(self, index: int):
        return self.x_data[index], self.y_data[index]

    def __len__(self) -> int:
        return len(self.x_data)


class DNNNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden1_size: int,
        hidden2_size: int,
        output_size: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Linear(input_size, hidden1_size),
            nn.ReLU(),
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.extend(
            [
                nn.Linear(hidden1_size, hidden2_size),
                nn.ReLU(),
            ]
        )
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden2_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TabTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        emb_dim: int,
        mlp_dim: int,
        num_classes: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError("EMB_DIM must be divisible by NUM_HEADS for TabTransformer")

        self.feature_projection = nn.Linear(1, emb_dim)
        self.feature_id_emb = nn.Parameter(torch.randn(num_features, emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        emb = self.feature_projection(x)
        emb = emb + self.feature_id_emb.unsqueeze(0)
        z = self.transformer(emb)
        pooled = z.mean(dim=1)
        return self.output_head(pooled)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: Sequence[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def infer_output_size(cfg: Dict[str, Any], y: Optional[np.ndarray] = None) -> int:
    if "OUTPUT_SIZE" in cfg and int(cfg["OUTPUT_SIZE"]) > 0:
        return int(cfg["OUTPUT_SIZE"])
    if "NUM_ATCKS" in cfg:
        return int(cfg["NUM_ATCKS"]) + 1
    if y is None:
        raise ValueError("Could not infer OUTPUT_SIZE; set OUTPUT_SIZE or NUM_ATCKS in config")
    return int(np.max(y)) + 1


class ClientData:
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        feature_count: int,
        output_size: int,
        train_size: int,
        test_size: int,
    ) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.feature_count = feature_count
        self.output_size = output_size
        self.train_size = train_size
        self.test_size = test_size


def resolve_client_csvs(
    cfg: Dict[str, Any],
    client_id: str,
    train_csv: Optional[str],
    test_csv: Optional[str],
) -> Tuple[Path, Optional[Path]]:
    if train_csv:
        return Path(train_csv).resolve(), Path(test_csv).resolve() if test_csv else None

    data_dir = Path(cfg.get("DATA_DIR", ".")).resolve()
    client_tag = str(client_id)

    train_candidates = [
        data_dir / f"client_{client_tag}_train.csv",
        data_dir / f"client_{client_tag}.csv",
        data_dir / f"train_client_{client_tag}.csv",
        data_dir / f"client{client_tag}_train.csv",
        data_dir / f"{client_tag}_train.csv",
    ]
    test_candidates = [
        data_dir / f"client_{client_tag}_test.csv",
        data_dir / f"test_client_{client_tag}.csv",
        data_dir / f"client{client_tag}_test.csv",
        data_dir / f"{client_tag}_test.csv",
    ]

    train_path = next((p for p in train_candidates if p.exists()), None)
    test_path = next((p for p in test_candidates if p.exists()), None)

    if train_path is None:
        raise FileNotFoundError(
            f"No local client CSV found for client_id={client_id}. "
            f"Checked under {data_dir}. Use --train-csv to specify the file explicitly."
        )
    return train_path, test_path


def load_client_data(
    cfg: Dict[str, Any],
    client_id: str,
    train_csv: Optional[str],
    test_csv: Optional[str],
    test_split: float,
) -> ClientData:
    use_notebook_stage = bool(cfg.get("USE_NOTEBOOK_DATA_STAGE", True))
    if use_notebook_stage:
        prepared = prepare_notebook_data(cfg)
        client_index = client_id_to_index(client_id, int(cfg["NUM_CLIENTS"]))
        round_loaders = build_client_round_loaders(
            prepared=prepared,
            round_idx=1,
            num_clients=int(cfg["NUM_CLIENTS"]),
            batch_size=int(cfg.get("BATCH_SIZE", 64)),
        )
        test_loader = build_eval_loader(prepared, batch_size=int(cfg.get("BATCH_SIZE", 64)))
        train_loader = round_loaders[client_index]
        train_size = len(train_loader.dataset)
        test_size = len(test_loader.dataset)
        return ClientData(
            train_loader=train_loader,
            test_loader=test_loader,
            feature_count=prepared.feature_count,
            output_size=prepared.output_size,
            train_size=train_size,
            test_size=test_size,
        )

    train_path, test_path = resolve_client_csvs(cfg, client_id, train_csv, test_csv)

    train_df = pd.read_csv(train_path, low_memory=False, quoting=csv.QUOTE_NONE, on_bad_lines="skip")
    if train_df.shape[1] < 2:
        raise ValueError(f"Training CSV must have at least 2 columns: {train_path}")

    if test_path is not None:
        test_df = pd.read_csv(test_path, low_memory=False, quoting=csv.QUOTE_NONE, on_bad_lines="skip")
        if test_df.shape[1] != train_df.shape[1]:
            raise ValueError("Train/test CSVs must have the same number of columns")

        x_train = train_df.iloc[:, :-1].to_numpy()
        y_train = train_df.iloc[:, -1].to_numpy().astype(np.int64)
        x_test = test_df.iloc[:, :-1].to_numpy()
        y_test = test_df.iloc[:, -1].to_numpy().astype(np.int64)
    else:
        x_all = train_df.iloc[:, :-1].to_numpy()
        y_all = train_df.iloc[:, -1].to_numpy().astype(np.int64)
        stratify = y_all if len(np.unique(y_all)) > 1 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x_all,
            y_all,
            test_size=test_split,
            random_state=int(cfg.get("SEED", 42)),
            stratify=stratify,
        )

    feature_count = int(x_train.shape[1])
    output_size = infer_output_size(cfg, np.concatenate([y_train, y_test]))

    scaler_name = str(cfg.get("SCALER", "auto")).lower()
    if scaler_name == "auto":
        scaler = StandardScaler() if str(cfg.get("MODE", "DNN")).upper() == "TT" else MinMaxScaler()
    elif scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported SCALER={cfg.get('SCALER')}")

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    batch_size = int(cfg.get("BATCH_SIZE", 64))
    train_loader = DataLoader(
        ClassifierDataset(np.asarray(x_train, dtype=np.float32), y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        ClassifierDataset(np.asarray(x_test, dtype=np.float32), y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    return ClientData(
        train_loader=train_loader,
        test_loader=test_loader,
        feature_count=feature_count,
        output_size=output_size,
        train_size=len(y_train),
        test_size=len(y_test),
    )


def build_model(cfg: Dict[str, Any], feature_count: int, output_size: int) -> nn.Module:
    mode = str(cfg["MODE"]).upper()
    if mode == "DNN":
        return DNNNet(
            input_size=feature_count,
            hidden1_size=int(cfg.get("HIDDEN1_SIZE", 64)),
            hidden2_size=int(cfg.get("HIDDEN2_SIZE", 32)),
            output_size=output_size,
            dropout_rate=float(cfg.get("DROPOUT_RATE", 0.0)),
        )
    if mode == "TT":
        emb_dim = int(cfg.get("EMB_DIM", 56))
        num_heads = int(cfg.get("NUM_HEADS", 4))
        if emb_dim % num_heads != 0:
            raise ValueError(f"EMB_DIM={emb_dim} must be divisible by NUM_HEADS={num_heads}")
        return TabTransformer(
            num_features=feature_count,
            emb_dim=emb_dim,
            mlp_dim=int(cfg.get("MLP_DIM", 112)),
            num_classes=output_size,
            num_layers=int(cfg.get("NUM_LAYERS", 3)),
            num_heads=num_heads,
            dropout=float(cfg.get("DROPOUT_RATE", 0.1)),
        )
    raise ValueError(f"Unsupported MODE={cfg['MODE']}")


def infer_dnn_cfg_from_parameters(
    parameters: Sequence[np.ndarray],
    feature_count: int,
    output_size: int,
    fallback_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    if len(parameters) < 5:
        raise ValueError("Not enough parameters to infer DNN architecture")
    hidden1_size = int(parameters[0].shape[0])
    hidden2_size = int(parameters[2].shape[0])
    return {
        **fallback_cfg,
        "MODE": "DNN",
        "INPUT_SIZE": feature_count,
        "OUTPUT_SIZE": output_size,
        "HIDDEN1_SIZE": hidden1_size,
        "HIDDEN2_SIZE": hidden2_size,
    }


def infer_tt_cfg_from_parameters(
    parameters: Sequence[np.ndarray],
    feature_count: int,
    output_size: int,
    fallback_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    if len(parameters) < 5:
        raise ValueError("Not enough parameters to infer TT architecture")

    emb_dim = int(parameters[0].shape[0])
    mlp_dim = int(parameters[-2].shape[0]) if len(parameters[-2].shape) == 2 else int(fallback_cfg.get("MLP_DIM", 112))

    num_layers = int(fallback_cfg.get("NUM_LAYERS", 3))
    if len(parameters) >= 3:
        # Estimate transformer depth from state_dict length only if a standard layout is used.
        # Fallback remains the local config value.
        num_layers = int(fallback_cfg.get("NUM_LAYERS", 3))

    num_heads = int(fallback_cfg.get("NUM_HEADS", 4))
    if emb_dim % max(num_heads, 1) != 0:
        valid_heads = [h for h in (8, 7, 6, 5, 4, 3, 2, 1) if emb_dim % h == 0]
        num_heads = valid_heads[0] if valid_heads else 1

    return {
        **fallback_cfg,
        "MODE": "TT",
        "INPUT_SIZE": feature_count,
        "OUTPUT_SIZE": output_size,
        "EMB_DIM": emb_dim,
        "MLP_DIM": mlp_dim,
        "NUM_LAYERS": num_layers,
        "NUM_HEADS": num_heads,
    }


def rebuild_model_from_parameters(
    mode: str,
    parameters: Sequence[np.ndarray],
    feature_count: int,
    output_size: int,
    cfg: Dict[str, Any],
) -> nn.Module:
    mode = mode.upper()
    if mode == "DNN":
        inferred_cfg = infer_dnn_cfg_from_parameters(parameters, feature_count, output_size, cfg)
    elif mode == "TT":
        inferred_cfg = infer_tt_cfg_from_parameters(parameters, feature_count, output_size, cfg)
    else:
        raise ValueError(f"Unsupported MODE={mode}")

    model = build_model(inferred_cfg, feature_count=feature_count, output_size=output_size)
    set_parameters(model, parameters)
    return model


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def build_loss(y_labels: Iterable[int], cfg: Dict[str, Any]) -> nn.Module:
    if not bool(cfg.get("USE_CLASS_WEIGHTS", False)):
        return nn.CrossEntropyLoss()

    labels = np.asarray(list(y_labels), dtype=np.int64)
    if labels.size == 0:
        return nn.CrossEntropyLoss()

    num_classes = int(cfg.get("OUTPUT_SIZE", labels.max() + 1))
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))


def train_one_round(
    model: nn.Module,
    train_loader: DataLoader,
    learning_rate: float,
    epochs: int,
    weight_decay: float,
    cfg: Dict[str, Any],
) -> Tuple[float, float]:
    model.train()
    model.to(DEVICE)

    all_labels: List[int] = []
    if bool(cfg.get("USE_CLASS_WEIGHTS", False)):
        for _, labels in train_loader:
            all_labels.extend(labels.tolist())
    criterion = build_loss(all_labels, cfg)
    optimizer = build_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)

    total_examples = 0
    correct = 0
    loss_sum = 0.0

    for _ in range(max(1, epochs)):
        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_examples += batch_size
            loss_sum += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

    if total_examples == 0:
        return 0.0, 0.0
    return loss_sum / total_examples, correct / total_examples


def evaluate_model(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.to(DEVICE)

    total = 0
    correct = 0
    loss_sum = 0.0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(features)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            batch_size = labels.size(0)
            total += batch_size
            correct += (preds == labels).sum().item()
            loss_sum += loss.item() * batch_size

    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total


class DemoClient(fl.client.NumPyClient):
    def __init__(
        self,
        cfg: Dict[str, Any],
        client_id: str,
        data: ClientData,
        output_dir: Path,
    ) -> None:
        self.cfg = dict(cfg)
        self.client_id = str(client_id)
        self.data = data
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cfg["INPUT_SIZE"] = int(self.cfg.get("INPUT_SIZE", data.feature_count))
        self.cfg["OUTPUT_SIZE"] = int(self.cfg.get("OUTPUT_SIZE", data.output_size))
        self.cfg["MODE"] = str(self.cfg.get("MODE", "DNN")).upper()

        if self.cfg["INPUT_SIZE"] != data.feature_count:
            raise ValueError(
                f"INPUT_SIZE mismatch: config has {self.cfg['INPUT_SIZE']}, local data has {data.feature_count} features"
            )

        self.model = build_model(self.cfg, data.feature_count, data.output_size)
        self.weight_decay = float(self.cfg.get("WEIGHT_DECAY", 0.0))
        self.fit_calls = 0
        self.use_notebook_stage = bool(self.cfg.get("USE_NOTEBOOK_DATA_STAGE", True))
        self.prepared = prepare_notebook_data(self.cfg) if self.use_notebook_stage else None
        self.client_index = client_id_to_index(self.client_id, int(self.cfg.get("NUM_CLIENTS", 1))) if self.use_notebook_stage else 0

    def _ensure_model_matches(self, parameters: Sequence[np.ndarray], mode: str) -> None:
        current_shapes = [tuple(p.shape) for p in get_parameters(self.model)]
        incoming_shapes = [tuple(np.asarray(p).shape) for p in parameters]
        if current_shapes != incoming_shapes:
            print(
                f"[Client {self.client_id}] Rebuilding model for new parameter shapes "
                f"({self.cfg.get('MODE')} -> {mode})"
            )
            self.cfg["MODE"] = mode.upper()
            self.model = rebuild_model_from_parameters(
                mode=mode,
                parameters=parameters,
                feature_count=self.data.feature_count,
                output_size=self.data.output_size,
                cfg=self.cfg,
            )
        else:
            set_parameters(self.model, parameters)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return get_parameters(self.model)

    def fit(self, parameters: Sequence[np.ndarray], config: Dict[str, Any]):
        mode = str(config.get("mode", self.cfg.get("MODE", "DNN"))).upper()
        local_epochs = int(config.get("local_epochs", self.cfg.get("EPOCHS", 1)))
        learning_rate = float(config.get("learning_rate", self.cfg.get("LEARNING_RATE", 1e-3)))
        batch_size = int(config.get("batch_size", self.cfg.get("BATCH_SIZE", 64)))
        current_round = int(config.get("current_round", 0))

        self.cfg["MODE"] = mode
        self.cfg["EPOCHS"] = local_epochs
        self.cfg["LEARNING_RATE"] = learning_rate
        self.cfg["BATCH_SIZE"] = batch_size

        if self.use_notebook_stage and self.prepared is not None:
            round_loaders = build_client_round_loaders(
                prepared=self.prepared,
                round_idx=current_round,
                num_clients=int(self.cfg.get("NUM_CLIENTS", 1)),
                batch_size=batch_size,
            )
            self.data.train_loader = round_loaders[self.client_index]
            self.data.test_loader = build_eval_loader(self.prepared, batch_size=batch_size)
            self.data.train_size = len(self.data.train_loader.dataset)
            self.data.test_size = len(self.data.test_loader.dataset)
        elif batch_size != self.data.train_loader.batch_size:
            self.data.train_loader = DataLoader(
                self.data.train_loader.dataset,
                batch_size=batch_size,
                shuffle=True,
            )
            self.data.test_loader = DataLoader(
                self.data.test_loader.dataset,
                batch_size=batch_size,
                shuffle=False,
            )

        self._ensure_model_matches(parameters, mode)

        started = time.time()
        train_loss, train_acc = train_one_round(
            model=self.model,
            train_loader=self.data.train_loader,
            learning_rate=learning_rate,
            epochs=local_epochs,
            weight_decay=self.weight_decay,
            cfg=self.cfg,
        )
        elapsed = time.time() - started
        self.fit_calls += 1

        round_payload = {
            "client_id": self.client_id,
            "round": current_round,
            "mode": mode,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "train_examples": int(self.data.train_size),
            "epochs": int(local_epochs),
            "learning_rate": float(learning_rate),
            "round_time_sec": float(round(elapsed, 4)),
        }
        with (self.output_dir / f"client_{self.client_id}_round_{current_round}.json").open("w", encoding="utf-8") as f:
            json.dump(round_payload, f, indent=2)

        print(
            f"[Client {self.client_id}] round={current_round} mode={mode} "
            f"loss={train_loss:.6f} acc={train_acc:.4f} time={elapsed:.2f}s"
        )

        return get_parameters(self.model), self.data.train_size, {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "local_accuracy": float(train_acc),
            "client_id": self.client_id,
            "round_time_sec": float(round(elapsed, 4)),
        }

    def evaluate(self, parameters: Sequence[np.ndarray], config: Dict[str, Any]):
        mode = str(config.get("mode", self.cfg.get("MODE", "DNN"))).upper()
        self._ensure_model_matches(parameters, mode)
        loss, accuracy = evaluate_model(self.model, self.data.test_loader)
        print(f"[Client {self.client_id}] eval loss={loss:.6f} acc={accuracy:.4f}")
        return float(loss), self.data.test_size, {"accuracy": float(accuracy)}


class MonitorController:
    def __init__(self, enabled: bool, output_dir: Path, fl_mode: bool, llm_mode: bool) -> None:
        self.enabled = enabled and SystemMonitor is not None
        self.output_dir = output_dir
        self.monitor = None
        if self.enabled:
            self.monitor = SystemMonitor(sampling_interval=1.0, fl_mode=fl_mode, llm_mode=llm_mode)

    def start(self) -> None:
        if self.monitor is not None:
            self.monitor.start()

    def stop(self) -> None:
        if self.monitor is not None:
            self.monitor.stop()
            self.monitor.save_report(str(self.output_dir / "client_system_report.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated IDS demo client")
    parser.add_argument("--server", required=True, help="Flower server address, e.g. 192.168.1.10:8080")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--client-id", required=True, help="Unique client identifier")
    parser.add_argument("--train-csv", default=None, help="Optional explicit training CSV path")
    parser.add_argument("--test-csv", default=None, help="Optional explicit test CSV path")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split if only one CSV is provided")
    parser.add_argument("--output-dir", default=None, help="Optional explicit output directory")
    parser.add_argument("--disable-monitor", action="store_true", help="Disable local system monitor")
    args = parser.parse_args()

    cfg = load_json(Path(args.config).resolve())
    if not bool(cfg.get("FL", True)):
        raise ValueError("This client script is for federated mode. Set FL=true in config.json.")

    data = load_client_data(
        cfg=cfg,
        client_id=args.client_id,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        test_split=args.test_split,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        base = Path(cfg.get("CLIENT_OUTPUT_DIR", "client_runs")).resolve()
        output_dir = base / f"client_{args.client_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = DemoClient(cfg=cfg, client_id=args.client_id, data=data, output_dir=output_dir)
    monitor = MonitorController(
        enabled=not args.disable_monitor,
        output_dir=output_dir,
        fl_mode=True,
        llm_mode=bool(cfg.get("LLM", False)),
    )

    def cleanup() -> None:
        monitor.stop()

    atexit.register(cleanup)

    print(f"[Client {args.client_id}] Device: {DEVICE}")
    print(f"[Client {args.client_id}] Mode: {cfg.get('MODE', 'DNN')}")
    if bool(cfg.get("USE_NOTEBOOK_DATA_STAGE", True)):
        print(f"[Client {args.client_id}] Using notebook-compatible data stage")
    print(f"[Client {args.client_id}] Train examples: {data.train_size}")
    print(f"[Client {args.client_id}] Test examples: {data.test_size}")
    print(f"[Client {args.client_id}] Feature count: {data.feature_count}")
    print(f"[Client {args.client_id}] Output classes: {data.output_size}")
    print(f"[Client {args.client_id}] Connecting to {args.server}")

    monitor.start()
    fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()
