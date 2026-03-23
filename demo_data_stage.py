
#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
import torch


class ClassifierDataset(Dataset):
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        self.x_data = torch.from_numpy(np.asarray(x_data, dtype=np.float32)).float()
        self.y_data = torch.from_numpy(np.asarray(y_data, dtype=np.int64)).long()

    def __getitem__(self, index: int):
        return self.x_data[index], self.y_data[index]

    def __len__(self) -> int:
        return len(self.x_data)


@dataclass
class PreparedNotebookData:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    round_x: Dict[int, np.ndarray]
    round_y: Dict[int, np.ndarray]
    feature_count: int
    output_size: int
    size_round: int


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, quoting=csv.QUOTE_NONE, on_bad_lines="skip")


def _resolve_raw_dataset(cfg: Dict[str, Any], data_num: str) -> Path:
    data_dir = Path(cfg.get("DATA_DIR", ".")).resolve()
    num_atcks = cfg.get("NUM_ATCKS")
    if num_atcks is None:
        raise ValueError("NUM_ATCKS is required to use the notebook-compatible data stage.")
    path = data_dir / f"2_Dataset_{num_atcks}_Attack_{data_num}_normal.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return path


def prepare_notebook_data(cfg: Dict[str, Any]) -> PreparedNotebookData:
    seed = int(cfg.get("SEED", 42))
    rounds = int(cfg["ROUNDS"])
    batch_size = int(cfg["BATCH_SIZE"])
    num_clients = int(cfg["NUM_CLIENTS"])
    data_groups = int(cfg["DATA_GROUPS"])
    batch_round = int(cfg["BATCH_ROUND"])
    size_round = batch_round * batch_size * num_clients

    datasets: Dict[str, pd.DataFrame] = {}
    for name in ["30", "100", "70", "50", "120"]:
        df = _read_csv(_resolve_raw_dataset(cfg, name))
        datasets[name] = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    splits = {name: np.array_split(datasets[name], data_groups) for name in ["30", "100", "70", "50"]}

    combined_parts: List[pd.DataFrame] = []
    for group in range(data_groups):
        combined_parts.extend(
            [splits["30"][group], splits["100"][group], splits["70"][group], splits["50"][group]]
        )
    combined = pd.concat(combined_parts, ignore_index=True)

    train_x_df = combined.iloc[:, :-1]
    train_y = combined.iloc[:, -1].to_numpy(dtype=np.int64)
    test_x_df = datasets["120"].iloc[:, :-1]
    test_y = datasets["120"].iloc[:, -1].to_numpy(dtype=np.int64)

    scaler_name = str(cfg.get("SCALER", "auto")).lower()
    if scaler_name == "auto":
        scaler = StandardScaler() if str(cfg.get("MODE", "DNN")).upper() == "TT" else MinMaxScaler()
    elif scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported SCALER={cfg.get('SCALER')}")

    train_x = scaler.fit_transform(train_x_df)
    test_x = scaler.transform(test_x_df)

    feature_count = int(train_x.shape[1])
    if "OUTPUT_SIZE" in cfg and int(cfg["OUTPUT_SIZE"]) > 0:
        output_size = int(cfg["OUTPUT_SIZE"])
    else:
        output_size = int(max(np.max(train_y), np.max(test_y))) + 1

    round_x: Dict[int, np.ndarray] = {}
    round_y: Dict[int, np.ndarray] = {}
    size_demo = size_round
    for round_idx in range(1, rounds + 1):
        start = 0 if round_idx == 1 else (size_demo - size_round)
        end = min(size_demo, len(train_x))
        round_x[round_idx] = np.asarray(train_x[start:end], dtype=np.float32)
        round_y[round_idx] = np.asarray(train_y[start:end], dtype=np.int64)
        size_demo += size_round

    return PreparedNotebookData(
        train_x=np.asarray(train_x, dtype=np.float32),
        train_y=np.asarray(train_y, dtype=np.int64),
        test_x=np.asarray(test_x, dtype=np.float32),
        test_y=np.asarray(test_y, dtype=np.int64),
        round_x=round_x,
        round_y=round_y,
        feature_count=feature_count,
        output_size=output_size,
        size_round=size_round,
    )


def build_eval_loader(prepared: PreparedNotebookData, batch_size: int) -> DataLoader:
    dataset = ClassifierDataset(prepared.test_x, prepared.test_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def build_round_dataset(prepared: PreparedNotebookData, round_idx: int) -> ClassifierDataset:
    if round_idx not in prepared.round_x:
        raise KeyError(f"Round {round_idx} not present in prepared data")
    return ClassifierDataset(prepared.round_x[round_idx], prepared.round_y[round_idx])


def _balanced_partition_indices(dataset_len: int, num_clients: int) -> List[List[int]]:
    if dataset_len == 0:
        return [[] for _ in range(num_clients)]
    base_portion = dataset_len // num_clients
    remainder = dataset_len % num_clients
    indices: List[List[int]] = []
    start_idx = 0
    for i in range(num_clients):
        size = base_portion + (1 if i < remainder else 0)
        end_idx = start_idx + size
        indices.append(list(range(start_idx, end_idx)))
        start_idx = end_idx
    return indices


def build_client_round_loaders(
    prepared: PreparedNotebookData,
    round_idx: int,
    num_clients: int,
    batch_size: int,
) -> List[DataLoader]:
    round_dataset = build_round_dataset(prepared, round_idx)
    portion_indices = _balanced_partition_indices(len(round_dataset), num_clients)
    loaders: List[DataLoader] = []
    for idxs in portion_indices:
        subset = Subset(round_dataset, idxs)
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=False))
    return loaders


def client_id_to_index(client_id: str, num_clients: int) -> int:
    cleaned = str(client_id).strip()
    if cleaned.isdigit():
        idx = int(cleaned)
        if 1 <= idx <= num_clients:
            return idx - 1
        if 0 <= idx < num_clients:
            return idx
    raise ValueError(
        f"client_id={client_id!r} could not be mapped to a client index in [1..{num_clients}] or [0..{num_clients-1}]"
    )
