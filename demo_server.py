#!/usr/bin/env python3
"""
Federated training server for the IoT IDS demo.

Features
- Flower FedAvg server
- DNN / TabTransformer model selection from config.json
- Optional LLM mid-training update through llm.py
- System monitoring through system_monitor.py
- Benchmark logging through benchmark.py
- Server-side evaluation and checkpoint saving

Expected client behavior
- Flower NumPyClient compatible
- Uses server config keys: current_round, local_epochs, learning_rate, batch_size, mode
- Returns num_examples and may optionally return metrics such as train_accuracy / train_loss
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from benchmark import BenchmarkLogger
from demo_data_stage import prepare_notebook_data, build_eval_loader
from llm import llm_mid_training_update
from system_monitor import SystemMonitor


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


@dataclass
class RuntimeState:
    cfg: Dict[str, Any]
    eval_loader: DataLoader
    feature_count: int
    output_dir: Path
    config_path: Path
    model_dir: Path
    benchmark: BenchmarkLogger
    monitor: SystemMonitor
    global_model: nn.Module
    current_round_started_at: float = 0.0
    total_started_at: float = 0.0
    llm_overhead_total: float = 0.0
    round_accuracy: List[float] = field(default_factory=list)
    round_loss: List[float] = field(default_factory=list)
    mean_client_accuracy: List[float] = field(default_factory=list)
    std_client_accuracy: List[float] = field(default_factory=list)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
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


def load_eval_data(cfg: Dict[str, Any]) -> Tuple[DataLoader, int, int]:
    use_notebook_stage = bool(cfg.get("USE_NOTEBOOK_DATA_STAGE", True))
    if use_notebook_stage:
        prepared = prepare_notebook_data(cfg)
        loader = build_eval_loader(prepared, batch_size=int(cfg.get("BATCH_SIZE", 64)))
        return loader, prepared.feature_count, prepared.output_size

    raise ValueError(
        "USE_NOTEBOOK_DATA_STAGE=false is not implemented in this version. "
        "Enable the notebook-compatible data stage in config.json."
    )


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
            loss_sum += criterion(outputs, labels).item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total


def build_results_snapshot(state: RuntimeState) -> Dict[str, List[float]]:
    return {
        "global_accuracy": list(state.round_accuracy),
        "global_loss": list(state.round_loss),
        "mean_client_accuracy": list(state.mean_client_accuracy),
        "std_client_accuracy": list(state.std_client_accuracy),
    }


def persist_round_artifacts(state: RuntimeState, round_idx: int, model: nn.Module, loss: float, accuracy: float) -> None:
    state.model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = state.model_dir / f"GlobalModel_{round_idx}.pth"
    torch.save(model.state_dict(), ckpt_path)

    with (state.output_dir / f"Global_{round_idx}_loss").open("wb") as f:
        pickle.dump([loss], f)
    with (state.output_dir / f"Global_{round_idx}_accuracy").open("wb") as f:
        pickle.dump([accuracy], f)

    print(f"[Server] Saved checkpoint: {ckpt_path}")


class DemoFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, state: RuntimeState, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state = state

    def configure_fit(self, server_round, parameters, client_manager):
        self.state.current_round_started_at = time.time()
        self.state.benchmark.start_round(server_round)
        print(f"\n{'=' * 70}\n[Server] Starting round {server_round}\n{'=' * 70}")
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            print(f"[Server] No aggregated parameters returned for round {server_round}")
            return aggregated

        aggregated_parameters, aggregated_metrics = aggregated
        if aggregated_parameters is None:
            print(f"[Server] Aggregated parameters are None for round {server_round}")
            return aggregated

        ndarray_params = fl.common.parameters_to_ndarrays(aggregated_parameters)
        set_parameters(self.state.global_model, ndarray_params)
        self.state.global_model.to(DEVICE)

        eval_loss, eval_acc = evaluate_model(self.state.global_model, self.state.eval_loader)
        self.state.round_loss.append(float(eval_loss))
        self.state.round_accuracy.append(float(eval_acc))

        client_accs: List[float] = []
        for _, fit_res in results:
            metrics = fit_res.metrics or {}
            for key in ("train_accuracy", "accuracy", "local_accuracy"):
                if key in metrics:
                    try:
                        client_accs.append(float(metrics[key]))
                        break
                    except Exception:
                        pass

        if client_accs:
            self.state.mean_client_accuracy.append(float(np.mean(client_accs)))
            self.state.std_client_accuracy.append(float(np.std(client_accs)))
        else:
            self.state.mean_client_accuracy.append(float(eval_acc))
            self.state.std_client_accuracy.append(0.0)

        persist_round_artifacts(self.state, server_round, self.state.global_model, eval_loss, eval_acc)

        round_train_time = max(time.time() - self.state.current_round_started_at, 0.0)
        samples_this_round = int(sum(fit_res.num_examples for _, fit_res in results))
        epochs = int(self.state.cfg.get("EPOCHS", 1))
        learning_rate = float(self.state.cfg.get("LEARNING_RATE", 1e-3))
        self.state.benchmark.log_memory()
        self.state.benchmark.end_round(
            train_time=round_train_time,
            eval_time=0.0,
            accuracy=float(eval_acc),
            loss=float(eval_loss),
            samples_this_round=samples_this_round,
            learning_rate=learning_rate,
            epochs=epochs,
        )

        print(
            f"[Server] Round {server_round} complete | "
            f"loss={eval_loss:.6f} acc={eval_acc:.4f} samples={samples_this_round}"
        )

        if bool(self.state.cfg.get("LLM", False)) and server_round == int(self.state.cfg.get("LLM_TRIGGER_ROUND", -1)):
            self._run_llm_update(server_round)

        return aggregated_parameters, aggregated_metrics

    def _run_llm_update(self, round_idx: int) -> None:
        model_path = self.state.model_dir / f"GlobalModel_{round_idx}.pth"
        results_snapshot = build_results_snapshot(self.state)

        print(f"[Server] Triggering LLM update at round {round_idx}")
        llm_started_at = time.time()
        try:
            new_cfg = llm_mid_training_update(
                model_path=str(model_path),
                config_path=str(self.state.config_path),
                results_snapshot=results_snapshot,
                round_idx=round_idx,
                trigger_round=int(self.state.cfg["LLM_TRIGGER_ROUND"]),
                allow_architecture_change=bool(self.state.cfg.get("ALLOW_ARCHITECTURE_CHANGE", False)),
            )
        finally:
            llm_elapsed = time.time() - llm_started_at
            self.state.llm_overhead_total += llm_elapsed
            self.state.benchmark.log_llm_overhead(self.state.llm_overhead_total)

        if not new_cfg:
            print("[Server] LLM returned no safe update; keeping current config")
            return

        tracked_keys = [
            "LEARNING_RATE",
            "EPOCHS",
            "BATCH_SIZE",
            "HIDDEN1_SIZE",
            "HIDDEN2_SIZE",
            "DROPOUT_RATE",
            "EMB_DIM",
            "MLP_DIM",
        ]
        changed = False
        for key in tracked_keys:
            if key in new_cfg and self.state.cfg.get(key) != new_cfg.get(key):
                print(f"[Server] Config update: {key}: {self.state.cfg.get(key)} -> {new_cfg.get(key)}")
                self.state.cfg[key] = new_cfg[key]
                changed = True

        if changed:
            save_json(self.state.config_path, self.state.cfg)
            print(f"[Server] Updated config written to {self.state.config_path}")
        else:
            print("[Server] LLM produced no effective runtime changes")



def make_fit_config_fn(state: RuntimeState):
    def fit_config(server_round: int) -> Dict[str, Any]:
        return {
            "current_round": server_round,
            "local_epochs": int(state.cfg.get("EPOCHS", 1)),
            "learning_rate": float(state.cfg.get("LEARNING_RATE", 1e-3)),
            "batch_size": int(state.cfg.get("BATCH_SIZE", 64)),
            "mode": str(state.cfg.get("MODE", "DNN")).upper(),
        }

    return fit_config



def initialise_global_model(state: RuntimeState, output_size: int) -> None:
    init_ckpt = state.cfg.get("INITIAL_CHECKPOINT")
    if init_ckpt:
        ckpt_path = Path(init_ckpt)
    else:
        ckpt_path = state.model_dir / f"0_Input_Random_Model_{str(state.cfg['MODE']).upper()}.pth"

    if ckpt_path.exists():
        print(f"[Server] Loading initial checkpoint: {ckpt_path}")
        loaded = torch.load(ckpt_path, map_location=DEVICE)
        state.global_model.load_state_dict(loaded)
    else:
        print(f"[Server] Initial checkpoint not found, creating: {ckpt_path}")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state.global_model.state_dict(), ckpt_path)

    state.global_model.to(DEVICE)



def build_runtime(config_path: Path, output_dir: Optional[Path]) -> RuntimeState:
    cfg = load_json(config_path)
    mode = str(cfg.get("MODE", "DNN")).upper()
    fl_enabled = bool(cfg.get("FL", True))
    if not fl_enabled:
        raise ValueError("This server script is for federated mode. Set FL=true in config.json.")

    eval_loader, feature_count, output_size = load_eval_data(cfg)
    cfg["MODE"] = mode
    cfg["OUTPUT_SIZE"] = output_size
    cfg["INPUT_SIZE"] = int(cfg.get("INPUT_SIZE", feature_count))
    if cfg["INPUT_SIZE"] != feature_count:
        raise ValueError(
            f"INPUT_SIZE mismatch: config has {cfg['INPUT_SIZE']}, evaluation data has {feature_count} features"
        )

    if output_dir is None:
        path_template = cfg.get(
            "PATH_TEMPLATE",
            "results/{MODE}-FL-{FL}-{NUM_CLIENTS}-clients-{NUM_ATCKS}-atk-{ROUNDS}-rounds-{EPOCHS}-epochs-{LEARNING_RATE}-lr-{DATA_GROUPS}-groups-llm-{LLM}",
        )
        safe_cfg = copy.deepcopy(cfg)
        output_dir = Path(path_template.format(**safe_cfg))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"

    model = build_model(cfg, feature_count=feature_count, output_size=output_size)
    monitor = SystemMonitor(
        sampling_interval=float(cfg.get("MONITOR_INTERVAL_SEC", 1.0)),
        fl_mode=True,
        llm_mode=bool(cfg.get("LLM", False)),
    )
    benchmark = BenchmarkLogger(
        output_path=str(output_dir),
        mode="federated",
        model_type=mode,
    )

    state = RuntimeState(
        cfg=cfg,
        eval_loader=eval_loader,
        feature_count=feature_count,
        output_dir=output_dir,
        config_path=config_path,
        model_dir=model_dir,
        benchmark=benchmark,
        monitor=monitor,
        global_model=model,
    )
    initialise_global_model(state, output_size)
    return state



def main() -> None:
    parser = argparse.ArgumentParser(description="Federated IDS demo server")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--host", default="0.0.0.0", help="Flower bind host")
    parser.add_argument("--port", type=int, default=8080, help="Flower bind port")
    parser.add_argument("--output-dir", default=None, help="Optional explicit output directory")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    state = build_runtime(config_path=config_path, output_dir=output_dir)

    strategy = DemoFedAvg(
        state=state,
        fraction_fit=float(state.cfg.get("FRACTION_FIT", 1.0)),
        fraction_evaluate=float(state.cfg.get("FRACTION_EVALUATE", 0.0)),
        min_fit_clients=int(state.cfg.get("MIN_FIT_CLIENTS", 2)),
        min_evaluate_clients=int(state.cfg.get("MIN_EVALUATE_CLIENTS", 0)),
        min_available_clients=int(state.cfg.get("MIN_AVAILABLE_CLIENTS", 2)),
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(state.global_model)),
        on_fit_config_fn=make_fit_config_fn(state),
    )

    server_address = f"{args.host}:{args.port}"
    print(f"[Server] Device: {DEVICE}")
    print(f"[Server] Mode: {state.cfg['MODE']}")
    print(f"[Server] FL: {state.cfg['FL']}")
    print(f"[Server] LLM: {state.cfg.get('LLM', False)}")
    print(f"[Server] Output dir: {state.output_dir}")
    print(f"[Server] Starting Flower server on {server_address}")

    state.total_started_at = time.time()
    state.monitor.start()
    try:
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=int(state.cfg.get("ROUNDS", 1))),
            strategy=strategy,
        )
    finally:
        total_time = time.time() - state.total_started_at
        state.benchmark.log_experiment_time(total_time)
        state.monitor.stop()
        state.monitor.save_report(str(state.output_dir / "system_report.json"))
        save_json(state.output_dir / "final_runtime_config.json", state.cfg)
        print(f"[Server] Finished. Total runtime: {total_time:.2f}s")


if __name__ == "__main__":
    main()
