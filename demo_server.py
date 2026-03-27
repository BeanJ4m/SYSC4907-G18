#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from demo_data_stage import build_eval_loader, prepare_notebook_data

try:
    from system_monitor import SystemMonitor
except Exception:
    SystemMonitor = None  # type: ignore


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODELS
# ============================================================

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


def build_model(cfg: Dict[str, Any], feature_count: int, output_size: int) -> nn.Module:
    mode = str(cfg["MODE"]).upper()
    if mode == "DNN":
        return DNNNet(
            input_size=feature_count,
            hidden1_size=int(cfg.get("HIDDEN1_SIZE", 64)),
            hidden2_size=int(cfg.get("HIDDEN2_SIZE", 32)),
            output_size=output_size,
            dropout_rate=float(cfg.get("DROUPOUT_RATE", 0.0)),
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
            dropout=float(cfg.get("DROUPOUT_RATE", 0.1)),
        )
    raise ValueError(f"Unsupported MODE={cfg['MODE']}")


# ============================================================
# PARAMETER HELPERS
# ============================================================

def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: Sequence[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# ============================================================
# METRICS / EVAL
# ============================================================

def evaluate_model(model: nn.Module, loader: DataLoader) -> Tuple[float, float, float, float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.to(DEVICE)

    total = 0
    loss_sum = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(features)
            loss_sum += criterion(outputs, labels).item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    if total == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    avg_loss = loss_sum / total
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    recall = float(recall_score(all_labels, all_preds, average="weighted", zero_division=0))
    precision = float(precision_score(all_labels, all_preds, average="weighted", zero_division=0))

    return avg_loss, acc, f1, recall, precision


def metric_stats(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    return float(min(values)), float(max(values)), float(sum(values) / len(values))


def print_final_metric_summary(state: "RuntimeState") -> None:
    acc_min, acc_max, acc_avg = metric_stats(state.round_accuracy)
    f1_min, f1_max, f1_avg = metric_stats(state.round_f1)
    recall_min, recall_max, recall_avg = metric_stats(state.round_recall)
    precision_min, precision_max, precision_avg = metric_stats(state.round_precision)

    print("\n" + "=" * 72)
    print("[Server] FINAL METRICS SUMMARY")
    print("=" * 72)
    print(f"Accuracy   | min={acc_min:.4f} max={acc_max:.4f} avg={acc_avg:.4f}")
    print(f"F1 Score   | min={f1_min:.4f} max={f1_max:.4f} avg={f1_avg:.4f}")
    print(f"Recall     | min={recall_min:.4f} max={recall_max:.4f} avg={recall_avg:.4f}")
    print(f"Precision  | min={precision_min:.4f} max={precision_max:.4f} avg={precision_avg:.4f}")


# ============================================================
# STATE / PATHS
# ============================================================

@dataclass
class RuntimeState:
    cfg: Dict[str, Any]
    prepared: Any
    output_dir: Path
    model_dir: Path
    server_dir: Path
    state_dir: Path
    eval_loader: DataLoader
    global_model: nn.Module
    round_accuracy: List[float] = field(default_factory=list)
    round_loss: List[float] = field(default_factory=list)
    round_f1: List[float] = field(default_factory=list)
    round_recall: List[float] = field(default_factory=list)
    round_precision: List[float] = field(default_factory=list)
    mean_client_accuracy: List[float] = field(default_factory=list)
    std_client_accuracy: List[float] = field(default_factory=list)
    monitor: Optional[Any] = None


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_output_dir(cfg: Dict[str, Any]) -> Path:
    safe_cfg = dict(cfg)
    safe_cfg.setdefault("EPOCHS", safe_cfg.get("LOCAL_EPOCHS", 1))
    return Path(cfg["PATH_TEMPLATE"].format(**safe_cfg)).resolve()


def load_config(config_path: Path) -> Dict[str, Any]:
    cfg = load_json(config_path)

    cfg.setdefault("FL", True)
    cfg.setdefault("MODE", "DNN")
    cfg.setdefault("EPOCHS", cfg.get("LOCAL_EPOCHS", 1))
    cfg.setdefault("NUM_CLIENTS", 1)
    cfg.setdefault("ROUNDS", 10)
    cfg.setdefault("BATCH_SIZE", 64)
    cfg.setdefault("LEARNING_RATE", 0.0025)
    cfg.setdefault("DATA_GROUPS", 240)
    cfg.setdefault("BATCH_ROUND", 6)
    cfg.setdefault("NUM_ATCKS", 14)
    cfg.setdefault("INPUT_SIZE", 98)
    cfg.setdefault("OUTPUT_SIZE", int(cfg["NUM_ATCKS"]) + 1)
    cfg.setdefault(
        "PATH_TEMPLATE",
        "results/{MODE}-FL-{FL}-{NUM_CLIENTS}-clients-{NUM_ATCKS}-atk-{ROUNDS}-rounds-{EPOCHS}-epochs-{LEARNING_RATE}-lr-{DATA_GROUPS}-groups-llm-{LLM}",
    )
    cfg.setdefault("LLM", False)
    cfg.setdefault("LLM_TRIGGER_ROUND", cfg["ROUNDS"] + 1)
    cfg.setdefault("ALLOW_ARCHITECTURE_CHANGE", False)

    cfg.setdefault("MIN_FIT_CLIENTS", cfg["NUM_CLIENTS"])
    cfg.setdefault("MIN_AVAILABLE_CLIENTS", cfg["NUM_CLIENTS"])
    cfg.setdefault("MIN_EVALUATE_CLIENTS", 0)
    cfg.setdefault("FRACTION_FIT", 1.0)
    cfg.setdefault("FRACTION_EVALUATE", 0.0)

    return cfg


def persist_round_artifacts(
    state: RuntimeState,
    round_idx: int,
    model: nn.Module,
    loss: float,
    accuracy: float,
    f1: float,
    recall: float,
    precision: float,
) -> None:
    model_path = state.model_dir / f"GlobalModel_{round_idx}.pth"
    torch.save(model.state_dict(), model_path)

    with (state.output_dir / f"Global_{round_idx}_loss").open("wb") as f:
        pickle.dump([loss], f)
    with (state.output_dir / f"Global_{round_idx}_accuracy").open("wb") as f:
        pickle.dump([accuracy], f)
    with (state.output_dir / f"Global_{round_idx}_f1").open("wb") as f:
        pickle.dump([f1], f)
    with (state.output_dir / f"Global_{round_idx}_recall").open("wb") as f:
        pickle.dump([recall], f)
    with (state.output_dir / f"Global_{round_idx}_precision").open("wb") as f:
        pickle.dump([precision], f)

    latest = {
        "round": round_idx,
        "loss": float(loss),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "model_path": str(model_path),
    }
    save_json(state.state_dir / "latest_round.json", latest)


def build_results_snapshot(state: RuntimeState) -> Dict[str, List[float]]:
    return {
        "global_accuracy": list(state.round_accuracy),
        "global_loss": list(state.round_loss),
        "global_f1": list(state.round_f1),
        "global_recall": list(state.round_recall),
        "global_precision": list(state.round_precision),
        "mean_client_accuracy": list(state.mean_client_accuracy),
        "std_client_accuracy": list(state.std_client_accuracy),
    }


# ============================================================
# FLOWER STRATEGY
# ============================================================

class DemoFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, state: RuntimeState, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.state = state

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return None

        parameters_aggregated, metrics_aggregated = aggregated
        aggregated_ndarrays = fl.common.parameters_to_ndarrays(parameters_aggregated)
        set_parameters(self.state.global_model, aggregated_ndarrays)

        client_accs: List[float] = []
        samples_this_round = 0
        for _, fit_res in results:
            samples_this_round += int(fit_res.num_examples)
            if "train_accuracy" in fit_res.metrics:
                client_accs.append(float(fit_res.metrics["train_accuracy"]))
            elif "local_accuracy" in fit_res.metrics:
                client_accs.append(float(fit_res.metrics["local_accuracy"]))

        if client_accs:
            self.state.mean_client_accuracy.append(float(np.mean(client_accs)))
            self.state.std_client_accuracy.append(float(np.std(client_accs)))
        else:
            self.state.mean_client_accuracy.append(0.0)
            self.state.std_client_accuracy.append(0.0)

        eval_loss, eval_acc, eval_f1, eval_recall, eval_precision = evaluate_model(
            self.state.global_model, self.state.eval_loader
        )

        self.state.round_loss.append(float(eval_loss))
        self.state.round_accuracy.append(float(eval_acc))
        self.state.round_f1.append(float(eval_f1))
        self.state.round_recall.append(float(eval_recall))
        self.state.round_precision.append(float(eval_precision))

        persist_round_artifacts(
            self.state,
            server_round,
            self.state.global_model,
            eval_loss,
            eval_acc,
            eval_f1,
            eval_recall,
            eval_precision,
        )

        print(
            f"[Server] Round {server_round} complete | "
            f"acc={eval_acc:.4f} | f1={eval_f1:.4f} | "
            f"recall={eval_recall:.4f} | precision={eval_precision:.4f} | "
            f"loss={eval_loss:.6f} | samples={samples_this_round}"
        )

        if bool(self.state.cfg.get("LLM", False)) and int(server_round) == int(self.state.cfg.get("LLM_TRIGGER_ROUND", -1)):
            try:
                from llm import llm_mid_training_update

                config_path = Path(self.state.cfg.get("CONFIG_PATH", "config.json")).resolve()
                model_path = self.state.model_dir / f"GlobalModel_{server_round}.pth"

                new_cfg = llm_mid_training_update(
                    model_path=str(model_path),
                    config_path=str(config_path),
                    results_snapshot=build_results_snapshot(self.state),
                    round_idx=server_round,
                    trigger_round=int(self.state.cfg["LLM_TRIGGER_ROUND"]),
                    allow_architecture_change=bool(self.state.cfg.get("ALLOW_ARCHITECTURE_CHANGE", False)),
                )

                if isinstance(new_cfg, dict) and new_cfg:
                    old_cfg = copy.deepcopy(self.state.cfg)
                    self.state.cfg.update(new_cfg)
                    save_json(
                        self.state.state_dir / f"llm_update_round_{server_round}.json",
                        {
                            "round": server_round,
                            "old_cfg": old_cfg,
                            "new_cfg": self.state.cfg,
                        },
                    )
                    print(f"[Server] LLM update applied at round {server_round}")
            except Exception as e:
                print(f"[Server] LLM update failed at round {server_round}: {e}")

        return parameters_aggregated, metrics_aggregated


# ============================================================
# MAIN
# ============================================================

def build_runtime(config_path: Path) -> RuntimeState:
    cfg = load_config(config_path)
    cfg["CONFIG_PATH"] = str(config_path.resolve())

    output_dir = resolve_output_dir(cfg)
    model_dir = output_dir / "models"
    server_dir = output_dir / "server"
    state_dir = output_dir / "state"

    model_dir.mkdir(parents=True, exist_ok=True)
    server_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_notebook_data(cfg)
    eval_loader = build_eval_loader(prepared, batch_size=int(cfg.get("BATCH_SIZE", 64)))

    feature_count = int(prepared.feature_count)
    output_size = int(prepared.output_size)

    global_model = build_model(cfg, feature_count=feature_count, output_size=output_size).to(DEVICE)

    save_json(server_dir / "resolved_config.json", cfg)

    if getattr(prepared, "scaler", None) is not None:
        with (state_dir / "scaler.pkl").open("wb") as f:
            pickle.dump(prepared.scaler, f)

    monitor = None
    if SystemMonitor is not None:
        try:
            monitor = SystemMonitor(
                sampling_interval=1.0,
                fl_mode=True,
                llm_mode=bool(cfg.get("LLM", False)),
            )
        except Exception:
            monitor = None

    return RuntimeState(
        cfg=cfg,
        prepared=prepared,
        output_dir=output_dir,
        model_dir=model_dir,
        server_dir=server_dir,
        state_dir=state_dir,
        eval_loader=eval_loader,
        global_model=global_model,
        monitor=monitor,
    )


def fit_config_fn_factory(state: RuntimeState):
    def fit_config_fn(server_round: int) -> Dict[str, Any]:
        return {
            "current_round": int(server_round),
            "local_epochs": int(state.cfg.get("EPOCHS", 1)),
            "epochs": int(state.cfg.get("EPOCHS", 1)),
            "learning_rate": float(state.cfg.get("LEARNING_RATE", 1e-3)),
            "batch_size": int(state.cfg.get("BATCH_SIZE", 64)),
            "mode": str(state.cfg.get("MODE", "DNN")).upper(),
        }
    return fit_config_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated IDS demo server")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=4269, help="Bind port")
    args = parser.parse_args()

    started = time.time()
    state = build_runtime(Path(args.config).resolve())

    print(f"[Server] Artifact root: {state.output_dir}")
    print(f"[Server] Feature count: {state.prepared.feature_count}")
    print(f"[Server] Output size: {state.prepared.output_size}")
    print("=" * 72)
    print(f"[Server] Starting FL for {state.cfg['ROUNDS']} rounds on {args.host}:{args.port}")
    print("=" * 72)

    if state.monitor is not None:
        try:
            state.monitor.start()
        except Exception:
            state.monitor = None

    strategy = DemoFedAvg(
        state=state,
        fraction_fit=float(state.cfg.get("FRACTION_FIT", 1.0)),
        fraction_evaluate=float(state.cfg.get("FRACTION_EVALUATE", 0.0)),
        min_fit_clients=int(state.cfg.get("MIN_FIT_CLIENTS", state.cfg["NUM_CLIENTS"])),
        min_evaluate_clients=int(state.cfg.get("MIN_EVALUATE_CLIENTS", 0)),
        min_available_clients=int(state.cfg.get("MIN_AVAILABLE_CLIENTS", state.cfg["NUM_CLIENTS"])),
        on_fit_config_fn=fit_config_fn_factory(state),
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(state.global_model)),
    )

    try:
        fl.server.start_server(
            server_address=f"{args.host}:{args.port}",
            config=fl.server.ServerConfig(num_rounds=int(state.cfg["ROUNDS"])),
            strategy=strategy,
        )
    finally:
        total_time = time.time() - started

        if state.monitor is not None:
            try:
                state.monitor.stop()
                state.monitor.save_report(str(state.output_dir / "system_report.json"))
            except Exception:
                pass

        print_final_metric_summary(state)
        save_json(state.output_dir / "final_runtime_config.json", state.cfg)
        print(f"[Server] Finished. Total runtime: {total_time:.2f}s")


if __name__ == "__main__":
    main()