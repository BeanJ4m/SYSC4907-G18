import time
import os
import json
import psutil
import torch


class BenchmarkLogger:
    def __init__(self, output_path, mode, model_type):
        self.output_path = output_path
        self.mode = mode              # "centralized" or "federated"
        self.model_type = model_type  # "DNN" or "TT"
        self.process = psutil.Process(os.getpid())

        self.records = []             # per-round logs
        self.experiment = {           # experiment-level metadata
            "mode": mode,
            "model": model_type,
            "total_time_sec": None,
            "llm_overhead_sec": None,
            "llm_used": False,
        }

        os.makedirs(output_path, exist_ok=True)

    # --------------------------------------------------
    # ROUND-LEVEL LOGGING
    # --------------------------------------------------
    def start_round(self, round_id):
        self.round_id = round_id
        self.round_start_time = time.time()

        self.cpu_time_start = self.process.cpu_times().user
        self.mem_peak = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def log_memory(self):
        mem = self.process.memory_info().rss / (1024 ** 2)
        self.mem_peak = max(self.mem_peak, mem)

    def end_round(
        self,
        train_time,
        eval_time,
        accuracy,
        loss,
        samples_this_round,
        learning_rate,
        epochs,
    ):
        round_time = time.time() - self.round_start_time
        cpu_time_end = self.process.cpu_times().user

        record = {
            "round": self.round_id,
            "mode": self.mode,
            "model": self.model_type,
            "train_time_sec": round(train_time, 4),
            "eval_time_sec": round(eval_time, 4),
            "round_time_sec": round(round_time, 4),
            "accuracy": round(accuracy, 6),
            "loss": round(loss, 6),
            "cpu_time_sec": round(cpu_time_end - self.cpu_time_start, 4),
            "ram_peak_mb": round(self.mem_peak, 2),
            "samples_this_round": samples_this_round,
            "samples_per_sec": round(
                samples_this_round / train_time, 2
            ) if train_time > 0 else 0.0,
            "learning_rate": learning_rate,
            "epochs": epochs,
        }

        if torch.cuda.is_available():
            record["gpu_mem_peak_mb"] = round(
                torch.cuda.max_memory_allocated() / (1024 ** 2), 2
            )

        self.records.append(record)

        with open(os.path.join(self.output_path, "benchmark.json"), "w") as f:
            json.dump(self.records, f, indent=2)

        return record

    # --------------------------------------------------
    # EXPERIMENT-LEVEL LOGGING
    # --------------------------------------------------
    def log_experiment_time(self, total_time_sec):
        self.experiment["total_time_sec"] = round(total_time_sec, 4)
        self._write_summary()

    def log_llm_overhead(self, overhead_time_sec):
        self.experiment["llm_overhead_sec"] = round(overhead_time_sec, 4)
        self.experiment["llm_used"] = True
        self._write_summary()

    def _write_summary(self):
        with open(
            os.path.join(self.output_path, "benchmark_summary.json"), "w"
        ) as f:
            json.dump(self.experiment, f, indent=2)
