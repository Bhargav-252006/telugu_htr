"""
src/training_logger.py

Comprehensive training logger that captures:
  - GPU usage (memory, utilization, temperature)
  - Training performance (samples/sec, epoch time)
  - System info (GPU model, CUDA version, Python version)
  - Per-epoch metrics (loss, CER, WER)
  - Full training summary

All logs are saved as both:
  - Human-readable .log file (for quick inspection)
  - JSON file (for programmatic analysis and paper)
"""

from __future__ import annotations
import json
import os
import platform
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import torch


class TrainingLogger:
    """
    Comprehensive logger for training runs.

    Usage:
        logger = TrainingLogger(log_dir="logs/ar", model_name="AR Transformer")
        logger.log_system_info(model)         # call once at start
        logger.log_epoch_start(epoch)          # call at epoch start
        logger.log_gpu_stats(epoch, step)      # call periodically
        logger.log_epoch_end(epoch, metrics)   # call at epoch end
        logger.log_training_complete(metrics)  # call at end
    """

    def __init__(self, log_dir: str, model_name: str = "model"):
        self.log_dir = log_dir
        self.model_name = model_name
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        self.json_file = os.path.join(log_dir, f"training_{timestamp}.json")

        self.training_start_time = time.time()
        self.epoch_start_time = None
        self.epoch_times = []
        self.epoch_metrics = []
        self.gpu_snapshots = []
        self.system_info = {}

        self._write_log(f"{'='*70}")
        self._write_log(f"  Training Log — {model_name}")
        self._write_log(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_log(f"{'='*70}\n")

    def _write_log(self, msg: str):
        """Append message to log file and print."""
        print(msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _save_json(self):
        """Save all collected data to JSON."""
        data = {
            "model_name": self.model_name,
            "system_info": self.system_info,
            "training_start": datetime.fromtimestamp(
                self.training_start_time
            ).isoformat(),
            "epoch_metrics": self.epoch_metrics,
            "epoch_times_seconds": self.epoch_times,
            "gpu_snapshots": self.gpu_snapshots,
        }
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    # ── System Info ──────────────────────────────────────────────

    def log_system_info(self, model=None, config: dict = None):
        """Log system info once at the start of training."""
        info = {
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = str(torch.backends.cudnn.version())
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_total_gb"] = round(props.total_mem / 1e9, 2)
            info["gpu_compute_capability"] = f"{props.major}.{props.minor}"

        if model is not None:
            info["model_parameters"] = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            info["model_parameters_formatted"] = f"{info['model_parameters']:,}"

        if config is not None:
            info["config"] = config

        self.system_info = info

        self._write_log("── System Info ─────────────────────────────────────")
        self._write_log(f"  Python     : {info['python_version']}")
        self._write_log(f"  PyTorch    : {info['pytorch_version']}")
        self._write_log(f"  Platform   : {info['platform']}")
        if info["cuda_available"]:
            self._write_log(f"  CUDA       : {info['cuda_version']}")
            self._write_log(f"  cuDNN      : {info['cudnn_version']}")
            self._write_log(f"  GPU        : {info['gpu_name']}")
            self._write_log(f"  GPU Memory : {info['gpu_memory_total_gb']} GB")
            self._write_log(f"  Compute Cap: {info['gpu_compute_capability']}")
        if model is not None:
            self._write_log(f"  Parameters : {info['model_parameters_formatted']}")
        self._write_log("")
        self._save_json()

    # ── GPU Stats ────────────────────────────────────────────────

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU memory and utilization stats."""
        if not torch.cuda.is_available():
            return {"gpu_available": False}

        stats = {
            "gpu_available": True,
            "gpu_memory_allocated_mb": round(
                torch.cuda.memory_allocated() / 1e6, 1
            ),
            "gpu_memory_reserved_mb": round(
                torch.cuda.memory_reserved() / 1e6, 1
            ),
            "gpu_memory_total_mb": round(
                torch.cuda.get_device_properties(0).total_mem / 1e6, 1
            ),
            "gpu_max_memory_allocated_mb": round(
                torch.cuda.max_memory_allocated() / 1e6, 1
            ),
        }
        total = stats["gpu_memory_total_mb"]
        used = stats["gpu_memory_allocated_mb"]
        stats["gpu_memory_utilization_pct"] = round(used / total * 100, 1)
        stats["gpu_memory_free_mb"] = round(total - used, 1)

        return stats

    def log_gpu_stats(self, epoch: int, step: int = 0):
        """Log GPU stats at a specific point in training."""
        stats = self.get_gpu_stats()
        if not stats["gpu_available"]:
            return

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "step": step,
            **stats,
        }
        self.gpu_snapshots.append(snapshot)

        self._write_log(
            f"  [GPU] Mem: {stats['gpu_memory_allocated_mb']:.0f}MB / "
            f"{stats['gpu_memory_total_mb']:.0f}MB "
            f"({stats['gpu_memory_utilization_pct']:.1f}%) | "
            f"Peak: {stats['gpu_max_memory_allocated_mb']:.0f}MB"
        )

    # ── Epoch Tracking ───────────────────────────────────────────

    def log_epoch_start(self, epoch: int):
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()
        self._write_log(f"\n{'─'*70}")
        self._write_log(
            f"  Epoch {epoch} started at "
            f"{datetime.now().strftime('%H:%M:%S')}"
        )
        self._write_log(f"{'─'*70}")

    def log_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_cer: float = None,
        val_wer: float = None,
        val_loss: float = None,
        extra_metrics: dict = None,
        samples_processed: int = 0,
    ):
        """Log metrics at the end of an epoch."""
        elapsed = time.time() - (self.epoch_start_time or self.training_start_time)
        self.epoch_times.append(elapsed)

        samples_per_sec = samples_processed / max(elapsed, 1e-6) if samples_processed else 0
        total_elapsed = time.time() - self.training_start_time
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)

        metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_cer": round(val_cer, 6) if val_cer is not None else None,
            "val_wer": round(val_wer, 6) if val_wer is not None else None,
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
            "epoch_time_seconds": round(elapsed, 1),
            "samples_per_second": round(samples_per_sec, 1),
            "total_time_seconds": round(total_elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }

        # Add GPU stats
        gpu_stats = self.get_gpu_stats()
        metrics["gpu_memory_allocated_mb"] = gpu_stats.get("gpu_memory_allocated_mb", 0)
        metrics["gpu_memory_peak_mb"] = gpu_stats.get("gpu_max_memory_allocated_mb", 0)

        if extra_metrics:
            metrics.update(extra_metrics)

        self.epoch_metrics.append(metrics)

        # Pretty print
        self._write_log(f"\n  ┌── Epoch {epoch} Summary ──────────────────────────")
        self._write_log(f"  │ Train Loss  : {train_loss:.4f}")
        if val_cer is not None:
            self._write_log(f"  │ Val CER     : {val_cer:.4f} ({val_cer*100:.2f}%)")
        if val_wer is not None:
            self._write_log(f"  │ Val WER     : {val_wer:.4f} ({val_wer*100:.2f}%)")
        self._write_log(f"  │ Epoch Time  : {elapsed:.0f}s ({timedelta(seconds=int(elapsed))})")
        self._write_log(f"  │ Speed       : {samples_per_sec:.1f} samples/sec")
        self._write_log(
            f"  │ GPU Memory  : {gpu_stats.get('gpu_memory_allocated_mb', 0):.0f}MB "
            f"(peak: {gpu_stats.get('gpu_max_memory_allocated_mb', 0):.0f}MB)"
        )
        self._write_log(f"  │ Total Time  : {timedelta(seconds=int(total_elapsed))}")

        # ETA
        remaining_epochs = extra_metrics.get("total_epochs", 50) - epoch if extra_metrics else 0
        if remaining_epochs > 0:
            eta_seconds = avg_epoch_time * remaining_epochs
            eta = timedelta(seconds=int(eta_seconds))
            self._write_log(f"  │ ETA         : {eta} ({remaining_epochs} epochs left)")

        self._write_log(f"  └─────────────────────────────────────────────────\n")
        self._save_json()

    # ── Training Complete ────────────────────────────────────────

    def log_training_complete(self, best_cer: float = None, best_epoch: int = None):
        """Log final training summary."""
        total_time = time.time() - self.training_start_time

        self._write_log(f"\n{'='*70}")
        self._write_log(f"  TRAINING COMPLETE — {self.model_name}")
        self._write_log(f"{'='*70}")
        self._write_log(f"  Total Time       : {timedelta(seconds=int(total_time))}")
        self._write_log(f"  Total Epochs     : {len(self.epoch_times)}")

        if self.epoch_times:
            self._write_log(f"  Avg Epoch Time   : {sum(self.epoch_times)/len(self.epoch_times):.0f}s")

        if best_cer is not None:
            self._write_log(f"  Best Val CER     : {best_cer:.4f} ({best_cer*100:.2f}%)")
        if best_epoch is not None:
            self._write_log(f"  Best Epoch       : {best_epoch}")

        gpu_stats = self.get_gpu_stats()
        if gpu_stats["gpu_available"]:
            self._write_log(f"  Peak GPU Memory  : {gpu_stats['gpu_max_memory_allocated_mb']:.0f}MB")

        # Find best and worst epochs
        if self.epoch_metrics:
            cers = [(m["epoch"], m["val_cer"]) for m in self.epoch_metrics if m.get("val_cer") is not None]
            if cers:
                best = min(cers, key=lambda x: x[1])
                worst = max(cers, key=lambda x: x[1])
                self._write_log(f"  Best Epoch CER   : Epoch {best[0]} → {best[1]:.4f}")
                self._write_log(f"  Worst Epoch CER  : Epoch {worst[0]} → {worst[1]:.4f}")

        self._write_log(f"\n  Log file  : {self.log_file}")
        self._write_log(f"  JSON file : {self.json_file}")
        self._write_log(f"{'='*70}\n")

        # Final JSON save
        self.system_info["total_training_time_seconds"] = round(total_time, 1)
        self.system_info["total_training_time_human"] = str(timedelta(seconds=int(total_time)))
        if best_cer is not None:
            self.system_info["best_val_cer"] = best_cer
        self._save_json()
