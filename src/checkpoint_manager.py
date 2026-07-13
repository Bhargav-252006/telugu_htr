"""
src/checkpoint_manager.py

Rolling 2-slot checkpoint manager.

DESIGN
──────
At most 3 files exist on disk at any time:

    checkpoints/<run>/
        best.pt        ← best val CER ever seen
        current.pt     ← end of most recent epoch
        previous.pt    ← end of the epoch before that

Algorithm on each save:
    1. If current.pt exists  →  rename it to previous.pt  (overwrites previous.pt)
    2. Write new current.pt
    3. If new val CER < best CER  →  copy current.pt to best.pt

DISK COST
─────────
For a ~200 MB model (fp32) or ~100 MB (fp16):
    3 files × 200 MB = 600 MB total  — fixed forever, no growth.

ROLLBACK
─────────
    - 1 epoch back  : load previous.pt
    - Best ever     : load best.pt
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """
    Manages rolling 2-slot checkpoints: best / current / previous.

    Parameters
    ----------
    save_dir  : directory where checkpoint files are stored.
    model     : the nn.Module to save.
    optimizer : the optimizer.
    scheduler : the LR scheduler.
    """

    BEST_NAME     = "best.pt"
    CURRENT_NAME  = "current.pt"
    PREVIOUS_NAME = "previous.pt"

    def __init__(
        self,
        save_dir:  str,
        model,
        optimizer,
        scheduler = None,
    ):
        self.save_dir  = save_dir
        self.model     = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        os.makedirs(save_dir, exist_ok=True)

        self.best_cer: float = float("inf")

        # Paths
        self.best_path     = os.path.join(save_dir, self.BEST_NAME)
        self.current_path  = os.path.join(save_dir, self.CURRENT_NAME)
        self.previous_path = os.path.join(save_dir, self.PREVIOUS_NAME)

    # ── Core save ─────────────────────────────────────────────────

    def save(self, epoch: int, val_cer: float, extra: Optional[Dict] = None) -> str:
        """
        Save at the end of an epoch.

        Steps:
          1. Promote current → previous  (if current exists)
          2. Write new current
          3. If val_cer < best_cer: update best

        Parameters
        ----------
        epoch   : current epoch number.
        val_cer : validation CER (used to decide whether to update best).
        extra   : any extra keys to include in the checkpoint dict.

        Returns
        -------
        Path of the file that was just written (current_path).
        """
        # Step 1: promote current → previous
        if os.path.exists(self.current_path):
            shutil.move(self.current_path, self.previous_path)

        # Step 2: update best_cer BEFORE building dict so checkpoint has correct value
        improved = False
        if val_cer < self.best_cer:
            self.best_cer = val_cer
            improved = True

        # Step 3: write new current (now contains the updated best_cer)
        ckpt = self._build_dict(epoch, val_cer, extra)
        torch.save(ckpt, self.current_path)

        # Step 4: copy to best if improved
        if improved:
            shutil.copy2(self.current_path, self.best_path)

        self._print_status(epoch, val_cer, improved)
        return self.current_path


    # ── Load helpers ──────────────────────────────────────────────

    def load(self, path: str, device: str = "cpu") -> Dict[str, Any]:
        """
        Load a checkpoint from any path (current, previous, best, or thermal).
        Restores model, optimizer, and scheduler weights in-place.

        Returns the full checkpoint dict (contains 'epoch', 'val_cer', etc).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=device, weights_only=False)

        self.model.load_state_dict(ckpt["model"])

        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

        if self.scheduler is not None and "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])

        self.best_cer = ckpt.get("best_cer", float("inf"))

        print(
            f"[CheckpointManager] Loaded '{os.path.basename(path)}' — "
            f"epoch={ckpt.get('epoch', '?')}  "
            f"val_cer={ckpt.get('val_cer', '?')}"
        )
        return ckpt

    def load_best(self, device: str = "cpu") -> Dict:
        return self.load(self.best_path, device)

    def load_current(self, device: str = "cpu") -> Dict:
        return self.load(self.current_path, device)

    def load_previous(self, device: str = "cpu") -> Dict:
        """Roll back one epoch."""
        return self.load(self.previous_path, device)

    # ── Status ───────────────────────────────────────────────────

    def disk_usage(self) -> str:
        """Return a human-readable summary of files on disk."""
        files = [
            (self.best_path,     "best"),
            (self.current_path,  "current"),
            (self.previous_path, "previous"),
        ]
        lines = []
        total_bytes = 0
        for path, label in files:
            if os.path.exists(path):
                size = os.path.getsize(path)
                total_bytes += size
                lines.append(f"  {label:>10}.pt  {size / 1024 / 1024:>8.1f} MB")
            else:
                lines.append(f"  {label:>10}.pt  {'—':>8}")
        lines.append(f"  {'TOTAL':>10}       {total_bytes / 1024 / 1024:>8.1f} MB")
        return "\n".join(lines)

    def _print_status(self, epoch: int, val_cer: float, improved: bool):
        tag = "  ★ NEW BEST" if improved else ""
        print(
            f"  [Checkpoint] epoch={epoch}  val_cer={val_cer:.4f}"
            f"  best={self.best_cer:.4f}{tag}"
        )
        if epoch % 5 == 0:   # show disk usage every 5 epochs
            print(f"\n  Disk usage:\n{self.disk_usage()}\n")

    def _build_dict(
        self, epoch: int, val_cer: float, extra: Optional[Dict]
    ) -> Dict:
        ckpt: Dict[str, Any] = {
            "epoch":    epoch,
            "val_cer":  val_cer,
            "best_cer": self.best_cer,
            "model":    self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            ckpt["scheduler"] = self.scheduler.state_dict()
        if extra:
            ckpt.update(extra)
        return ckpt
