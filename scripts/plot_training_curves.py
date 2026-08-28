"""
Generate training curves plot from the actual training log JSON.
Can be run locally (only needs matplotlib + the JSON log file).

Usage:
    python scripts/plot_training_curves.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "ar_v2", "training_20260731_134445.json")
out_path = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "training_curves.png")

with open(log_path) as f:
    data = json.load(f)

epochs = [e["epoch"] for e in data["epoch_metrics"]]
train_loss = [e["train_loss"] for e in data["epoch_metrics"]]
val_cer = [e["val_cer"] * 100 for e in data["epoch_metrics"]]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(epochs, train_loss, "b-", linewidth=1.5, label="Train Loss")
ax1.set_ylabel("Joint Loss (CE + CTC)")
ax1.set_title("AR Transformer v2 \u2014 Training Curves")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, val_cer, "g-", linewidth=1.5, label="Val CER (%)")
best_idx = int(np.argmin(val_cer))
ax2.plot(epochs[best_idx], val_cer[best_idx], "r*", markersize=14,
         label=f"Best: {val_cer[best_idx]:.2f}% (epoch {epochs[best_idx]})")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("CER (%)")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.tight_layout()
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved {out_path}")
