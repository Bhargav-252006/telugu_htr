"""
scripts/generate_paper_figures.py

Generate publication-quality figures for the IEEE conference paper.

Outputs (saved to results/figures/):
  1. confusion_matrix.pdf     — Top-20 character confusion heatmap
  2. error_examples.pdf       — Qualitative error examples table
  3. training_curves.pdf      — CER/Loss vs Epoch for all models
  4. ablation_bar.pdf         — Bar chart comparing CER across ablation rows

Usage:
    python scripts/generate_paper_figures.py

Requirements:
    pip install matplotlib seaborn
"""

from __future__ import annotations

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works on headless servers)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Use a clean, publication-quality style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Figure 1: Confusion Matrix Heatmap
# ═══════════════════════════════════════════════════════════════════

def generate_confusion_matrix(model_type="ctc"):
    """
    Run evaluation and generate a character confusion heatmap.
    This requires the model checkpoints to be available.
    """
    import torch
    import yaml
    import unicodedata
    from collections import Counter

    from src.vocab import TeluguVocab
    from src.dataset import build_dataloader
    from src.evaluate import compute_cer_wer

    if model_type == "ctc":
        config_path = "configs/ctc_config.yaml"
        checkpoint_path = "checkpoints/ctc/best.pt"
    else:
        config_path = "configs/ar_config.yaml"
        checkpoint_path = "checkpoints/ar/best.pt"

    cfg = yaml.safe_load(open(config_path))
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = TeluguVocab.load(cfg["training"]["vocab_path"])

    loader = build_dataloader(
        dcfg.get("test_annotation", "data/raw/test/labels.txt"),
        dcfg.get("test_image_root", "data/raw/test"),
        vocab,
        split="test",
        batch_size=64,
        num_workers=dcfg.get("num_workers", 2),
        max_label_len=dcfg["max_label_len"],
        add_sos_eos=(model_type == "ar"),
    )

    # Load model
    if model_type == "ctc":
        from src.models.ctc_model import CTCModel
        model = CTCModel(
            vocab_size=len(vocab),
            d_model=mcfg["d_model"],
            lstm_hidden=mcfg["lstm_hidden"],
            lstm_layers=mcfg["lstm_layers"],
            dropout=mcfg["dropout"],
            pretrained=mcfg["pretrained"],
        )
    else:
        from src.models.ar_model import ARModel
        model = ARModel(
            vocab_size=len(vocab),
            sos_id=vocab.sos_id,
            eos_id=vocab.eos_id,
            d_model=mcfg["d_model"],
            nhead=mcfg["nhead"],
            num_decoder_layers=mcfg["num_decoder_layers"],
            dim_feedforward=mcfg["dim_feedforward"],
            dropout=mcfg["dropout"],
            max_label_len=mcfg["max_label_len"],
            label_smoothing=mcfg["label_smoothing"],
            pretrained=mcfg["pretrained"],
            num_encoder_layers=mcfg.get("num_encoder_layers", 2),
            high_res_temporal=mcfg.get("high_res_temporal", False),
            ctc_weight=mcfg.get("ctc_weight", 0.3),
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Collect predictions
    all_preds, all_gts = [], []
    with torch.no_grad():
        for images, labels, label_lens, image_widths in loader:
            images = images.to(device)
            labels = labels.to(device)
            label_lens = label_lens.to(device)
            image_widths = image_widths.to(device)

            if model_type == "ctc":
                pred_ids = model.greedy_decode(images, input_widths=image_widths)
                for i, (pred, llen) in enumerate(zip(pred_ids, label_lens.tolist())):
                    gt_ids = labels[i, :llen].tolist()
                    all_preds.append(unicodedata.normalize("NFC", vocab.decode(pred)))
                    all_gts.append(unicodedata.normalize("NFC", vocab.decode(gt_ids)))
            else:
                pred_ids = model.greedy_decode(images, max_len=36, input_widths=image_widths)
                for i, (pred, lab, llen) in enumerate(zip(pred_ids, labels, label_lens.tolist())):
                    gt_ids = lab[1:llen - 1].tolist()
                    all_preds.append(unicodedata.normalize("NFC", vocab.decode(pred)))
                    all_gts.append(unicodedata.normalize("NFC", vocab.decode(gt_ids)))

    # Build confusion counter
    confusion = Counter()
    for pred, gt in zip(all_preds, all_gts):
        for g, p in zip(gt, pred):
            if g != p:
                confusion[(g, p)] += 1

    # Get top-20 confusions
    top_confusions = confusion.most_common(20)

    # Get unique characters involved
    chars_gt = []
    chars_pred = []
    for (g, p), count in top_confusions:
        if g not in chars_gt:
            chars_gt.append(g)
        if p not in chars_pred:
            chars_pred.append(p)

    # Build matrix
    n_gt = len(chars_gt)
    n_pred = len(chars_pred)
    matrix = np.zeros((n_gt, n_pred))
    for (g, p), count in top_confusions:
        if g in chars_gt and p in chars_pred:
            matrix[chars_gt.index(g), chars_pred.index(p)] = count

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n_pred))
    ax.set_yticks(range(n_gt))
    ax.set_xticklabels(chars_pred, fontsize=12)
    ax.set_yticklabels(chars_gt, fontsize=12)
    ax.set_xlabel("Predicted Character", fontsize=12)
    ax.set_ylabel("Ground Truth Character", fontsize=12)
    ax.set_title(f"Character Confusion Matrix — {model_type.upper()} Model", fontsize=13)

    # Add count annotations
    for i in range(n_gt):
        for j in range(n_pred):
            if matrix[i, j] > 0:
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center",
                        fontsize=8, color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")

    fig.colorbar(im, ax=ax, label="Substitution Count")
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, f"confusion_matrix_{model_type}.pdf")
    plt.savefig(out_path)
    plt.savefig(out_path.replace(".pdf", ".png"))
    plt.close()
    print(f"[✓] Saved confusion matrix → {out_path}")

    # Also save the error examples for the qualitative figure
    errors = []
    for pred, gt in zip(all_preds, all_gts):
        if pred != gt:
            import editdistance
            cer = editdistance.eval(list(pred), list(gt)) / max(len(gt), 1)
            errors.append({"gt": gt, "pred": pred, "cer": cer})
    errors.sort(key=lambda x: x["cer"], reverse=True)

    # Save top-20 errors to JSON for the qualitative figure
    errors_path = os.path.join(RESULTS_DIR, f"error_examples_{model_type}.json")
    with open(errors_path, "w", encoding="utf-8") as f:
        json.dump(errors[:30], f, ensure_ascii=False, indent=2)
    print(f"[✓] Saved error examples → {errors_path}")

    return all_preds, all_gts, confusion


# ═══════════════════════════════════════════════════════════════════
# Figure 2: Qualitative Error Examples Table
# ═══════════════════════════════════════════════════════════════════

def generate_error_examples_figure(model_type="ctc"):
    """Generate a table figure showing top error examples."""
    errors_path = os.path.join(RESULTS_DIR, f"error_examples_{model_type}.json")
    if not os.path.exists(errors_path):
        print(f"[!] Run generate_confusion_matrix('{model_type}') first to generate error data")
        return

    with open(errors_path, "r", encoding="utf-8") as f:
        errors = json.load(f)

    # Pick the 8 most interesting errors (mix of high and medium CER)
    selected = errors[:4] + errors[10:12] + errors[20:22]  # 8 examples
    selected = selected[:8]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")

    # Create table data
    table_data = [["#", "Ground Truth", "Prediction", "CER (%)"]]
    for i, err in enumerate(selected):
        table_data.append([
            str(i + 1),
            err["gt"],
            err["pred"],
            f"{err['cer'] * 100:.1f}%"
        ])

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.06, 0.35, 0.35, 0.12],
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Header styling
    for j in range(4):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(table_data)):
        color = "#ecf0f1" if i % 2 == 0 else "white"
        for j in range(4):
            table[i, j].set_facecolor(color)

    ax.set_title(
        f"Representative Misrecognition Examples — {model_type.upper()} Model",
        fontsize=12, fontweight="bold", pad=20
    )

    out_path = os.path.join(FIGURES_DIR, f"error_examples_{model_type}.pdf")
    plt.savefig(out_path)
    plt.savefig(out_path.replace(".pdf", ".png"))
    plt.close()
    print(f"[✓] Saved error examples figure → {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Figure 3: Training Curves (CER and Loss vs Epoch)
# ═══════════════════════════════════════════════════════════════════

def generate_training_curves():
    """Generate training curves from JSON logs."""
    import glob

    # Auto-discover all JSON log files per model
    log_groups = {
        "CTC Baseline": sorted(glob.glob("logs/ctc/training_*.json")),
        "AR Transformer": sorted(glob.glob("logs/ar/training_*.json")),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = {"CTC Baseline": "#2980b9", "AR Transformer": "#e74c3c"}
    markers = {"CTC Baseline": "o", "AR Transformer": "s"}

    for name, log_paths in log_groups.items():
        if not log_paths:
            print(f"[!] No log files found for {name}")
            continue

        # Merge epoch_metrics from all log files for this model
        all_metrics = []
        for log_path in log_paths:
            print(f"  Loading: {log_path}")
            with open(log_path, "r") as f:
                data = json.load(f)
            metrics = data.get("epoch_metrics", [])
            all_metrics.extend(metrics)

        if not all_metrics:
            print(f"[!] No epoch_metrics found in {log_paths}")
            continue

        # De-duplicate by epoch (keep the last occurrence)
        seen = {}
        for m in all_metrics:
            seen[m["epoch"]] = m
        all_metrics = [seen[k] for k in sorted(seen.keys())]

        epochs = [e["epoch"] for e in all_metrics]
        train_losses = [e["train_loss"] for e in all_metrics]
        val_cers = [e["val_cer"] * 100 for e in all_metrics]  # Convert to percentage

        color = colors[name]
        marker = markers[name]

        # Plot 1: Val CER vs Epoch
        axes[0].plot(epochs, val_cers, color=color, marker=marker, markersize=3,
                     linewidth=1.5, label=name, alpha=0.8)

        # Plot 2: Train Loss vs Epoch
        axes[1].plot(epochs, train_losses, color=color, marker=marker, markersize=3,
                     linewidth=1.5, label=name, alpha=0.8)

        print(f"  {name}: {len(epochs)} epochs, final CER={val_cers[-1]:.2f}%, final loss={train_losses[-1]:.4f}")

    # Style plot 1
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation CER (%)")
    axes[0].set_title("(a) Character Error Rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    # Style plot 2
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Training Loss")
    axes[1].set_title("(b) Training Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training Convergence", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "training_curves.pdf")
    plt.savefig(out_path)
    plt.savefig(out_path.replace(".pdf", ".png"))
    plt.close()
    print(f"[✓] Saved training curves → {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Figure 4: Ablation Bar Chart
# ═══════════════════════════════════════════════════════════════════

def generate_ablation_bar_chart():
    """Generate a grouped bar chart from ablation results."""
    ablation_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    if not os.path.exists(ablation_path):
        print(f"[!] Ablation results not found: {ablation_path}")
        return

    with open(ablation_path, "r") as f:
        data = json.load(f)

    # Filter out metadata and broken rows
    models = []
    cers = []
    wers = []
    ci_lows = []
    ci_highs = []

    for name, res in data.items():
        if name.startswith("_"):
            continue
        if res.get("cer", 0) > 0.10:  # Skip broken rows (>10% CER)
            continue
        models.append(name.replace(" + ", "\n+ "))
        cers.append(res["cer"] * 100)
        wers.append(res["wer"] * 100)
        ci = res.get("cer_ci", [0, 0])
        ci_lows.append(res["cer"] * 100 - ci[0] * 100)
        ci_highs.append(ci[1] * 100 - res["cer"] * 100)

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # CER bars
    bars1 = ax1.bar(x - width / 2, cers, width, label="CER (%)",
                     color="#3498db", alpha=0.85,
                     yerr=[ci_lows, ci_highs], capsize=4, error_kw={"linewidth": 1.2})

    # WER bars
    bars2 = ax1.bar(x + width / 2, wers, width, label="WER (%)",
                     color="#e74c3c", alpha=0.85)

    ax1.set_xlabel("Model Configuration")
    ax1.set_ylabel("Error Rate (%)")
    ax1.set_title("Ablation Study — Test Set Results (N = 17,910)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=8)
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
                 f"{height:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
                 f"{height:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "ablation_bar.pdf")
    plt.savefig(out_path)
    plt.savefig(out_path.replace(".pdf", ".png"))
    plt.close()
    print(f"[✓] Saved ablation bar chart → {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Figure 5: Compound vs Simple CER Comparison
# ═══════════════════════════════════════════════════════════════════

def generate_compound_simple_comparison():
    """Bar chart comparing CER on compound vs simple characters."""
    ablation_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    if not os.path.exists(ablation_path):
        print(f"[!] Ablation results not found: {ablation_path}")
        return

    with open(ablation_path, "r") as f:
        data = json.load(f)

    models = []
    compound_cers = []
    simple_cers = []

    for name, res in data.items():
        if name.startswith("_"):
            continue
        if res.get("cer", 0) > 0.10:
            continue
        vb = res.get("virama_breakdown", {})
        models.append(name.replace(" + ", "\n+ "))
        compound_cers.append(vb.get("compound", {}).get("cer", 0) * 100)
        simple_cers.append(vb.get("simple", {}).get("cer", 0) * 100)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width / 2, compound_cers, width, label="Compound (with virama)",
                    color="#27ae60", alpha=0.85)
    bars2 = ax.bar(x + width / 2, simple_cers, width, label="Simple (without virama)",
                    color="#f39c12", alpha=0.85)

    ax.set_xlabel("Model Configuration")
    ax.set_ylabel("CER (%)")
    ax.set_title("CER by Character Complexity — Compound vs Simple", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                f"{height:.2f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                f"{height:.2f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "compound_vs_simple.pdf")
    plt.savefig(out_path)
    plt.savefig(out_path.replace(".pdf", ".png"))
    plt.close()
    print(f"[✓] Saved compound vs simple comparison → {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate IEEE paper figures")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip model evaluation (use cached error data)")
    parser.add_argument("--model", default="ctc", choices=["ctc", "ar"],
                        help="Which model to generate confusion matrix for")
    args = parser.parse_args()

    print("=" * 60)
    print("  Generating IEEE Paper Figures")
    print("=" * 60)

    # Figures that don't require model evaluation
    print("\n── Training Curves ──────────────────────────────────")
    generate_training_curves()

    print("\n── Ablation Bar Chart ───────────────────────────────")
    generate_ablation_bar_chart()

    print("\n── Compound vs Simple Comparison ────────────────────")
    generate_compound_simple_comparison()

    # Figures that require model evaluation
    if not args.skip_eval:
        print(f"\n── Confusion Matrix ({args.model.upper()}) ─────────────────────")
        generate_confusion_matrix(model_type=args.model)

        print(f"\n── Error Examples Figure ({args.model.upper()}) ─────────────────")
        generate_error_examples_figure(model_type=args.model)
    else:
        print("\n[SKIP] Model evaluation skipped (--skip-eval)")
        # Try to generate error examples from cached data
        generate_error_examples_figure(model_type=args.model)

    print("\n" + "=" * 60)
    print(f"  All figures saved to: {FIGURES_DIR}/")
    print("=" * 60)
