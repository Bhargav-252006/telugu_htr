"""
src/evaluate.py

Evaluation utilities for Telugu HTR.

Provides:
  - compute_cer_wer()          — CER and WER over a list of predictions
  - evaluate_model_ctc()       — full test-set eval for CTC model
  - evaluate_model_ar()        — full test-set eval for AR model (greedy + beam)
  - breakdown_by_virama()      — split results by compound-character words
  - print_error_examples()     — show worst-case failures
  - confusion_matrix_chars()   — character confusion analysis

Usage:
    python -m src.evaluate \
        --model_type ctc \
        --checkpoint checkpoints/ctc/best.pt \
        --config configs/ctc_config.yaml \
        --split test
"""

from __future__ import annotations
import argparse
import os
import unicodedata
import time
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

import editdistance
import torch
import yaml

from src.vocab import TeluguVocab, VIRAMA


# ═══════════════════════════════════════════════════════════════════
# Core metrics
# ═══════════════════════════════════════════════════════════════════

def compute_cer_wer(
    predictions: List[str],
    ground_truths: List[str],
) -> Tuple[float, float]:
    """
    Compute Character Error Rate (CER) and Word Error Rate (WER).

    CER = sum(edit_distance(pred, gt)) / sum(len(gt))
    WER = number of words where pred != gt / total words

    Returns (CER, WER) as floats in [0, 1].
    """
    assert len(predictions) == len(ground_truths), \
        f"Length mismatch: {len(predictions)} vs {len(ground_truths)}"

    total_chars  = 0
    total_edits  = 0
    word_errors  = 0

    for pred, gt in zip(predictions, ground_truths):
        # CER
        total_edits += editdistance.eval(list(pred), list(gt))
        total_chars += max(len(gt), 1)

        # WER
        if pred != gt:
            word_errors += 1

    cer = total_edits / max(total_chars, 1)
    wer = word_errors / max(len(predictions), 1)
    return cer, wer


def compute_bootstrap_ci(
    predictions: List[str],
    ground_truths: List[str],
    n_resamples: int = 1000,
) -> Tuple[float, float, float]:
    """
    Compute 95% Confidence Interval for CER using bootstrap resampling.
    Returns (mean_cer, lower_bound, upper_bound).
    """
    import numpy as np
    n = len(predictions)
    if n == 0:
        return 0.0, 0.0, 0.0
        
    preds = np.array(predictions)
    gts = np.array(ground_truths)
    
    bootstrap_cers = []
    for _ in range(n_resamples):
        indices = np.random.choice(n, n, replace=True)
        resample_preds = preds[indices].tolist()
        resample_gts = gts[indices].tolist()
        cer, _ = compute_cer_wer(resample_preds, resample_gts)
        bootstrap_cers.append(cer)
        
    bootstrap_cers = np.array(bootstrap_cers)
    mean_cer = np.mean(bootstrap_cers)
    lower = np.percentile(bootstrap_cers, 2.5)
    upper = np.percentile(bootstrap_cers, 97.5)
    return float(mean_cer), float(lower), float(upper)


# ═══════════════════════════════════════════════════════════════════
# Per-sample result dataclass
# ═══════════════════════════════════════════════════════════════════

class SampleResult:
    __slots__ = ("image_path", "gt", "pred", "cer", "correct")

    def __init__(self, image_path: str, gt: str, pred: str):
        self.image_path = image_path
        self.gt         = gt
        self.pred       = pred
        self.cer        = editdistance.eval(list(pred), list(gt)) / max(len(gt), 1)
        self.correct    = (pred == gt)


# ═══════════════════════════════════════════════════════════════════
# Virama / compound-character breakdown
# ═══════════════════════════════════════════════════════════════════

def breakdown_by_virama(
    predictions: List[str],
    ground_truths: List[str],
) -> Dict[str, dict]:
    """
    Split results into two groups:
      - "compound" : words whose GT contains Virama (్)
      - "simple"   : words without Virama

    Returns a dict with keys 'compound' and 'simple', each containing
    {'count', 'cer', 'wer'}.
    """
    groups: Dict[str, Tuple[List, List]] = {
        "compound": ([], []),
        "simple":   ([], []),
    }

    for pred, gt in zip(predictions, ground_truths):
        key = "compound" if VIRAMA in gt else "simple"
        groups[key][0].append(pred)
        groups[key][1].append(gt)

    results = {}
    for key, (preds, gts) in groups.items():
        if preds:
            cer, wer = compute_cer_wer(preds, gts)
            results[key] = {"count": len(preds), "cer": cer, "wer": wer}
        else:
            results[key] = {"count": 0, "cer": 0.0, "wer": 0.0}

    return results


# ═══════════════════════════════════════════════════════════════════
# Error analysis
# ═══════════════════════════════════════════════════════════════════

def print_error_examples(
    predictions: List[str],
    ground_truths: List[str],
    n: int = 20,
    sort_by_cer: bool = True,
):
    """Print the N highest-error examples."""
    errors = []
    for pred, gt in zip(predictions, ground_truths):
        if pred != gt:
            cer = editdistance.eval(list(pred), list(gt)) / max(len(gt), 1)
            errors.append((cer, gt, pred))

    if sort_by_cer:
        errors.sort(reverse=True)

    print(f"\n── Top-{min(n, len(errors))} error examples ──────────────────")
    for i, (cer, gt, pred) in enumerate(errors[:n]):
        print(f"  [{i+1:2d}] CER={cer:.3f}  GT='{gt}'  PRED='{pred}'")


def character_confusion_matrix(
    predictions: List[str],
    ground_truths: List[str],
    top_n: int = 20,
) -> Counter:
    """
    Approximate character confusion: align GT and pred at character level
    using the edit-distance traceback (simplified: just count substitutions
    from the naive alignment).

    Returns a Counter of (gt_char, pred_char) pairs, sorted by frequency.
    """
    confusion: Counter = Counter()

    for pred, gt in zip(predictions, ground_truths):
        # Simple prefix alignment (not traceback, but fast and good enough
        # for identifying common confusions)
        for g, p in zip(gt, pred):
            if g != p:
                confusion[(g, p)] += 1

    print(f"\n── Top-{top_n} character confusions ─────────────────────────")
    print(f"  {'GT':>6}  →  {'PRED':>6}  {'COUNT':>8}")
    for (g, p), count in confusion.most_common(top_n):
        print(f"  '{g}'  →  '{p}'  {count:>8}")

    return confusion


def avg_pred_length(predictions: List[str]) -> float:
    if not predictions:
        return 0.0
    return sum(len(p) for p in predictions) / len(predictions)


# ═══════════════════════════════════════════════════════════════════
# Full model evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model_ctc(
    model,
    loader,
    vocab:   TeluguVocab,
    device:  torch.device,
) -> Dict:
    """Run CTC model on a loader, return dict of metrics + lists."""
    model.eval()
    all_preds, all_gts = [], []
    inference_time = 0.0
    total_samples = 0

    for images, labels, label_lens, image_widths in loader:
        images       = images.to(device)
        labels       = labels.to(device)
        label_lens   = label_lens.to(device)
        image_widths = image_widths.to(device)

        t0 = time.time()
        pred_ids = model.greedy_decode(images, input_widths=image_widths)
        inference_time += time.time() - t0
        total_samples += images.size(0)

        for i, (pred, llen) in enumerate(zip(pred_ids, label_lens.tolist())):
            gt_ids = labels[i, :llen].tolist()
            pred_str = unicodedata.normalize("NFC", vocab.decode(pred))
            gt_str   = unicodedata.normalize("NFC", vocab.decode(gt_ids))
            all_preds.append(pred_str)
            all_gts.append(gt_str)

    cer, wer = compute_cer_wer(all_preds, all_gts)
    virama_bd = breakdown_by_virama(all_preds, all_gts)
    mean_cer, lower_ci, upper_ci = compute_bootstrap_ci(all_preds, all_gts)

    return {
        "cer": cer, "wer": wer,
        "cer_ci": (lower_ci, upper_ci),
        "inference_time_ms_per_sample": (inference_time / max(total_samples, 1)) * 1000,
        "predictions": all_preds,
        "ground_truths": all_gts,
        "virama_breakdown": virama_bd,
        "avg_pred_len": avg_pred_length(all_preds),
    }


@torch.no_grad()
def evaluate_model_ar(
    model,
    loader,
    vocab:         TeluguVocab,
    device:        torch.device,
    use_beam:      bool  = False,
    beam_size:     int   = 5,
    constrain:     bool  = True,
    constrain_penalty: float = None,
) -> Dict:
    """Run AR model on a loader, return dict of metrics + lists."""
    model.eval()
    all_preds, all_gts = [], []
    inference_time = 0.0
    total_samples = 0

    for images, labels, label_lens, image_widths in loader:
        images       = images.to(device)
        labels       = labels.to(device)
        label_lens   = label_lens.to(device)
        image_widths = image_widths.to(device)

        t0 = time.time()
        if use_beam:
            pred_ids = model.beam_decode(
                images, beam_size=beam_size,
                max_len=36, vocab=vocab if constrain else None,
                constrain=constrain,
                constrain_penalty=constrain_penalty,
                input_widths=image_widths,
                length_penalty=0.6
            )
        else:
            pred_ids = model.greedy_decode(
                images, max_len=36,
                vocab=vocab if constrain else None,
                constrain=constrain,
                constrain_penalty=constrain_penalty,
                input_widths=image_widths
            )
        inference_time += time.time() - t0
        total_samples += images.size(0)

        for i, (pred, lab, llen) in enumerate(zip(pred_ids, labels, label_lens.tolist())):
            # Skip SOS (index 0) and EOS when extracting GT
            gt_ids = lab[1:llen - 1].tolist()
            pred_str = unicodedata.normalize("NFC", vocab.decode(pred))
            gt_str   = unicodedata.normalize("NFC", vocab.decode(gt_ids))
            all_preds.append(pred_str)
            all_gts.append(gt_str)

    cer, wer = compute_cer_wer(all_preds, all_gts)
    virama_bd = breakdown_by_virama(all_preds, all_gts)
    mean_cer, lower_ci, upper_ci = compute_bootstrap_ci(all_preds, all_gts)

    return {
        "cer": cer, "wer": wer,
        "cer_ci": (lower_ci, upper_ci),
        "inference_time_ms_per_sample": (inference_time / max(total_samples, 1)) * 1000,
        "predictions": all_preds,
        "ground_truths": all_gts,
        "virama_breakdown": virama_bd,
        "avg_pred_len": avg_pred_length(all_preds),
    }


# ═══════════════════════════════════════════════════════════════════
# Ablation table printer
# ═══════════════════════════════════════════════════════════════════

def print_ablation_table(results: Dict[str, Dict]):
    """
    Print a formatted ablation table.

    results = {
        "CTC Baseline":           {"cer": 0.12, "wer": 0.45, ...},
        "AR (no constraint)":     {"cer": 0.09, ...},
        "AR + Telugu constraint": {"cer": 0.07, ...},
        ...
    }
    """
    print("\n" + "═" * 100)
    print(f"  {'Model':<30} {'CER':>8} {'WER':>8} "
          f"{'95% CI':>16} {'Comp CER':>10} {'Speed':>10}")
    print("─" * 100)
    for name, res in results.items():
        vb      = res.get("virama_breakdown", {})
        comp    = vb.get("compound", {})
        simple  = vb.get("simple", {})
        ci      = res.get("cer_ci", (0.0, 0.0))
        speed   = res.get("inference_time_ms_per_sample", 0.0)
        print(
            f"  {name:<30} "
            f"{res['cer'] * 100:>7.2f}% "
            f"{res['wer'] * 100:>7.2f}% "
            f"[{ci[0]*100:>5.2f}, {ci[1]*100:>5.2f}] "
            f"{comp.get('cer', 0.0) * 100:>9.2f}% "
            f"{speed:>7.1f} ms"
        )
    print("═" * 100 + "\n")


# ═══════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",  choices=["ctc", "ar"], required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--config",      required=True)
    parser.add_argument("--split",       default="test")
    parser.add_argument("--beam",        action="store_true")
    parser.add_argument(
        "--constrain-penalty",
        type=float,
        default=None,
        help="Soft penalty for invalid bigrams instead of -inf",
    )
    parser.add_argument("--beam_size",   type=int, default=5)
    parser.add_argument("--no_constrain", action="store_true")
    args = parser.parse_args()

    cfg   = yaml.safe_load(open(args.config))
    dcfg  = cfg["data"]
    mcfg  = cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.vocab import TeluguVocab
    vocab  = TeluguVocab.load(cfg["training"]["vocab_path"])

    ann_key  = f"{args.split}_annotation"
    root_key = f"{args.split}_image_root"
    from src.dataset import build_dataloader
    loader = build_dataloader(
        dcfg.get(ann_key,  f"data/raw/{args.split}/labels.txt"),
        dcfg.get(root_key, f"data/raw/{args.split}"),
        vocab,
        split       = args.split,
        batch_size  = 64,
        num_workers = dcfg.get("num_workers", 4),
        max_label_len = dcfg["max_label_len"],
        add_sos_eos = (args.model_type == "ar"),
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if args.model_type == "ctc":
        from src.models.ctc_model import CTCModel
        model = CTCModel(vocab_size=len(vocab), **{k: mcfg[k] for k in
            ["d_model", "lstm_hidden", "lstm_layers", "dropout", "pretrained"]})
        model.load_state_dict(ckpt["model"])
        model.to(device)
        res = evaluate_model_ctc(model, loader, vocab, device)
    else:
        from src.models.ar_model import ARModel
        model = ARModel(vocab_size=len(vocab), sos_id=vocab.sos_id, eos_id=vocab.eos_id,
                        num_encoder_layers=mcfg.get("num_encoder_layers", 2),
                        high_res_temporal=mcfg.get("high_res_temporal", False),
                        ctc_weight=mcfg.get("ctc_weight", 0.3),
                        **{k: mcfg[k] for k in
                           ["d_model", "nhead", "num_decoder_layers", "dim_feedforward",
                            "dropout", "max_label_len", "label_smoothing", "pretrained"]})
        model.load_state_dict(ckpt["model"])
        model.to(device)
        res = evaluate_model_ar(
            model, loader, vocab, device,
            use_beam  = args.beam,
            beam_size = args.beam_size,
            constrain = not args.no_constrain,
            constrain_penalty = args.constrain_penalty,
        )

    print(f"\n── Results on [{args.split}] ──────────────────────────────")
    print(f"  CER           : {res['cer']:.4f}")
    print(f"  WER           : {res['wer']:.4f}")
    if "cer_ci" in res:
        print(f"  95% CI (CER)  : [{res['cer_ci'][0]:.4f}, {res['cer_ci'][1]:.4f}]")
    if "inference_time_ms_per_sample" in res:
        print(f"  Speed         : {res['inference_time_ms_per_sample']:.1f} ms/sample")
    print(f"  Avg pred len  : {res['avg_pred_len']:.1f}")
    vb = res["virama_breakdown"]
    print(f"  Compound CER  : {vb['compound']['cer']:.4f}  (n={vb['compound']['count']})")
    print(f"  Simple CER    : {vb['simple']['cer']:.4f}   (n={vb['simple']['count']})")

    print_error_examples(res["predictions"], res["ground_truths"], n=20)
    character_confusion_matrix(res["predictions"], res["ground_truths"], top_n=20)
