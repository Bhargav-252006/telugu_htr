"""
scripts/evaluate_benchmark.py

Evaluate trained models on the IIIT-INDIC-HW-WORDS Telugu split for
cross-comparison with published results.

This script supports both:
  1. Your private dataset (internal evaluation)
  2. IIIT-INDIC-HW-WORDS Telugu split (external benchmark)

Usage:
    # Internal test set
    python scripts/evaluate_benchmark.py \
        --config configs/ar_config.yaml \
        --checkpoint checkpoints/ar/best.pt \
        --model_type ar

    # External benchmark (IIIT-INDIC-HW-WORDS)
    python scripts/evaluate_benchmark.py \
        --config configs/ar_config.yaml \
        --checkpoint checkpoints/ar/best.pt \
        --model_type ar \
        --benchmark_annotation data/benchmark/iiit_telugu/labels.txt \
        --benchmark_image_root data/benchmark/iiit_telugu/images

    # Full ablation table across all models
    python scripts/evaluate_benchmark.py --run_ablation \
        --ctc_config configs/ctc_config.yaml \
        --ctc_checkpoint checkpoints/ctc/best.pt \
        --ar_config configs/ar_config.yaml \
        --ar_checkpoint checkpoints/ar/best.pt \
        --ar_no_ctc_config configs/ar_no_ctc_config.yaml \
        --ar_no_ctc_checkpoint checkpoints/ar_no_ctc/best.pt
"""

from __future__ import annotations
import argparse
import os
import sys
import json
from datetime import datetime

import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vocab import TeluguVocab
from src.dataset import build_dataloader
from src.models.ctc_model import CTCModel
from src.models.ar_model import ARModel
from src.evaluate import (
    evaluate_model_ctc,
    evaluate_model_ar,
    print_ablation_table,
    compute_cer_wer,
    breakdown_by_virama,
    compute_bootstrap_ci,
    print_error_examples,
    character_confusion_matrix,
)


def load_model(model_type, config_path, checkpoint_path, device):
    """Load a trained model from config + checkpoint."""
    cfg = yaml.safe_load(open(config_path))
    mcfg = cfg["model"]
    dcfg = cfg["data"]

    vocab = TeluguVocab.load(cfg["training"]["vocab_path"])

    if model_type == "ctc":
        model = CTCModel(
            vocab_size=len(vocab),
            d_model=mcfg["d_model"],
            lstm_hidden=mcfg["lstm_hidden"],
            lstm_layers=mcfg["lstm_layers"],
            dropout=mcfg["dropout"],
            pretrained=mcfg["pretrained"],
        )
    else:
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

    return model, vocab, cfg


def evaluate_on_split(model, model_type, vocab, cfg, split, device,
                      annotation_override=None, image_root_override=None,
                      use_beam=False, beam_size=5, constrain=True,
                      constrain_penalty=None):
    """Evaluate a model on a given data split or external benchmark."""
    dcfg = cfg["data"]

    ann_file = annotation_override or dcfg.get(
        f"{split}_annotation", f"data/raw/{split}/labels.txt"
    )
    img_root = image_root_override or dcfg.get(
        f"{split}_image_root", f"data/raw/{split}"
    )

    loader = build_dataloader(
        ann_file, img_root, vocab,
        split=split,
        batch_size=64,
        num_workers=dcfg.get("num_workers", 2),
        max_label_len=dcfg["max_label_len"],
        add_sos_eos=(model_type == "ar"),
    )

    if model_type == "ctc":
        return evaluate_model_ctc(model, loader, vocab, device)
    else:
        return evaluate_model_ar(
            model, loader, vocab, device,
            use_beam=use_beam,
            beam_size=beam_size,
            constrain=constrain,
            constrain_penalty=constrain_penalty,
        )


def run_full_ablation(args, device):
    """Run the complete ablation table across all model variants."""
    results = {}

    # ── Row 1: CTC Baseline ──────────────────────────────────────
    if args.ctc_checkpoint and os.path.exists(args.ctc_checkpoint):
        print("\n" + "=" * 60)
        print("  Evaluating: CTC Baseline")
        print("=" * 60)
        model, vocab, cfg = load_model("ctc", args.ctc_config, args.ctc_checkpoint, device)
        results["CTC Baseline"] = evaluate_on_split(
            model, "ctc", vocab, cfg, "test", device
        )
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    else:
        print("[SKIP] CTC Baseline — checkpoint not found")

    # ── Row 2: AR (no CTC aux) — the new ablation row ───────────
    if args.ar_no_ctc_checkpoint and os.path.exists(args.ar_no_ctc_checkpoint):
        print("\n" + "=" * 60)
        print("  Evaluating: AR (no CTC aux)")
        print("=" * 60)
        model, vocab, cfg = load_model("ar", args.ar_no_ctc_config, args.ar_no_ctc_checkpoint, device)
        results["AR (no CTC aux)"] = evaluate_on_split(
            model, "ar", vocab, cfg, "test", device, constrain=False
        )
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    else:
        print("[SKIP] AR (no CTC aux) — checkpoint not found")

    # ── Row 3: AR unconstrained ──────────────────────────────────
    if args.ar_checkpoint and os.path.exists(args.ar_checkpoint):
        print("\n" + "=" * 60)
        print("  Evaluating: AR (unconstrained)")
        print("=" * 60)
        model, vocab, cfg = load_model("ar", args.ar_config, args.ar_checkpoint, device)
        results["AR (unconstrained)"] = evaluate_on_split(
            model, "ar", vocab, cfg, "test", device, constrain=False
        )

        # ── Row 4: AR + Telugu constraint ────────────────────────
        print("\n" + "=" * 60)
        print("  Evaluating: AR + Telugu constraint")
        print("=" * 60)
        tcfg = cfg["training"]
        results["AR + Telugu constraint"] = evaluate_on_split(
            model, "ar", vocab, cfg, "test", device,
            constrain=True,
            constrain_penalty=tcfg.get("constrain_penalty"),
        )

        # ── Row 5: AR + constraint + beam search ─────────────────
        print("\n" + "=" * 60)
        print("  Evaluating: AR + constraint + beam(5)")
        print("=" * 60)
        results["AR + constraint + beam(5)"] = evaluate_on_split(
            model, "ar", vocab, cfg, "test", device,
            use_beam=True,
            beam_size=tcfg.get("beam_size", 5),
            constrain=True,
            constrain_penalty=tcfg.get("constrain_penalty"),
        )
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    else:
        print("[SKIP] AR models — checkpoint not found")

    # ── Print ablation table ─────────────────────────────────────
    if results:
        print_ablation_table(results)

        # Save results to JSON for paper
        save_path = os.path.join("results", "ablation_results.json")
        os.makedirs("results", exist_ok=True)
        serializable = {}
        for name, res in results.items():
            serializable[name] = {
                k: v for k, v in res.items()
                if k not in ("predictions", "ground_truths")
            }
        serializable["_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\n[Saved] Ablation results → {save_path}")

    # ── External benchmark (if provided) ─────────────────────────
    if args.benchmark_annotation and args.ar_checkpoint:
        print("\n" + "=" * 60)
        print("  Evaluating on External Benchmark (IIIT-INDIC-HW-WORDS)")
        print("=" * 60)
        model, vocab, cfg = load_model("ar", args.ar_config, args.ar_checkpoint, device)
        tcfg = cfg["training"]

        benchmark_results = {}

        # AR + constraint on benchmark
        benchmark_results["AR + constraint (benchmark)"] = evaluate_on_split(
            model, "ar", vocab, cfg, "test", device,
            annotation_override=args.benchmark_annotation,
            image_root_override=args.benchmark_image_root,
            constrain=True,
            constrain_penalty=tcfg.get("constrain_penalty"),
        )

        print_ablation_table(benchmark_results)

        # Save benchmark results
        bench_path = os.path.join("results", "benchmark_results.json")
        bench_serializable = {}
        for name, res in benchmark_results.items():
            bench_serializable[name] = {
                k: v for k, v in res.items()
                if k not in ("predictions", "ground_truths")
            }
        bench_serializable["_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
            "benchmark_annotation": args.benchmark_annotation,
            "benchmark_image_root": args.benchmark_image_root,
        }
        with open(bench_path, "w", encoding="utf-8") as f:
            json.dump(bench_serializable, f, indent=2, default=str)
        print(f"[Saved] Benchmark results → {bench_path}")

        # Print comparison with published results
        print("\n── Comparison with Published Telugu HTR Results ─────────")
        print(f"  {'Method':<40} {'CER':>8}")
        print("  " + "─" * 50)
        # Published baselines (update these from actual papers)
        print(f"  {'Dutta et al. (2018) CNN-RNN-CTC':<40} {'~15.0%':>8}")
        print(f"  {'Deshpande et al. (2021) Attn-based':<40} {'~10.2%':>8}")
        print(f"  {'Ours: CTC Baseline':<40}", end="")
        if "CTC Baseline" in results:
            print(f" {results['CTC Baseline']['cer']*100:>7.2f}%")
        else:
            print(f" {'N/A':>8}")
        print(f"  {'Ours: AR + Telugu constraint':<40}", end="")
        if "AR + Telugu constraint" in results:
            print(f" {results['AR + Telugu constraint']['cer']*100:>7.2f}%")
        else:
            print(f" {'N/A':>8}")
        bench_res = benchmark_results.get("AR + constraint (benchmark)")
        if bench_res:
            print(f"  {'Ours: AR + constraint (IIIT-HW)':<40} {bench_res['cer']*100:>7.2f}%")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Telugu HTR models and generate ablation table"
    )

    # Single model evaluation
    parser.add_argument("--model_type", choices=["ctc", "ar"])
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", default="test")

    # Full ablation mode
    parser.add_argument("--run_ablation", action="store_true",
                        help="Run full ablation table across all models")
    parser.add_argument("--ctc_config", default="configs/ctc_config.yaml")
    parser.add_argument("--ctc_checkpoint", default="checkpoints/ctc/best.pt")
    parser.add_argument("--ar_config", default="configs/ar_config.yaml")
    parser.add_argument("--ar_checkpoint", default="checkpoints/ar/best.pt")
    parser.add_argument("--ar_no_ctc_config", default="configs/ar_no_ctc_config.yaml")
    parser.add_argument("--ar_no_ctc_checkpoint", default="checkpoints/ar_no_ctc/best.pt")

    # External benchmark
    parser.add_argument("--benchmark_annotation",
                        help="Path to external benchmark labels.txt")
    parser.add_argument("--benchmark_image_root",
                        help="Path to external benchmark image directory")

    # Decoding options
    parser.add_argument("--beam", action="store_true")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--no_constrain", action="store_true")
    parser.add_argument("--constrain_penalty", type=float, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[benchmark] Device: {device}")

    if args.run_ablation:
        run_full_ablation(args, device)
    elif args.config and args.checkpoint and args.model_type:
        # Single model evaluation
        model, vocab, cfg = load_model(
            args.model_type, args.config, args.checkpoint, device
        )
        res = evaluate_on_split(
            model, args.model_type, vocab, cfg, args.split, device,
            annotation_override=args.benchmark_annotation,
            image_root_override=args.benchmark_image_root,
            use_beam=args.beam,
            beam_size=args.beam_size,
            constrain=not args.no_constrain,
            constrain_penalty=args.constrain_penalty,
        )
        print(f"\n── Results on [{args.split}] ──────────────────────────────")
        print(f"  CER           : {res['cer']:.4f}")
        print(f"  WER           : {res['wer']:.4f}")
        if "cer_ci" in res:
            print(f"  95% CI (CER)  : [{res['cer_ci'][0]:.4f}, {res['cer_ci'][1]:.4f}]")
        if "inference_time_ms_per_sample" in res:
            print(f"  Speed         : {res['inference_time_ms_per_sample']:.1f} ms/sample")
        vb = res["virama_breakdown"]
        print(f"  Compound CER  : {vb['compound']['cer']:.4f}  (n={vb['compound']['count']})")
        print(f"  Simple CER    : {vb['simple']['cer']:.4f}   (n={vb['simple']['count']})")
        print_error_examples(res["predictions"], res["ground_truths"], n=20)
        character_confusion_matrix(res["predictions"], res["ground_truths"], top_n=20)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
