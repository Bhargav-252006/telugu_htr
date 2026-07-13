"""
src/train_ctc.py

Training script for the CTC baseline model.

Usage:
    python -m src.train_ctc --config configs/ctc_config.yaml
    python -m src.train_ctc --config configs/ctc_config.yaml --resume checkpoints/ctc/current.pt
"""

from __future__ import annotations
import argparse
import os
import time
import unicodedata

import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from src.vocab import TeluguVocab
from src.dataset import build_dataloader
from src.models.ctc_model import CTCModel
from src.evaluate import compute_cer_wer
from src.checkpoint_manager import CheckpointManager
from src.training_logger import TrainingLogger


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_or_build_vocab(vocab_path, train_ann, val_ann) -> TeluguVocab:
    if os.path.exists(vocab_path):
        vocab = TeluguVocab.load(vocab_path)
    else:
        print("[train_ctc] Building vocab from training data ...")
        vocab = TeluguVocab.from_annotation_files([train_ann], build_matrix=True)
        vocab.validate_against_split(val_ann, split_name="val")
        vocab.save(vocab_path)
    return vocab


# ═══════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, loader, vocab, device, writer, epoch):
    model.eval()
    all_preds, all_gts = [], []
    total_loss   = 0.0
    n_batches    = 0
    pred_lengths = []

    for images, labels, label_lens, image_widths in loader:
        images       = images.to(device)
        labels       = labels.to(device)
        label_lens   = label_lens.to(device)
        image_widths = image_widths.to(device)

        loss = model.compute_loss(images, labels, label_lens, input_widths=image_widths)
        total_loss += loss.item()
        n_batches  += 1

        pred_ids = model.greedy_decode(images, input_widths=image_widths)
        pred_lengths.extend([len(p) for p in pred_ids])

        for i, (pred, llen) in enumerate(zip(pred_ids, label_lens.tolist())):
            gt_ids = labels[i, :llen].tolist()
            all_preds.append(unicodedata.normalize("NFC", vocab.decode(pred)))
            all_gts.append(unicodedata.normalize("NFC", vocab.decode(gt_ids)))

    cer, wer     = compute_cer_wer(all_preds, all_gts)
    avg_loss     = total_loss / max(n_batches, 1)
    avg_pred_len = sum(pred_lengths) / max(len(pred_lengths), 1)

    writer.add_scalar("val/loss",         avg_loss,     epoch)
    writer.add_scalar("val/CER",          cer,          epoch)
    writer.add_scalar("val/WER",          wer,          epoch)
    writer.add_scalar("val/avg_pred_len", avg_pred_len, epoch)

    print(f"  [Val epoch {epoch}] loss={avg_loss:.4f}  CER={cer:.4f}  "
          f"WER={wer:.4f}  avg_pred_len={avg_pred_len:.1f}")
    return cer


# ═══════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════

def train(config_path: str, resume_path: str = None):
    cfg  = load_config(config_path)
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        
    # Reproducibility
    torch.manual_seed(42)
    import random, numpy as np
    random.seed(42)
    np.random.seed(42)

    print(f"[train_ctc] Device: {device}")

    # ── Training Logger ──────────────────────────────────────────
    tlogger = TrainingLogger(log_dir=tcfg["log_dir"], model_name="CTC Baseline")

    # ── Vocab ────────────────────────────────────────────────────
    vocab = load_or_build_vocab(
        tcfg["vocab_path"],
        dcfg["train_annotation"],
        dcfg["val_annotation"],
    )
    vocab_size = len(vocab)
    print(f"[train_ctc] Vocab size: {vocab_size}")

    # ── Data ─────────────────────────────────────────────────────
    train_loader = build_dataloader(
        dcfg["train_annotation"], dcfg["train_image_root"], vocab,
        split         = "train",
        batch_size    = tcfg["batch_size"],
        num_workers   = dcfg["num_workers"],
        max_label_len = dcfg["max_label_len"],
        add_sos_eos   = False,
        use_elastic   = dcfg.get("use_elastic", False),
    )
    val_loader = build_dataloader(
        dcfg["val_annotation"], dcfg["val_image_root"], vocab,
        split         = "val",
        batch_size    = tcfg["batch_size"],
        num_workers   = dcfg["num_workers"],
        max_label_len = dcfg["max_label_len"],
        add_sos_eos   = False,
    )

    # ── Model ────────────────────────────────────────────────────
    model = CTCModel(
        vocab_size  = vocab_size,
        d_model     = mcfg["d_model"],
        lstm_hidden = mcfg["lstm_hidden"],
        lstm_layers = mcfg["lstm_layers"],
        dropout     = mcfg["dropout"],
        pretrained  = mcfg["pretrained"],
    ).to(device)
    print(f"[train_ctc] Parameters: {model.count_params():,}")
    tlogger.log_system_info(model=model, config=cfg)

    # ── Optimiser + Scheduler ────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr           = tcfg["lr"],
        weight_decay = tcfg["weight_decay"],
    )
    total_steps = len(train_loader) * tcfg["epochs"]
    scheduler   = OneCycleLR(
        optimizer,
        max_lr          = tcfg["lr"],
        total_steps     = total_steps,
        pct_start       = 0.1,
        anneal_strategy = "cos",
    )
    scaler = GradScaler("cuda", enabled=tcfg["mixed_precision"])

    # ── Checkpoint manager (rolling 2-slot) ───────────────────────
    ckpt_mgr = CheckpointManager(
        save_dir  = tcfg["save_dir"],
        model     = model,
        optimizer = optimizer,
        scheduler = scheduler,
    )

    start_epoch = 1
    if resume_path and os.path.exists(resume_path):
        ckpt = ckpt_mgr.load(resume_path, device=str(device))
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[train_ctc] Resuming from epoch {start_epoch}")

    # ── Logging ──────────────────────────────────────────────────
    writer      = SummaryWriter(tcfg["log_dir"])
    global_step = (start_epoch - 1) * len(train_loader)

    # ── Training ─────────────────────────────────────────────────
    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        t0 = time.time()
        tlogger.log_epoch_start(epoch)

        for batch_idx, (images, labels, label_lens, image_widths) in enumerate(train_loader):
            images       = images.to(device)
            labels       = labels.to(device)
            label_lens   = label_lens.to(device)
            image_widths = image_widths.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=tcfg["mixed_precision"]):
                loss = model.compute_loss(images, labels, label_lens, input_widths=image_widths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss  += loss.item()
            epoch_samples += images.size(0)
            global_step += 1

            if batch_idx % tcfg["log_interval"] == 0:
                lr_now = scheduler.get_last_lr()[0]
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr",   lr_now,      global_step)
                print(f"  Epoch {epoch}/{tcfg['epochs']}  "
                      f"Step {batch_idx}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  lr={lr_now:.2e}")

            # Log GPU stats every 200 steps
            if batch_idx % 200 == 0 and batch_idx > 0:
                tlogger.log_gpu_stats(epoch, batch_idx)

        avg_epoch_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}] avg_loss={avg_epoch_loss:.4f}  time={elapsed:.0f}s")
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)

        # ── Validate + save rolling checkpoint ───────────────────
        if epoch % tcfg["val_interval"] == 0:
            val_cer = validate(model, val_loader, vocab, device, writer, epoch)
            ckpt_mgr.save(epoch=epoch, val_cer=val_cer)

            # Log epoch summary
            tlogger.log_epoch_end(
                epoch=epoch,
                train_loss=avg_epoch_loss,
                val_cer=val_cer,
                samples_processed=epoch_samples,
                extra_metrics={"total_epochs": tcfg["epochs"]},
            )

    writer.close()
    tlogger.log_training_complete(
        best_cer=ckpt_mgr.best_cer,
    )
    print(f"\nDisk usage:\n{ckpt_mgr.disk_usage()}")


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ctc_config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Resume from: current.pt / previous.pt / best.pt")
    args = parser.parse_args()
    train(args.config, args.resume)
