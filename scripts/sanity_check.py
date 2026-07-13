"""
scripts/sanity_check.py

Pre-training sanity test suite. Run this BEFORE starting any training.

Tests (in order):
  1. Encoder shape          — verify S=64, CTC safety margin
  2. Vocabulary round-trip  — encode→decode on 100 random labels, UNK count
  3. Unicode normalisation  — detect inconsistent Unicode codepoints in labels
  4. Mask violation         — how many val/test transitions the constraint blocks
  5. Data loading           — image shape, label shape, NaN/Inf check
  6. CTC forward pass       — loss is finite, output shape is correct
  7. AR forward pass        — loss is finite, output shape is correct

Usage:
    python scripts/sanity_check.py \
        --train_ann   data/raw/train/labels.txt \
        --val_ann     data/raw/val/labels.txt \
        --test_ann    data/raw/test/labels.txt \
        --train_root  data/raw/train \
        --val_root    data/raw/val \
        --vocab_path  checkpoints/vocab.pkl

All tests print PASS or FAIL with a reason.
A summary table is printed at the end.
"""

from __future__ import annotations
import argparse
import os
import sys
import random
import unicodedata
from typing import List, Tuple

import torch

# ── make sure src/ is importable ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════
# Result tracker
# ═══════════════════════════════════════════════════════════════════

class Results:
    def __init__(self):
        self._rows: List[Tuple[str, bool, str]] = []

    def record(self, name: str, passed: bool, note: str = ""):
        self._rows.append((name, passed, note))
        tag = "  ✅ PASS" if passed else "  ❌ FAIL"
        msg = f"  {note}" if note else ""
        print(f"{tag}  {name}{msg}")

    def summary(self):
        passed = sum(1 for _, ok, _ in self._rows if ok)
        total  = len(self._rows)
        print(f"\n{'═'*62}")
        print(f"  SANITY CHECK SUMMARY: {passed}/{total} passed")
        print(f"{'─'*62}")
        for name, ok, note in self._rows:
            tag = "✅" if ok else "❌"
            print(f"  {tag}  {name}")
            if not ok and note:
                print(f"       → {note}")
        print(f"{'═'*62}")
        if passed == total:
            print("  All checks passed. Safe to start training.\n")
        else:
            print("  Fix the failing checks before training.\n")
        return passed == total


results = Results()


# ═══════════════════════════════════════════════════════════════════
# Test 1: Encoder shape
# ═══════════════════════════════════════════════════════════════════

def test_encoder_shape(max_label_len: int = 32):
    print("\n── Test 1: Encoder output shape ───────────────────────────")
    try:
        from src.models.cnn_encoder import ResNetEncoder

        enc   = ResNetEncoder(d_model=256, pretrained=False)
        dummy = torch.zeros(1, 1, 64, 512)

        with torch.no_grad():
            out = enc(dummy)

        B, S, D = out.shape
        print(f"  Input  : {list(dummy.shape)}")
        print(f"  Output : {list(out.shape)}   (B={B}, S={S}, D={D})")

        # CTC safety: need S >= 2*L - 1.  For L=max_label_len: S >= 2*32-1 = 63
        ctc_min = 2 * max_label_len - 1
        if S >= ctc_min and D == 256:
            results.record(
                "Encoder shape",
                True,
                f"S={S} >= CTC minimum {ctc_min}, D={D}"
            )
        else:
            results.record(
                "Encoder shape",
                False,
                f"S={S} < CTC minimum {ctc_min} OR D={D}≠256. "
                f"CTC will fail on long Telugu words."
            )

        # Also test different width to make sure it scales
        for W in [256, 384]:
            d2 = torch.zeros(1, 1, 64, W)
            with torch.no_grad():
                o2 = enc(d2)
            print(f"  Width={W:4d} → S={o2.shape[1]}")

    except Exception as e:
        results.record("Encoder shape", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# Test 2: Vocabulary round-trip
# ═══════════════════════════════════════════════════════════════════

def test_vocab_roundtrip(vocab, train_ann: str, n: int = 100):
    print("\n── Test 2: Vocabulary encode→decode round-trip ─────────────")
    try:
        labels = _sample_labels(train_ann, n)
        if not labels:
            results.record("Vocab round-trip", False, "No labels loaded from annotation file")
            return

        n_unk_total    = 0
        n_mismatch     = 0
        mismatch_examples = []

        for label in labels:
            ids  = vocab.encode(label)
            back = vocab.decode(ids, strip_special=True)

            n_unk_total += ids.count(vocab.unk_id)

            if back != label:
                n_mismatch += 1
                if len(mismatch_examples) < 3:
                    mismatch_examples.append((label, back))

        unk_rate      = n_unk_total / max(sum(len(l) for l in labels), 1)
        mismatch_rate = n_mismatch / len(labels)

        print(f"  Tested {len(labels)} labels")
        print(f"  UNK tokens       : {n_unk_total}  (rate={unk_rate*100:.2f}%)")
        print(f"  Round-trip errors: {n_mismatch}  (rate={mismatch_rate*100:.1f}%)")

        if mismatch_examples:
            print(f"  Mismatch examples:")
            for orig, decoded in mismatch_examples:
                print(f"    original : '{orig}'")
                print(f"    decoded  : '{decoded}'")

        if unk_rate < 0.01 and mismatch_rate == 0:
            results.record("Vocab round-trip", True,
                           f"0 mismatches, UNK rate={unk_rate*100:.2f}%")
        elif mismatch_rate > 0:
            results.record("Vocab round-trip", False,
                           f"{n_mismatch} labels don't round-trip. Check Unicode normalisation.")
        else:
            results.record("Vocab round-trip", False,
                           f"High UNK rate {unk_rate*100:.2f}%. Vocab may be incomplete.")

    except Exception as e:
        results.record("Vocab round-trip", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# Test 3: Unicode normalisation consistency
# ═══════════════════════════════════════════════════════════════════

def test_unicode_normalisation(train_ann: str, val_ann: str, n: int = 200):
    print("\n── Test 3: Unicode normalisation consistency ────────────────")
    try:
        train_labels = _sample_labels(train_ann, n)
        val_labels   = _sample_labels(val_ann, n)

        def nfc_issues(labels: List[str]) -> int:
            count = 0
            for lab in labels:
                nfc = unicodedata.normalize("NFC", lab)
                if nfc != lab:
                    count += 1
            return count

        train_issues = nfc_issues(train_labels)
        val_issues   = nfc_issues(val_labels)

        print(f"  Non-NFC labels in train sample : {train_issues}/{len(train_labels)}")
        print(f"  Non-NFC labels in val sample   : {val_issues}/{len(val_labels)}")

        if train_issues == 0 and val_issues == 0:
            results.record("Unicode normalisation", True,
                           "All sampled labels are NFC-normalised")
        else:
            results.record("Unicode normalisation", False,
                           f"Found non-NFC labels: train={train_issues}, val={val_issues}. "
                           f"Add unicodedata.normalize('NFC', label) in dataset.py._load_annotations()")

    except Exception as e:
        results.record("Unicode normalisation", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# Test 4: Mask violation on val and test
# ═══════════════════════════════════════════════════════════════════

def test_mask_violation(vocab, val_ann: str, test_ann: str):
    print("\n── Test 4: Transition mask violation check ─────────────────")
    if vocab._matrix_source == "permissive_default":
        results.record("Mask violation (val)",  False,
                       "Matrix not built yet — run src.vocab first")
        results.record("Mask violation (test)", False,
                       "Matrix not built yet — run src.vocab first")
        return

    for ann_path, split_name in [(val_ann, "val"), (test_ann, "test")]:
        if not os.path.exists(ann_path):
            results.record(f"Mask violation ({split_name})", False,
                           f"File not found: {ann_path}")
            continue
        try:
            stats = vocab.validate_against_split(ann_path, split_name=split_name)
            vrate = stats["violation_rate"]

            if vrate == 0.0:
                results.record(f"Mask violation ({split_name})", True,
                               f"0 blocked transitions on {split_name}")
            elif vrate < 0.001:
                results.record(f"Mask violation ({split_name})", True,
                               f"Very low violation rate {vrate*100:.4f}% on {split_name} — acceptable")
            else:
                results.record(f"Mask violation ({split_name})", False,
                               f"Violation rate {vrate*100:.3f}% on {split_name}. "
                               f"Constraint will hurt decoding on these samples.")
        except Exception as e:
            results.record(f"Mask violation ({split_name})", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# Test 5: Data loading
# ═══════════════════════════════════════════════════════════════════

def test_data_loading(vocab, train_ann, train_root, val_ann, val_root):
    print("\n── Test 5: Data loading (1 batch from train and val) ───────")
    try:
        from src.dataset import build_dataloader

        for ann, root, split, sos_eos in [
            (train_ann, train_root, "train", False),
            (val_ann,   val_root,  "val",   False),
        ]:
            if not os.path.exists(ann):
                results.record(f"Data loading ({split})", False, f"Not found: {ann}")
                continue

            loader = build_dataloader(
                ann, root, vocab,
                split         = split,
                batch_size    = 4,
                num_workers   = 0,
                max_label_len = 32,
                add_sos_eos   = sos_eos,
            )

            try:
                images, labels, lengths, widths = next(iter(loader))
            except StopIteration:
                results.record(f"Data loading ({split})", False, "DataLoader returned nothing")
                continue

            has_nan = torch.isnan(images).any().item()
            has_inf = torch.isinf(images).any().item()

            print(f"  [{split}] images={list(images.shape)}  "
                  f"labels={list(labels.shape)}  "
                  f"lengths={lengths.tolist()}  "
                  f"NaN={has_nan}  Inf={has_inf}")

            if (images.shape[1] == 1 and images.shape[2] == 64
                    and not has_nan and not has_inf
                    and labels.shape[0] == 4):
                results.record(f"Data loading ({split})", True,
                               f"images={list(images.shape)}, labels={list(labels.shape)}")
            else:
                results.record(f"Data loading ({split})", False,
                               f"Unexpected shape or NaN/Inf. images={list(images.shape)}")

    except Exception as e:
        results.record("Data loading", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# Test 6: CTC forward pass
# ═══════════════════════════════════════════════════════════════════

def test_ctc_forward(vocab_size: int):
    print("\n── Test 6: CTC model forward pass ──────────────────────────")
    try:
        from src.models.ctc_model import CTCModel

        model  = CTCModel(vocab_size=vocab_size, pretrained=False)
        imgs   = torch.randn(2, 1, 64, 512)
        labels = torch.randint(4, vocab_size, (2, 10))
        llens  = torch.tensor([10, 9])

        loss = model.compute_loss(imgs, labels, llens)
        preds = model.greedy_decode(imgs)

        is_finite = torch.isfinite(loss).item()
        print(f"  Loss={loss.item():.4f}  finite={is_finite}")
        print(f"  Greedy decode lengths: {[len(p) for p in preds]}")

        if is_finite and loss.item() > 0:
            results.record("CTC forward pass", True,
                           f"loss={loss.item():.4f}")
        else:
            results.record("CTC forward pass", False,
                           f"Loss is {'inf/nan' if not is_finite else 'zero or negative'}")

    except Exception as e:
        results.record("CTC forward pass", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# Test 7: AR forward pass
# ═══════════════════════════════════════════════════════════════════

def test_ar_forward(vocab, vocab_size: int):
    print("\n── Test 7: AR model forward pass ───────────────────────────")
    try:
        from src.models.ar_model import ARModel

        model  = ARModel(
            vocab_size=vocab_size,
            sos_id=vocab.sos_id,
            eos_id=vocab.eos_id,
            pretrained=False,
        )
        imgs   = torch.randn(2, 1, 64, 512)

        # labels: SOS + 8 chars + EOS, padded to 12
        labels = torch.zeros(2, 12, dtype=torch.long)
        labels[:, 0] = vocab.sos_id
        labels[:, 1:9] = torch.randint(4, vocab_size, (2, 8))
        labels[:, 9]  = vocab.eos_id
        llens  = torch.tensor([10, 10])

        loss, ce_loss, ctc_loss  = model.compute_loss(imgs, labels, llens)
        preds = model.greedy_decode(imgs, max_len=12,
                                    vocab=vocab, constrain=True)

        is_finite = torch.isfinite(loss).item()
        print(f"  Loss={loss.item():.4f} (CE={ce_loss.item():.4f}, CTC={ctc_loss.item():.4f})  finite={is_finite}")
        print(f"  Greedy decode lengths: {[len(p) for p in preds]}")

        if is_finite and loss.item() > 0:
            results.record("AR forward pass", True,
                           f"loss={loss.item():.4f}")
        else:
            results.record("AR forward pass", False,
                           f"Loss is {'inf/nan' if not is_finite else 'zero or negative'}")

    except Exception as e:
        results.record("AR forward pass", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _sample_labels(ann_path: str, n: int) -> List[str]:
    labels = []
    if not os.path.exists(ann_path):
        return labels
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                labels.append(parts[1])
    return random.sample(labels, min(n, len(labels)))


def _load_or_build_vocab(vocab_path, train_ann, val_ann):
    from src.vocab import TeluguVocab
    if os.path.exists(vocab_path):
        return TeluguVocab.load(vocab_path)
    print(f"[sanity] vocab not found at {vocab_path} — building from {train_ann}")
    vocab = TeluguVocab.from_annotation_files([train_ann], build_matrix=True)
    vocab.validate_against_split(val_ann, split_name="val")
    vocab.save(vocab_path)
    return vocab


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pre-training sanity checks")
    parser.add_argument("--train_ann",  default="data/raw/train/labels.txt")
    parser.add_argument("--val_ann",    default="data/raw/val/labels.txt")
    parser.add_argument("--test_ann",   default="data/raw/test/labels.txt")
    parser.add_argument("--train_root", default="data/raw/train")
    parser.add_argument("--val_root",   default="data/raw/val")
    parser.add_argument("--vocab_path", default="checkpoints/vocab.pkl")
    parser.add_argument("--max_label_len", type=int, default=32)
    args = parser.parse_args()

    print("═" * 62)
    print("  Telugu HTR — Pre-training Sanity Check")
    print("═" * 62)

    # ── Load vocab ───────────────────────────────────────────────
    print("\n[Loading vocabulary ...]")
    try:
        vocab = _load_or_build_vocab(args.vocab_path, args.train_ann, args.val_ann)
        print(f"  Vocab size: {len(vocab)}  "
              f"matrix_source: {vocab._matrix_source}")
    except Exception as e:
        print(f"  ERROR loading vocab: {e}")
        print("  Cannot continue without a vocab. Exiting.")
        sys.exit(1)

    vocab_size = len(vocab)

    # ── Run all tests ─────────────────────────────────────────────
    test_encoder_shape(args.max_label_len)
    test_vocab_roundtrip(vocab, args.train_ann, n=100)
    test_unicode_normalisation(args.train_ann, args.val_ann, n=200)
    test_mask_violation(vocab, args.val_ann, args.test_ann)
    test_data_loading(vocab, args.train_ann, args.train_root,
                             args.val_ann,   args.val_root)
    test_ctc_forward(vocab_size)
    test_ar_forward(vocab, vocab_size)

    # ── Summary ───────────────────────────────────────────────────
    all_passed = results.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
