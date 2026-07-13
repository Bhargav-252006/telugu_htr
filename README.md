# Telugu Handwritten Text Recognition (HTR)

A research-grade system for recognising handwritten Telugu words using a
**CNN + Transformer autoregressive decoder** with a **data-driven Telugu
script constraint** that is built directly from the training labels —
never from hard-coded linguistic rules.

---

## Research Contribution

| Approach | Prior Work? | This Project |
|---|---|---|
| CNN + BiLSTM + CTC | Yes | Baseline benchmark |
| CNN + Transformer Encoder + Decoder (AR) | Yes | SOTA architecture baseline |
| + Data-derived Telugu constraints | Partially | Rigorous empirical evaluation |

**Core objective**: Empirically evaluate whether a data-derived, Telugu-script-aware autoregressive decoder outperforms standard CTC and plain autoregressive baselines on the IIIT-HW-Telugu benchmark, particularly on compound/ligature-heavy words (words containing Virama ్).

## Project Structure

```
major/
├── src/
│   ├── vocab.py                # Telugu vocab + data-driven transition matrix
│   ├── transforms.py           # Image preprocessing + augmentation
│   ├── dataset.py              # IIIT-HW-Telugu dataset loader
│   ├── checkpoint_manager.py   # Rolling 2-slot checkpoint system
│   ├── train_ctc.py            # CTC baseline training
│   ├── train_ar.py             # Autoregressive model training
│   ├── evaluate.py             # CER / WER + virama breakdown + error analysis
│   ├── models/
│   │   ├── cnn_encoder.py      # ResNet-18 encoder (stride-patched for HTR, supports S=64/128)
│   │   ├── ctc_model.py        # CNN + BiLSTM + CTC
│   │   └── ar_model.py         # CNN + Transformer Encoder + Decoder (greedy + beam)
│   └── decoding/
│       └── telugu_mask.py      # Data-driven Telugu constraint mask
├── configs/
│   ├── ctc_config.yaml         # CTC training hyperparameters
│   └── ar_config.yaml          # AR training hyperparameters
├── data/
│   └── raw/
│       ├── train/              # Train images + labels.txt
│       ├── val/                # Val images + labels.txt
│       └── test/               # Test images + labels.txt
├── checkpoints/                # Saved model weights (rolling 2-slot)
├── logs/                       # TensorBoard logs
├── notebooks/                  # Exploration + results notebooks
├── requirements.txt
└── README.md
```

---

## Dataset

**IIIT-HW-Telugu** — [CVIT IIIT Hyderabad](http://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data)

| Split | Images |
|---|---|
| Train | 88,534 |
| Val | 19,980 |
| Test | 17,899 |
| **Total** | **126,413** |

**Annotation file format** (`labels.txt`, one line per sample):
```
word_00001.png  కాలం
word_00002.png  పూజ
```

Place data under:
```
data/raw/train/   ← images + labels.txt
data/raw/val/     ← images + labels.txt
data/raw/test/    ← images + labels.txt
```

---

## Architecture

### Shared Visual Encoder — ResNet-18 (stride-patched)

Standard ResNet-18 downsamples both height and width equally, producing
sequence length S = 32 — too short for CTC on Telugu words with 10–15
characters. This project patches the `maxpool` layer and `layer3/4` to 
preserve width resolution, giving a configurable sequence length (default S = 64, 
ablation S = 128).

```
Input  [B, 1, 64, 512]
  → Channel Replication (1→3)  preserve ImageNet pretraining
  → ResNet-18 backbone         stride-patched to preserve width
  → feature map [B, 512, 1, 64 or 128]
  → squeeze height, Conv1D(512→256)
  → Positional encoding
  → Encoder memory [B, 64/128, 256]
```

### Autoregressive Model (AR)
The AR model follows a state-of-the-art TrOCR-style architecture:
1. **Transformer Encoder (2 layers):** Adds global visual context to the CNN features before decoding.
2. **Transformer Decoder (4 layers):** Predicts character by character, attending to the encoder memory.

### CTC Baseline

```
Encoder memory [B, 64, 256]
  → 2-layer BiLSTM (hidden=256, bidirectional)
  → Linear(512 → vocab_size)
  → CTC loss / greedy decode
```

### Telugu-Aware Constraint Mask

Built **purely from observed label bigrams** in the training set —
no hard-coded linguistic rules.

Algorithm:
1. Scan every training label → extract every `(prev_char, next_char)` pair
2. `valid_next[prev][next] = True` only if that pair was seen in training data
3. During AR decoding: apply a soft penalty to logits of blocked next tokens before argmax
4. Run `validate_against_split(val_ann)` before training → violation rate must be ~0%

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| CNN backbone | ResNet-18 pretrained | Stronger features, stable training |
| Sequence model | CNN + Transformer hybrid | Better than pure ViT on limited HTR data |
| Stride fix | layer3/layer4 → (1,1) | S=64/128 instead of S=32; safe CTC margin |
| Script rules | Data-driven from labels | Never blocks a real training transition |
| Evaluation Metric | NFC Normalized CER | Ensures visually identical representations are scored fairly |
| Baselines | CTC + plain AR | Cleanly isolates novelty |
| Checkpoints | Unbiased Unconstrained CER | Ensures objective model selection |

---

## Checkpoint System

Exactly **3 files on disk at all times**, no matter how many epochs run:

```
checkpoints/<run>/
    best.pt        ← best val CER ever
    current.pt     ← most recent completed epoch
    previous.pt    ← epoch before that
```

Algorithm each epoch end:
1. `current.pt` → renamed to `previous.pt`
2. New state saved as `current.pt`
3. If val CER improved → `current.pt` copied to `best.pt`

Resume from any point:
```bash
python -m src.train_ctc --config configs/ctc_config.yaml --resume checkpoints/ctc/current.pt
python -m src.train_ar  --config configs/ar_config.yaml  --resume checkpoints/ar/current.pt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step-by-Step: How to Run

### Step 1 — Build vocabulary from training data

```bash
python -m src.vocab \
    data/raw/train/labels.txt \
    checkpoints/vocab.pkl \
    data/raw/val/labels.txt
```

This will:
- Scan all training labels → collect observed Telugu characters
- Build character vocabulary (expect ~80–100 tokens)
- Build **data-driven transition validity matrix** from observed bigrams
- Print a per-category transition audit table
- Validate the matrix against the val split → must show ~0% violation rate
- Save vocab to `checkpoints/vocab.pkl`

**Read the audit output before training.** If violation rate > 0.1%, the
constraint matrix will hurt AR decoding on that fraction of samples.

---

### Step 2 — Train CTC baseline (~1 hour on RTX 3090 Ti)

```bash
python -m src.train_ctc --config configs/ctc_config.yaml
```

Monitor live:
```bash
tensorboard --logdir logs/ctc
```

Logs per epoch: `train/loss`, `train/lr`, `val/loss`, `val/CER`,
`val/WER`, `val/avg_pred_len`.

---

### Step 3 — Train autoregressive model (~3 hours on RTX 3090 Ti)

```bash
python -m src.train_ar --config configs/ar_config.yaml
```

Resume after any interruption:
```bash
python -m src.train_ar --config configs/ar_config.yaml --resume checkpoints/ar/current.pt
```

---

### Step 4 — Run the 4-row ablation on test set

```bash
# Run A — CTC baseline
python -m src.evaluate \
    --model_type ctc \
    --checkpoint checkpoints/ctc/best.pt \
    --config configs/ctc_config.yaml \
    --split test

# Run B — AR decoder, no Telugu constraint
python -m src.evaluate \
    --model_type ar \
    --checkpoint checkpoints/ar/best.pt \
    --config configs/ar_config.yaml \
    --split test --no_constrain

# Run C — AR decoder + Telugu constraint (greedy)
python -m src.evaluate \
    --model_type ar \
    --checkpoint checkpoints/ar/best.pt \
    --config configs/ar_config.yaml \
    --split test

# Run D — AR decoder + Telugu constraint + beam search
python -m src.evaluate \
    --model_type ar \
    --checkpoint checkpoints/ar/best.pt \
    --config configs/ar_config.yaml \
    --split test --beam --beam_size 5
```

---

## Ablation Table (expected format)

| Model | Telugu Constraint | Beam | CER | WER | Compound CER | Simple CER |
|---|---|---|---|---|---|---|
| CTC Baseline | — | greedy | — | — | — | — |
| AR Decoder | ✗ | greedy | — | — | — | — |
| AR + Constraint | ✓ | greedy | — | — | — | — |
| AR + Constraint | ✓ | beam=5 | — | — | — | — |

Primary metric: **CER** (Character Error Rate).
Secondary: **WER**, compound CER (Virama words), simple CER.

---

## Hyperparameters

| Parameter | CTC | AR |
|---|---|---|
| Image H × W | 64 × 512 | 64 × 512 |
| Batch size | 64 | 64 |
| Optimizer | AdamW | AdamW |
| Learning rate | 1e-3 | 3e-4 |
| LR schedule | OneCycleLR | Warmup + Cosine |
| Warmup steps | — | 4000 |
| Weight decay | 1e-4 | 1e-4 |
| Label smoothing | — | 0.1 |
| Gradient clip | 5.0 | 5.0 |
| Mixed precision | fp16 | fp16 |
| Max epochs | 30 | 50 |
| Encoder layers | — | 2 |
| Decoder layers | — | 4 |
| Attention heads | — | 8 |

---

## Hardware

| Component | Spec |
|---|---|
| GPU | RTX 3090 Ti (24 GB VRAM) |
| System RAM | 4 GB |

> **Note on system RAM**: `num_workers` is set to 2 (not 4) in both
> configs to avoid RAM pressure with 4 GB system memory.
> Drop to `num_workers: 0` if you still see OOM errors during data loading.

Estimated training times:

| Run | Duration |
|---|---|
| CTC (30 epochs) | ~1 hour |
| AR (50 epochs) | ~3 hours |
| Full ablation eval | ~30–40 min |
| **Total** | **~4.5 hours** |

---

## File Reference

| File | Purpose |
|---|---|
| `src/vocab.py` | Telugu Unicode vocab, data-driven transition matrix, audit + validation |
| `src/transforms.py` | Grayscale, resize H=64, pad W=512, augmentation |
| `src/dataset.py` | Dataset class, `build_dataloader()` factory |
| `src/checkpoint_manager.py` | Rolling 2-slot: best / current / previous |
| `src/models/cnn_encoder.py` | ResNet-18, stride patch, positional encoding |
| `src/models/ctc_model.py` | CTC model, greedy decode |
| `src/models/ar_model.py` | AR model, greedy decode, beam search |
| `src/decoding/telugu_mask.py` | Constraint mask, vectorised apply, stats |
| `src/train_ctc.py` | CTC training loop |
| `src/train_ar.py` | AR training loop |
| `src/evaluate.py` | CER, WER, virama breakdown, confusion matrix, ablation table |
| `configs/ctc_config.yaml` | CTC hyperparameters |
| `configs/ar_config.yaml` | AR hyperparameters |
