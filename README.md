# Telugu Handwritten Text Recognition (HTR)

A research-grade system for recognising handwritten Telugu words using a
**CNN + Transformer autoregressive decoder (AR v2)** with a **data-driven Telugu
script constraint** built directly from training labels — never from hard-coded linguistic rules.

> **Paper submitted to:** International Journal on Document Analysis and Recognition (IJDAR), Springer Nature  
> **Status:** Under Review

---

## Results (IIIT-HW-Telugu Test Set, N = 17,910)

| Model | CER (%) | WER (%) | 95% CI (CER) | Speed (ms/img) |
|---|---|---|---|---|
| CTC Baseline (BiLSTM) | 3.91 | 24.80 | [3.79, 4.04] | 0.3 |
| AR v1 — no CTC aux loss (ablation) | 5.43 | 32.37 | — | 1.7 |
| AR v1 — unconstrained (ablation) | 4.89 | 29.83 | — | 1.7 |
| **AR v2 — greedy, unconstrained** | **3.67** | **23.33** | [3.55, 3.79] | 1.4 |
| AR v2 — greedy, constrained | 3.74 | 23.32 | [3.60, 3.88] | 1.8 |
| **AR v2 — beam=5, unconstrained ✅ BEST** | **3.66** | **23.34** | [3.54, 3.79] | 77.0 |
| AR v2 — beam=5, constrained | 3.67 | 23.28 | [3.55, 3.79] | 82.5 |

> **Best model: `checkpoints/ar_v2/best.pt`** — AR v2 with beam=5, unconstrained decoding  
> 3.66% CER = **20% relative reduction** from prior best (Dutta et al. 2018: 4.58%)

---

## Project Structure

```
major/
├── src/
│   ├── vocab.py                # Telugu vocab + data-driven transition matrix
│   ├── transforms.py           # Image preprocessing + augmentation
│   ├── dataset.py              # IIIT-HW-Telugu dataset loader
│   ├── checkpoint_manager.py   # Rolling 3-slot checkpoint system
│   ├── train_ctc.py            # CTC baseline training
│   ├── train_ar.py             # AR model training (used for both ar and ar_v2)
│   ├── evaluate.py             # CER / WER + virama breakdown + error analysis
│   ├── generate_paper_figures.py
│   ├── training_logger.py
│   ├── models/
│   │   ├── cnn_encoder.py      # ResNet-18 encoder (stride-patched for HTR)
│   │   ├── ctc_model.py        # CNN + BiLSTM + CTC
│   │   └── ar_model.py         # CNN + Transformer Encoder + Decoder (greedy + beam)
│   └── decoding/
│       └── telugu_mask.py      # Data-driven Telugu constraint mask
├── configs/
│   ├── ctc_config.yaml         # CTC hyperparameters
│   ├── ar_config.yaml          # AR v1 hyperparameters (ablation run)
│   ├── ar_no_ctc_config.yaml   # AR without CTC aux loss (ablation run)
│   └── ar_v2_config.yaml       # ⭐ AR v2 — final model config
├── results/
│   ├── ablation_results.json   # AR v1 ablation numbers
│   ├── paper_figures/
│   │   └── all_results.json    # ⭐ Final results (AR v2 + CTC)
│   └── figures/                # Training curves, confusion matrix, ablation bar plots
├── paper/
│   ├── main_ijdar.tex          # IJDAR journal paper (LaTeX)
│   ├── main.tex                # IEEE conference format (backup)
│   ├── Major_Project_Documentation.docx  # Phase-1 project documentation
│   └── figures/                # Paper figures
├── checkpoints/                # Model weights (see note below)
├── logs/                       # TensorBoard logs + training log
├── notebooks/
├── requirements.txt
└── README.md
```

> **Note on checkpoints:** Model weights are 160–360 MB each and exceed GitHub's file limit.  
> 📥 **Download from Google Drive:** https://drive.google.com/drive/folders/1_cAdVYOuqEzvmORYGvFKmJvAHpbKGo5N?usp=sharing  
> Required files: `checkpoints/ar_v2/best.pt`, `checkpoints/ctc/best.pt`, `checkpoints/vocab.pkl`  
> Place them under the `checkpoints/` folder before running evaluation.


---

## Dataset

**IIIT-HW-Telugu** — [CVIT IIIT Hyderabad](http://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data)

| Split | Images |
|---|---|
| Train | 80,693 |
| Val | 20,048 |
| Test | 17,910 |
| **Total** | **118,651** |

Annotation file format (`labels.txt`, one line per sample):
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
characters. This project patches `layer3` and `layer4` strides to
preserve width resolution, giving S = 64.

```
Input  [B, 1, 64, 512]
  → Channel Replication (1→3)         preserve ImageNet pretraining
  → ResNet-18 backbone                stride-patched: layers 3/4 stride (2,1)
  → feature map [B, 512, 4, 64]
  → squeeze height via AvgPool
  → Conv1x1(512 → d_model)
  → Positional encoding (sinusoidal)
  → Encoder memory [B, 64, d_model]
```

### AR v2 — Final Model

Trained with `configs/ar_v2_config.yaml` using `src/train_ar.py`.

| Parameter | Value |
|---|---|
| d_model | 384 |
| Encoder layers | 3 (Transformer) |
| Decoder layers | 6 (Transformer) |
| Attention heads | 8 |
| d_ff | 1,536 |
| Dropout | 0.15 |
| Label smoothing | 0.05 |
| CTC aux weight (λ) | 0.3 |
| Batch size | 64 |
| LR | 3e-4 (warmup 4000 steps + cosine) |
| Epochs | 80 |
| Mixed precision | FP16 |

Two critical architectural fixes vs AR v1:
- **Embedding scaling** by √d_model — without this, val CER stalls at ~4.89%
- **Weight tying** between input embedding and output projection

### CTC Baseline

```
Encoder memory [B, 64, 256]
  → 2-layer BiLSTM (hidden=256, bidirectional)
  → Linear(512 → vocab_size)
  → CTC loss / greedy decode
```

### Telugu-Aware Constraint Mask

Built **purely from observed label bigrams** in the training set — no hard-coded rules.

1. Scan every training label → extract every `(prev_char, next_char)` pair
2. `valid_next[prev][next] = True` only if that pair was seen in training data
3. During AR decoding: subtract penalty δ=10.0 from logits of blocked tokens
4. Run `validate_against_split(val_ann)` → violation rate must be ~0%

---

## Setup

```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1 — Build vocabulary

```bash
python -m src.vocab \
    data/raw/train/labels.txt \
    checkpoints/vocab.pkl \
    data/raw/val/labels.txt
```

### Step 2 — Train CTC baseline (~3.5 hours on RTX 3090 Ti)

```bash
python -m src.train_ctc --config configs/ctc_config.yaml
```

### Step 3 — Train AR v2 (final model) (~5.4 hours on RTX 3090 Ti)

```bash
python -m src.train_ar --config configs/ar_v2_config.yaml
```

Resume after interruption:
```bash
python -m src.train_ar --config configs/ar_v2_config.yaml --resume checkpoints/ar_v2/current.pt
```

### Step 4 — Evaluate (reproduce paper results)

```bash
# CTC Baseline
python -m src.evaluate \
    --model_type ctc \
    --checkpoint checkpoints/ctc/best.pt \
    --config configs/ctc_config.yaml \
    --split test

# AR v2 — greedy, unconstrained
python -m src.evaluate \
    --model_type ar \
    --checkpoint checkpoints/ar_v2/best.pt \
    --config configs/ar_v2_config.yaml \
    --split test --no_constrain

# AR v2 — greedy, constrained
python -m src.evaluate \
    --model_type ar \
    --checkpoint checkpoints/ar_v2/best.pt \
    --config configs/ar_v2_config.yaml \
    --split test

# AR v2 — beam=5, unconstrained (BEST RESULT: 3.66% CER)
python -m src.evaluate \
    --model_type ar \
    --checkpoint checkpoints/ar_v2/best.pt \
    --config configs/ar_v2_config.yaml \
    --split test --beam --beam_size 5 --no_constrain

# AR v2 — beam=5, constrained
python -m src.evaluate \
    --model_type ar \
    --checkpoint checkpoints/ar_v2/best.pt \
    --config configs/ar_v2_config.yaml \
    --split test --beam --beam_size 5
```

---

## Checkpoint System

Exactly **3 files on disk at all times**:

```
checkpoints/<run>/
    best.pt        ← best val CER ever seen
    current.pt     ← most recent completed epoch
    previous.pt    ← epoch before that
```

---

## Hardware

| Component | Spec |
|---|---|
| GPU | RTX 3090 Ti (24 GB VRAM) |
| System RAM | 4 GB |

> **Note:** `num_workers` is set to 8 in `ar_v2_config.yaml`. Drop to `num_workers: 0` if you see OOM errors during data loading.

| Run | Duration |
|---|---|
| CTC (50 epochs) | ~3.5 hours |
| AR v2 (80 epochs) | ~5.4 hours |
| Full ablation eval | ~30–40 min |
| **Total** | **~9 hours** |

---

## File Reference

| File | Purpose |
|---|---|
| `src/vocab.py` | Telugu Unicode vocab, data-driven transition matrix, audit + validation |
| `src/transforms.py` | Grayscale, resize H=64, pad W=512, augmentation pipeline |
| `src/dataset.py` | Dataset class, `build_dataloader()` factory |
| `src/checkpoint_manager.py` | Rolling 3-slot: best / current / previous |
| `src/models/cnn_encoder.py` | ResNet-18, stride patch, positional encoding |
| `src/models/ctc_model.py` | CTC model, greedy decode |
| `src/models/ar_model.py` | AR model, greedy decode, beam search, weight tying |
| `src/decoding/telugu_mask.py` | Constraint mask, vectorised apply, stats |
| `src/train_ctc.py` | CTC training loop |
| `src/train_ar.py` | AR training loop (used for ar, ar_no_ctc, and ar_v2) |
| `src/evaluate.py` | CER, WER, virama breakdown, confusion matrix, ablation table |
| `configs/ctc_config.yaml` | CTC hyperparameters |
| `configs/ar_v2_config.yaml` | ⭐ AR v2 final model hyperparameters |
| `configs/ar_config.yaml` | AR v1 (ablation) hyperparameters |
| `configs/ar_no_ctc_config.yaml` | AR without CTC aux loss (ablation) |
| `results/paper_figures/all_results.json` | Final test-set numbers (AR v2 + CTC) |
| `results/ablation_results.json` | AR v1 ablation numbers |

---

## Authors

- **Sakilam Bhargav** (23211A67A7) — [23211a67a7@bvrit.ac.in](mailto:23211a67a7@bvrit.ac.in)
- **Thatha Nikitha** (23211A67B6)
- **Sabavat Vinod Nayak** (23211A67A6)
- **Dr. R. Venkata Ramana Chary** *(Guide)* — Professor, CSE (Data Science), BVRIT Narsapur
