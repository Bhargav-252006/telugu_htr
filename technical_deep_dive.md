# Telugu HTR — Complete Technical Deep-Dive

> Every number, shape, and formula here is taken directly from the actual source code.

---

## Table of Contents
1. [Problem Formulation](#1-problem-formulation)
2. [Dataset & Vocabulary](#2-dataset--vocabulary)
3. [Data Preprocessing & Augmentation](#3-data-preprocessing--augmentation)
4. [CNN Feature Encoder (ResNet-18)](#4-cnn-feature-encoder-resnet-18)
5. [Model A — CTC Baseline (BiLSTM + CTC)](#5-model-a--ctc-baseline-bilstm--ctc)
6. [Model B — Autoregressive Transformer](#6-model-b--autoregressive-transformer)
7. [Telugu Grammar Mask](#7-telugu-grammar-mask)
8. [Training Pipeline](#8-training-pipeline)
9. [Inference & Decoding](#9-inference--decoding)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Final Results](#11-final-results)
12. [Full Tensor Flow — End to End](#12-full-tensor-flow--end-to-end)

---

## 1. Problem Formulation

**Task:** Given a grayscale image of a single handwritten Telugu word, predict the Unicode character sequence.

**Formal definition:**
- Input: image $I \in \mathbb{R}^{1 \times 64 \times 512}$ (channels × height × width)
- Output: sequence $\mathbf{y} = (y_1, y_2, \ldots, y_L)$ where $y_i \in \mathcal{V}$, $|\mathcal{V}| = 91$

**Why hard:**
- Telugu has 56 base consonants + 16 vowels + diacritics + virama conjuncts = 87 unique characters in practice
- Virama (U+0C4D, "్") fuses two consonants into visually merged glyphs — the combined shape looks nothing like either standalone character
- Average word length: 8.7 characters with high structural density per glyph

---

## 2. Dataset & Vocabulary

### Dataset: IIIT-HW-Telugu

| Split | Samples |
|-------|---------|
| Train | 80,693 |
| Validation | 20,048 |
| **Test** | **17,910** |
| **Total** | **118,651** |

**Annotation format:**
```
train/word_00001.png  కాలం
train/word_00002.png  అభివృద్ధి
```

Labels are **NFC-normalised** (`unicodedata.normalize("NFC", label)`) at load time to ensure canonical Unicode composition.

---

### Vocabulary (`src/vocab.py`)

**Total vocab size: 91 tokens**

| Index | Token | Purpose |
|-------|-------|---------|
| 0 | `PAD` | Padding + CTC blank |
| 1 | `SOS` | Start-of-sequence (AR only) |
| 2 | `EOS` | End-of-sequence (AR only) |
| 3 | `UNK` | Unknown character |
| 4–90 | Telugu chars | Actual Unicode characters |

**Telugu character categories:**

| Category | Unicode Range | Count |
|----------|--------------|-------|
| Vowels (స్వరాలు) | U+0C05–U+0C14 | 14 |
| Consonants (హల్లులు) | U+0C15–U+0C39 + extras | ~39 |
| Vowel signs / Matras | U+0C3E–U+0C56 | 15 |
| Virama (హల్) | U+0C4D | 1 |
| Modifiers (అనుస్వారం, విసర్గ) | U+0C00, U+0C02, U+0C03 | 3 |
| Telugu digits | U+0C66–U+0C6F | 10 |

**Key methods:**
```python
vocab.encode("కాలం")  → [4, 17, 22, 8]       # char → int
vocab.decode([4,17,22,8])  → "కాలం"           # int → str
vocab.get_valid_next_tensor(prev_id, device)   # for grammar mask
```

---

### Data-Driven Validity Matrix

This is the backbone of the grammar mask. The vocab scans **every bigram** in the training set:

```
SOS → first_char
char[i] → char[i+1]
last_char → EOS
```

Result: a boolean matrix `_valid_next[V][V]` where `_valid_next[i][j] = True` means "character j is allowed to follow character i" — based purely on what was *actually observed* in training data, not hard-coded grammar rules.

**Hardcoded overrides applied after observation:**
- `UNK → anything` = **True** (safety fallback)
- `PAD → anything` = **False** (PAD is terminal)
- `EOS → anything` = **False** (EOS is terminal)
- `anything → PAD` = **False** (PAD can't be predicted)

---

## 3. Data Preprocessing & Augmentation

**`src/transforms.py` — Two transform pipelines:**

### ValTransform (deterministic, no augmentation)
```
PIL Image → grayscale (L mode)
         → resize: height=64, keep aspect ratio, crop if W>512
         → right-pad to W=512 with white (255)
         → to_tensor → [1, 64, 512] in [0,1]
         → Normalize(mean=[0.5], std=[0.5]) → [-1, 1]
Returns: (tensor [1,64,512], scaled_width: int)
```

### TrainTransform (stochastic augmentation pipeline)

Applied **in this exact order:**

| Step | Operation | Probability | Parameters |
|------|-----------|-------------|------------|
| 1 | Grayscale | always | PIL L-mode |
| 2 | **Elastic distortion** | p=0.3 | α=10.0, σ=3.0 via `scipy.gaussian_filter` + `cv2.remap` |
| 3 | **Morphological ops** | p=0.3 | random MinFilter(3) [thicken] or MaxFilter(3) [thin] |
| 4 | **Random rotation** | p=0.5 | uniform angle ∈ [−5°, +5°], fill=255 |
| 5 | **Random perspective** | p=0.3 | 5% corner jitter |
| 6 | Resize (keep aspect) | always | H=64, crop W if >512 |
| 7 | Pad | always | right-pad to W=512 |
| 8 | **ColorJitter** | always | brightness=0.3, contrast=0.3 |
| 9 | to_tensor → Normalize | always | [0,1] → [−1,1] |

> **CTC uses augmentation steps 1,4,6,7,9 only** (`use_elastic=false`)  
> **AR uses all steps** (`use_elastic=true`)

**Why elastic distortion?** It simulates natural pen-pressure variation and stroke waviness, making the model robust to different handwriting styles without needing more data.

---

### Collation & Batching (`src/dataset.py`)

The `collate_fn_pad` function assembles variable-length label sequences into a uniform batch:

```python
# Input: list of (img[1,64,512], width, label_ids, label_len)
# Output:
images  → [B, 1, 64, 512]    # all images same shape already
labels  → [B, T_max]          # zero-padded to longest label in batch
lengths → [B]                  # LongTensor of true label lengths
widths  → [B]                  # LongTensor of true image widths
```

**Important:** `add_sos_eos=False` for CTC (CTC doesn't use SOS/EOS) and `add_sos_eos=True` for AR.

---

## 4. CNN Feature Encoder (ResNet-18)

**File:** `src/models/cnn_encoder.py`  
**The same encoder is shared by both CTC and AR models.**

### Why a custom ResNet?

Standard ResNet-18 ends with `AdaptiveAvgPool2d(1,1)` → spatial output is `1×1`. We need the model to output an **ordered sequence** (left-to-right), preserving horizontal (width) information. So we surgically modify ResNet's stride pattern.

### Stride-Patching Surgery

```python
# Before patching (standard ResNet-18):
layer3[0].conv1.stride = (2, 2)    # collapses both H and W
layer4[0].conv1.stride = (2, 2)    # same

# After patching:
layer3[0].conv1.stride = (2, 1)    # collapses H only, preserves W
layer4[0].conv1.stride = (2, 1)    # same
# Downsample connections patched identically to keep residual shapes
```

This gives us **spatial feature maps that are tall-but-collapsed** — height shrinks aggressively, width is preserved at 1/8 the input width.

### Grayscale → RGB Bridge

ResNet-18 expects 3-channel input (pretrained on ImageNet RGB):
```python
x = x.repeat(1, 3, 1, 1)   # [B,1,H,W] → [B,3,H,W]
```
This is a standard trick — duplicate the single grayscale channel into all 3 colour channels.

### Projection Head

After the modified ResNet backbone:
```python
proj = nn.Sequential(
    nn.Conv2d(512, d_model, kernel_size=(2,1)),   # (H=2, W=1) conv
    nn.BatchNorm2d(d_model),
    nn.GELU(),
)
feat = proj(feature_map)            # [B, 256, 1, 64]
feat = feat.mean(dim=2)             # [B, 256, 64]  ← average remaining H
feat = feat.permute(0, 2, 1)        # [B, 64, 256]  ← (batch, sequence, features)
```

### Sinusoidal Positional Encoding

Standard transformer positional encoding added to the sequence dimension:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**max_len = 512** (more than enough for any input width)

### Complete Encoder Shape Flow

```
Input image:          [B,  1,  64, 512]
repeat to RGB:        [B,  3,  64, 512]
Conv1 + MaxPool:      [B, 64,  32, 256]
ResNet Layer1:        [B, 64,  32, 256]
ResNet Layer2:        [B,128,  16, 128]
ResNet Layer3*:       [B,256,   8,  64]   ← H halved, W preserved by patched stride
ResNet Layer4*:       [B,512,   4,  64]   ← H halved, W preserved
proj Conv2d(2,1):     [B,256,   1,  64]
mean(dim=2):          [B,256,      64]
permute(0,2,1):       [B, 64,     256]   ← final: 64 time steps × 256 features
+ pos encoding:       [B, 64,     256]   ← ENCODER OUTPUT
```

**The 64 time steps correspond to 64 equally-spaced horizontal slices of the word image.** Each "slice" is a 256-dim representation of one column of the word.

---

## 5. Model A — CTC Baseline (BiLSTM + CTC)

**File:** `src/models/ctc_model.py`  
**Parameters: ~14.1M** (as logged during training)

### Architecture

```
Encoder output:   [B, 64, 256]
                  ↓
BiLSTM (2 layers)
  input_size=256, hidden_size=256, bidirectional=True, batch_first=True
  each direction: 256 hidden units
  concatenated: 512 total
                  ↓
output:           [B, 64, 512]
                  ↓
Dropout(0.1) + Linear(512 → 91)
                  ↓
logits:           [B, 64, 91]
                  ↓
log_softmax(dim=2)
                  ↓
permute(1, 0, 2)
                  ↓
CTC input:        [64, B, 91]     ← CTCLoss format: [T, B, C]
```

### CTC Loss

$$\mathcal{L}_{CTC} = -\log P(\mathbf{y} | \mathbf{x}) = -\log \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})} P(\pi | \mathbf{x})$$

Where $\mathcal{B}$ is the "collapse" function that removes blanks and consecutive duplicates from any path $\pi$.

Key settings:
- `blank = 0` (PAD token doubles as CTC blank)
- `zero_infinity=True` (prevents inf/nan from pathological alignments)
- Input lengths: `input_widths // 8` (dynamic, per-image)
- Target lengths: `label_lengths` (from collate)

### Weight Initialisation

```python
# LSTM weights: orthogonal init (prevents vanishing/exploding gradients in RNNs)
nn.init.orthogonal_(lstm.weight_hh_lX)
nn.init.orthogonal_(lstm.weight_ih_lX)
nn.init.zeros_(lstm.bias_hh_lX)
nn.init.zeros_(lstm.bias_ih_lX)

# Output linear: Xavier uniform (variance-stable for classification heads)
nn.init.xavier_uniform_(ctc_head.weight)
nn.init.zeros_(ctc_head.bias)
```

### Greedy CTC Decoding

```python
# Step 1: argmax over vocabulary at each timestep
best_path = logits.argmax(dim=-1)   # [B, T]

# Step 2: collapse consecutive duplicates
# Step 3: remove blanks (id=0)
# e.g., [0,4,4,0,7,0,7,2] → [4,7,7,2] → [4,7,2]
```

**Output:** `List[List[int]]` — one predicted sequence per batch item.

---

## 6. Model B — Autoregressive Transformer

**File:** `src/models/ar_model.py`  
**Parameters: ~17.3M**

### Full Architecture

```
Encoder output:          [B, S=64, d=256]
                          ↓
TransformerEncoder
  num_layers=2, nhead=4, dim_ff=1024
  Pre-LN (norm_first=True), dropout=0.2
  (global self-attention over the 64 visual feature slices)
                          ↓
memory:                  [B, 64, 256]
                          ├──────────────────── AUX CTC HEAD
                          │                    Linear(256→91)
                          │                    → CTCLoss (λ=0.3)
                          │
Target tokens (shifted):  [B, T]     ← teacher forcing: input = y[:-1]
Token Embedding(91→256)
Decoder Positional Enc
                          ↓
TransformerDecoder
  num_layers=4, nhead=4, dim_ff=1024
  Pre-LN (norm_first=True), dropout=0.2
  Causal self-attention (future tokens masked)
  Cross-attention to memory (visual features)
                          ↓
decoder output:          [B, T, 256]
                          ↓
output_proj Linear(256→91)
                          ↓
CE logits:               [B, T, 91]
                          ↓
CrossEntropyLoss(label_smoothing=0.1)  (target = y[1:])
```

### Joint Loss Function

$$\mathcal{L}_{total} = (1 - \lambda) \cdot \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{CTC}$$

With $\lambda = 0.3$:
$$\mathcal{L}_{total} = 0.7 \cdot \mathcal{L}_{CE} + 0.3 \cdot \mathcal{L}_{CTC}$$

**Why CTC auxiliary loss?** The Transformer decoder can learn to "cheat" early in training — attending to random positions and getting lucky. The CTC head forces the encoder to produce strongly aligned, left-to-right features. This alignment signal bleeds into the encoder's representations, making cross-attention in the decoder converge faster and more stably. **Ablation proof: removing CTC increases CER from 4.89% to 5.43%.**

### Causal Mask (No Future Peeking)

```python
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
# causal_mask[i][j] = True means "position i cannot attend to position j"
# Upper triangle = future positions → all masked
```

### Teacher Forcing

During training:
```python
decoder_input = labels[:, :-1]    # [B, T-1]: SOS + all chars except last
target        = labels[:, 1:]     # [B, T-1]: all chars except SOS
```
The decoder sees the ground truth prefix at every step (not its own previous predictions). This is standard for training seq2seq models — faster convergence.

### Pre-LN vs Post-LN

```python
# We use Pre-LN (norm_first=True in PyTorch TransformerLayer):
# Pre-LN:  x → LayerNorm → SelfAttn → + x → LayerNorm → FFN → + x
# Post-LN: x → SelfAttn → + x → LayerNorm → FFN → + x → LayerNorm
```

Pre-LN is more stable for training from scratch — gradients don't blow up in early epochs.

### Label Smoothing (ε = 0.1)

Instead of hard one-hot targets:
$$q(k) = \begin{cases} 1 - \varepsilon & k = y^* \\ \varepsilon / (|\mathcal{V}| - 1) & k \neq y^* \end{cases}$$

Prevents over-confident predictions and improves generalisation.

---

## 7. Telugu Grammar Mask

**Files:** `src/vocab.py` + `src/decoding/telugu_mask.py`

### The Problem

During AR greedy/beam decoding, the model might predict character sequences that are **phonologically impossible in Telugu**. For example:
- Virama (్) followed by a vowel (e.g., అ) — **impossible** (virama must be followed by a consonant)
- Two consecutive virama characters — **impossible**

### Implementation

The validity matrix `_valid_next[V][V]` (bool) is pre-computed from training bigrams and stored in vocab.

At each decoder step $t$, given the last predicted token $c_{t-1}$:

```python
# Soft penalty approach (penalty=10.0, not -inf):
valid_row = vocab.get_valid_next_tensor(c_{t-1}, device)  # [V] bool
logits[~valid_row] -= 10.0     # subtract 10 from all invalid token logits
```

This is a **soft** constraint ($-10$ not $-\infty$) — theoretically the model can still pick a blocked token if it's *extremely* confident, but practically it never does because the remaining options will always score higher.

### Why Soft Penalty and Not Hard Masking?

Hard masking ($-\infty$) can cause NaN in softmax if ALL tokens in a category are blocked (edge case with rare characters). The $-10$ penalty is safe and practically equivalent.

### Stats Tracking

```python
# MaskStats dataclass:
total_steps   # decoder steps processed
masked_steps  # steps where ≥1 token was blocked  
tokens_masked # cumulative count of blocked tokens
fire_rate = masked_steps / total_steps × 100%
```

### Result in Practice

Grammar mask: **CER 4.89% → 4.99%** (slightly worse CER, marginally better WER 29.83%→29.71%). The model already learned most of these rules implicitly from training data — the mask catches edge cases but rarely fires.

---

## 8. Training Pipeline

### Optimizer

```python
optimizer = AdamW(
    params=model.parameters(),
    lr=3e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)
```

**Weight decay** = L2 regularisation on weights (not biases/norms). Prevents overfitting with 14-17M parameters on 80K samples.

### Learning Rate Schedule

```
Warmup: linear ramp from 0 → 3e-4 over 4000 steps
Then: CosineAnnealingLR → decays to 0 by epoch 50
```

This warmup is critical for Transformers — jumping straight to a high LR causes attention weights to diverge.

### Mixed Precision Training (FP16)

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    loss = model.compute_loss(images, labels, lengths, widths)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
scaler.step(optimizer)
scaler.update()
```

- **FP16** forward pass: saves ~50% GPU memory, ~2x speed on Tensor Cores
- **GradScaler**: scales loss up before backward (to avoid FP16 underflow), then unscales before optimizer step
- **Gradient clipping at 5.0**: prevents exploding gradients, essential for RNNs and early Transformer training

### Hardware

- **GPU:** NVIDIA RTX 3090 Ti (24 GB VRAM)
- **CTC training:** 3h 22m (50 epochs)
- **AR training:** ~2h 15m (resumed run, ~50 total epochs equivalent)

### Checkpoint Manager (Rolling 3-File Window)

```
checkpoints/ctc/
├── best.pt       ← model with lowest val CER ever seen
├── current.pt    ← last epoch's weights
└── previous.pt   ← second-to-last epoch (safety backup)
```

Validation is run every epoch. If `val_cer < best_val_cer`, overwrite `best.pt`.

---

## 9. Inference & Decoding

### CTC Greedy Decode

```
logits [T=64, B, 91]
→ argmax over dim=2        → best path [B, T]
→ collapse consecutive dups → [B, T']
→ remove blanks (id=0)     → List[List[int]] (variable length)
→ vocab.decode()           → List[str]
```

Time complexity: O(T × B) — very fast (~2.7 ms/sample)

### AR Greedy Decode

```python
# Initialise: decoder_input = [[SOS]] for each batch item
# Loop until all items emit EOS or max_len=36 reached:
#   1. Forward decoder with current sequence
#   2. Take logits at last position: [B, V]
#   3. Apply grammar mask (if enabled)
#   4. argmax → next_token: [B]
#   5. Append next_token to decoder_input
#   6. Mark finished items (emitted EOS)
```

Time complexity: O(L × T × d²) per batch — sequential, L steps. (~1.7 ms/sample due to short sequences)

### AR Beam Search (k=5)

For **each image** (no batch-level beam parallelism):

```python
# State: List of (score, token_sequence) — k=5 beams
# At each step:
#   For each active beam:
#     Forward decoder → logits [1, V]
#     Apply grammar mask
#     topk(5) → candidate next tokens
#     New beams = cross(active_beams, candidates) → k² candidates
#     Keep top-k by score
#     Move completed beams (EOS) to 'done' list

# Length penalty: score / len(seq)^0.6
# Final: pick highest-scoring beam from 'done'
```

Time complexity: O(k × L × T × d²) per image. (~327.5 ms/sample — 192× slower than greedy, no significant accuracy gain in our experiments)

---

## 10. Evaluation Metrics

### Character Error Rate (CER)

$$\text{CER} = \frac{\sum_{i=1}^{N} \text{EditDistance}(\hat{y}_i, y_i)}{\sum_{i=1}^{N} |y_i|}$$

Where EditDistance = Levenshtein distance (min insertions + deletions + substitutions). NFC-normalised strings are compared at the Unicode codepoint level.

### Word Error Rate (WER)

$$\text{WER} = \frac{\text{count}(\hat{y}_i \neq y_i)}{N}$$

Whole-word: 0 if perfectly correct, 1 if any character differs.

### Bootstrap 95% Confidence Interval

```python
n_resamples = 1000
for _ in range(n_resamples):
    sample_idx = np.random.choice(N, N, replace=True)  # resample with replacement
    resample_cer = compute_cer([all_preds[i] for i in sample_idx],
                               [all_gts[i]   for i in sample_idx])
    bootstrap_cers.append(resample_cer)
ci_lower = np.percentile(bootstrap_cers, 2.5)
ci_upper = np.percentile(bootstrap_cers, 97.5)
```

**Non-overlapping CIs → statistically significant difference at α=0.05**

### Virama Breakdown

Test samples split into:
- **Compound** (GT contains U+0C4D virama): n=10,129
- **Simple** (no virama): n=7,781

CER and WER computed separately for each group.

---

## 11. Final Results

### Ablation Table (Test Set N=17,910)

| # | Model | CER | WER | 95% CI | Compound CER | Simple CER | Speed |
|---|-------|-----|-----|--------|--------------|------------|-------|
| 1 | **CTC Baseline** | **3.91%** | **24.80%** | [3.79, 4.03] | **3.70%** | **4.29%** | **2.7 ms** |
| 2 | AR (unconstrained) | 4.89% | 29.83% | [4.75, 5.02] | 4.61% | 5.38% | 1.7 ms |
| 3 | AR + grammar mask | 4.99% | 29.71% | [4.85, 5.13] | 4.68% | 5.54% | 40.2 ms |
| 4 | AR + mask + beam(5) | 4.85% | 29.49% | [4.72, 5.00] | 4.57% | 5.35% | 327.5 ms |
| 5 | AR (no CTC aux) | 5.43% | 32.37% | [5.28, 5.58] | 4.99% | 6.21% | 1.7 ms |

### Key Findings

**1. CTC beats Transformer** — CIs [3.79,4.03] vs [4.72,5.00] are non-overlapping → statistically significant. Reason: short word-level sequences (avg 8.7 chars) don't benefit from long-range language modeling. BiLSTM's bidirectional context is sufficient.

**2. Auxiliary CTC is critical** — Row 5 vs Row 2: removing auxiliary CTC raises CER from 4.89% → 5.43% (+10.9% relative). The CTC head forces the encoder to produce alignment-aware features.

**3. Grammar mask: marginal** — CI [4.85,5.13] (masked) vs [4.75,5.02] (unmasked) overlap heavily → **not significant**. Model already learned most orthographic rules implicitly.

**4. Compound < Simple CER** — Counterintuitive: compound characters (with virama conjuncts) are *easier* (3.70% vs 4.29%). Hypothesis: conjunct forms have more visually distinctive shapes → stronger CNN discriminative features.

### vs. Published SOTA (IIIT-HW-Telugu)

| Method | CER | Lexicon? |
|--------|-----|---------|
| Dutta et al. 2018 | ~15.0% | No |
| Deshpande et al. 2021 | ~10.2% | No |
| Gongidi & Jawahar 2021 | ~6.41% | No |
| **Ours (CTC)** | **3.91%** | **No** |
| Gongidi & Jawahar 2021 | ~1.52% | **Yes** |

**→ New lexicon-free SOTA. 39% relative improvement over prior best.**

---

## 12. Full Tensor Flow — End to End

```
RAW INPUT
─────────────────────────────────────────────────────────────────────
PIL Image (RGB, arbitrary size)

PREPROCESSING
─────────────────────────────────────────────────────────────────────
→ Grayscale → Augment (train) → Resize H=64 → Pad W=512 → Normalize
→ Tensor [1, 64, 512]   float32 in [-1, 1]
→ scaled_width (int): true content width before padding

BATCHING
─────────────────────────────────────────────────────────────────────
collate_fn_pad:
→ images  [B, 1, 64, 512]
→ labels  [B, T_max]       (padded with 0=PAD)
→ lengths [B]              (true label lengths)
→ widths  [B]              (true image widths)

SHARED CNN ENCODER (ResNet-18, patched)
─────────────────────────────────────────────────────────────────────
[B, 1, 64, 512]
→ repeat to RGB           [B,  3, 64, 512]
→ Conv1(7×7) + MaxPool    [B, 64, 32, 256]
→ Layer1 (2×BasicBlock)   [B, 64, 32, 256]
→ Layer2 (2×BasicBlock)   [B,128, 16, 128]
→ Layer3 (2×BasicBlock)*  [B,256,  8,  64]   *stride=(2,1)
→ Layer4 (2×BasicBlock)*  [B,512,  4,  64]   *stride=(2,1)
→ Conv2d(512→256, (2,1))  [B,256,  1,  64]
→ GELU + BN
→ mean(dim=2)             [B,256,     64]
→ permute(0,2,1)          [B, 64,    256]
→ + sinusoidal PE         [B, 64,    256]    ← ENCODER MEMORY

══════════════════════════════════════════════════════
PATH A — CTC BASELINE
══════════════════════════════════════════════════════
memory [B, 64, 256]
→ BiLSTM(256→256, 2L, bidir)  [B, 64, 512]
→ Dropout(0.1)
→ Linear(512→91)              [B, 64,  91]
→ log_softmax(dim=2)          [B, 64,  91]
→ permute(1,0,2)              [64, B,  91]
→ CTCLoss(blank=0)            scalar loss

  GREEDY DECODE (inference):
  [B, 64, 91] → argmax → [B, 64] → collapse+remove_blank → List[List[int]]
  → vocab.decode() → List[str]
  → editdistance → CER/WER

══════════════════════════════════════════════════════
PATH B — AR TRANSFORMER
══════════════════════════════════════════════════════
memory [B, 64, 256]
→ TransformerEncoder(2L, 4H, FF=1024, Pre-LN)  [B, 64, 256]
  │
  ├── AUX CTC HEAD (training only):
  │   Linear(256→91) → log_softmax → permute → CTCLoss  (×0.3)
  │
  └── MAIN AR DECODER:
      target_ids [B, T] (teacher-forced: y[:-1])
      → Embedding(91→256) + PE  [B, T, 256]
      → TransformerDecoder(4L, 4H, FF=1024, Pre-LN)
           self-attn: causal mask [T,T]
           cross-attn: attends to memory [B,64,256]
        → [B, T, 256]
      → Linear(256→91)           [B, T, 91]
      → CrossEntropyLoss(smooth=0.1, target=y[1:])  (×0.7)

  TOTAL LOSS = 0.7·CE + 0.3·CTC

  GREEDY DECODE (inference):
  Loop T steps:
    [B, t, 256] → decoder → [B, 91] (last step logits)
    → grammar_mask (−10 to invalid tokens)
    → argmax → next_token [B]
    → append → repeat
  → List[List[int]] → vocab.decode() → List[str] → CER/WER

  BEAM DECODE (k=5, length_penalty=0.6, inference):
  Per image: maintain 5 beams, expand topk, score = log_prob / len^0.6
  → best completed beam → str → CER/WER
```
