# Telugu HTR: Final System Architecture & Data Flow

This document outlines the exact flow of data, architectural choices, and methodological upgrades implemented in the Telugu Handwritten Text Recognition (HTR) pipeline. It serves as a structural reference for writing the final research paper.

---

## 1. Data Pipeline & Preprocessing

The data pipeline has been explicitly designed to handle variable-width Telugu words while preserving true sequence lengths for downstream attention masking.

### Image Transformations (`src/transforms.py`)
1. **Grayscale Conversion**: Images are loaded as RGB and converted to Grayscale.
2. **Resize (Aspect-Preserving)**: All images are resized to a fixed height of `H=64`. The width scales proportionally.
3. **Width Capture**: The *true* scaled width of the image is captured before padding.
4. **Padding**: Images are padded on the right with white pixels (255) to a fixed `MAX_WIDTH = 512`.
5. **Output**: The transform yields `(tensor, original_width)`.

### Batch Collation (`src/dataset.py`)
The `collate_fn_pad` batches samples into a 4-tuple:
- `images`: `[B, 1, 64, 512]` (fixed tensor for `cudnn.benchmark` optimizations).
- `labels`: `[B, max_label_len]` (padded token IDs).
- `label_lens`: `[B]` (true length of each label sequence).
- **`image_widths`**: `[B]` (true unpadded width of each image in pixels).

---

## 2. Model Architecture (TrOCR-Style)

The autoregressive (AR) model uses an Encoder-Decoder architecture to map visual features to sequence tokens.

### A. Visual CNN Encoder (ResNet-18)
- **Pretraining Integrity**: The 1-channel grayscale image is replicated across 3 channels (`x.repeat(1, 3, 1, 1)`) to preserve the semantic weights of the ImageNet-pretrained ResNet-18.
- **Stride Patching**: Standard ResNet-18 downsamples spatial dimensions by a factor of 32, which destroys the temporal resolution required for long Telugu words. We patch the strides of `layer3` and `layer4` to `(1, 1)`. 
- **Temporal Resolution**: 
  - *Default (S=64)*: `maxpool` stride is `(2, 2)`. Width is downsampled by a factor of 8. A 512px image yields 64 visual steps.
  - *High-Res Ablation (S=128)*: `maxpool` stride is patched to `(2, 1)`. Width is downsampled by a factor of 4. A 512px image yields 128 visual steps.

### B. Transformer Encoder (Global Context)
- **Feature Projection**: CNN features are squeezed into 1D sequences and projected to `d_model=256`. 1D Positional Encodings are added.
- **Self-Attention**: The sequence `[B, S, 256]` passes through a **2-layer Transformer Encoder**. This is critical: it allows the model to globally contextualize visual features across the entire image (e.g., matching a modifier on the left to a base character on the right) *before* the autoregressive decoding begins.
- **Padding Mask**: The `image_widths` tensor is divided by the downsample factor (8 or 4) to create a `src_key_padding_mask`. The Transformer Encoder entirely ignores the white padding.

### C. Transformer Decoder (Autoregressive)
- **Architecture**: A 4-layer Transformer Decoder with causal masking.
- **Cross-Attention Masking**: Uses the same `memory_key_padding_mask` to ensure the decoder only attends to valid visual features and ignores the padded white space.

---

## 3. Decoding & Telugu-Aware Constraints

### Data-Derived Transition Matrix
Instead of hard-coding linguistic rules, the model scans the training set labels to build a sparse validity matrix of observed character bigrams (prev_char $\rightarrow$ next_char).

### Soft Penalty Masking
During `greedy_decode` or `beam_decode`:
- For a given `prev_token`, the validity matrix yields a boolean mask of allowed next tokens.
- **Soft Penalty**: Instead of assigning a hard `-inf` to invalid transitions, we subtract a hyperparameter `constrain_penalty` (e.g., 10.0) from the logits. 
- **Rationale**: If the model encounters a valid but exceedingly rare Telugu ligature not seen in the training set, a strong visual signal can overcome the soft penalty. A hard mask would force a misprediction.

### Beam Search Normalization
The beam search includes a `length_penalty` (default 0.6) to prevent the decoder from heavily favoring shorter sequences, ensuring fair comparison for long compound words.

---

## 4. Training & Evaluation Methodology

### Unbiased Checkpointing
During training, the system evaluates validation CER in two ways:
1. **Constrained**: Logs the metric for observational tracking.
2. **Unconstrained**: Used to select and save `best.pt`.
*Why?* If we select the best checkpoint using constrained decoding, we bias the model selection towards the proposed method. Selecting via unconstrained decoding ensures fair scientific ablations.

### Fair Metric Evaluation (NFC Normalization)
Telugu syllables can be represented in Unicode as pre-composed characters or decomposed sequences (e.g., base + virama + consonant). 
- Before computing CER or WER, all predicted and ground truth strings are passed through `unicodedata.normalize("NFC", text)`. 
- This prevents the standard `editdistance` metric from penalizing the model for generating visually and linguistically identical text that happens to have a different byte representation.
