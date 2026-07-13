# Comprehensive Project Context: Telugu Handwritten Text Recognition (HTR)

**Purpose of this document:** This file contains the complete, detailed context of the Telugu HTR project. It is specifically designed to be read by another AI agent to understand the entire architecture, data flow, and design decisions so that it can hunt for deep logical flaws, architectural bottlenecks, or hidden bugs.

---

## 1. Project Goal & Scope
The objective is to build a state-of-the-art Offline Handwritten Text Recognition system for Telugu words (specifically targeting the IIIT-HW-Telugu dataset). Because Telugu is a complex script with intricate ligatures (vothulu and matralu), standard CTC models often fail to capture the orthographic rules. This project uses a **Multi-Task Autoregressive Transformer** with a data-derived linguistic constraint layer to solve this.

---

## 2. Directory Structure & File Responsibilities

*   **`configs/`**
    *   `ctc_config.yaml`: Configuration for the baseline CNN+BiLSTM+CTC model.
    *   `ar_config.yaml`: Configuration for the primary Transformer AR model.
*   **`src/`**
    *   `dataset.py`: Handles loading IIIT-HW-Telugu images and labels. Returns `(image_tensor, original_width, label_ids, label_len)`.
    *   `transforms.py`: Handles image augmentation (elastic distortion, morphological thinning/thickening, rotation) and resizing. Keeps aspect ratio, pads width to 512, height fixed at 64.
    *   `vocab.py`: Builds and manages the Telugu character vocabulary and bigram transition matrix.
    *   `checkpoint_manager.py`: Manages saving/loading weights, optimizer states, and rolling checkpoints (`current.pt`, `previous.pt`, `best.pt`).
    *   `train_ctc.py` / `train_ar.py`: Training loops with AMP, OneCycleLR/Cosine LR, and tensorboard logging.
    *   `evaluate.py`: Evaluation script that computes CER/WER and breaks down errors by simple vs. compound characters. Uses Unicode NFC normalization.
*   **`src/models/`**
    *   `cnn_encoder.py`: A modified ResNet-18 backbone. Removes fully connected layers. Modifies pooling strides to preserve horizontal sequence length (`width_downsample` = 8 by default, or 4 if `high_res_temporal=True`).
    *   `ctc_model.py`: The baseline model. ResNet → BiLSTM → CTC Head.
    *   `ar_model.py`: The main SOTA model. ResNet → Transformer Encoder → Transformer Decoder + Joint CTC Head.
*   **`src/decoding/`**
    *   `telugu_mask.py`: The "Grammar Coach". Uses the transition matrix from `vocab.py` to apply a soft penalty (`constrain_penalty`) or hard `-inf` mask to invalid Telugu character bigrams during autoregressive decoding.

---

## 3. Data Pipeline & Processing
1.  **Image Resize:** All images are resized to `H=64` while maintaining aspect ratio, then padded with white pixels to `W=512`.
2.  **Width Tracking:** Because images are padded, the dataloader passes the *original scaled width* (`input_widths`) through the batch so that the model can generate a padding mask and ignore the white space during Transformer attention and CTC loss calculation.
3.  **Augmentation (`transforms.py`):**
    *   Elastic Distortion (mimics natural handwriting variance).
    *   Morphological Ops (erosion/dilation via PIL Min/Max filters to simulate pen thickness).
    *   Rotation (±5°).
    *   Color Jitter (Brightness/Contrast).
4.  **Collation:** `collate_fn_pad` pads the variable-length label sequences with `0` (the `<PAD>` token).

---

## 4. Main Architecture (`ar_model.py`)
This is a **Multi-Task Autoregressive Transformer**.

*   **Visual Encoder:** ResNet-18 modified to output a spatial feature map of shape `[B, 512, 1, W']`. This is squeezed and projected to `[B, W', d_model]`.
*   **Transformer Encoder:** Processes the visual features using self-attention. A `memory_key_padding_mask` is applied using the `input_widths` to prevent attention on padded regions.
*   **Transformer Decoder:** Autoregressive generation. Uses causal masking to prevent looking ahead.
*   **Joint Loss Training:** 
    *   The `compute_loss()` function calculates standard Cross-Entropy loss for the Autoregressive Decoder output.
    *   It also contains an auxiliary `ctc_head` attached directly to the Transformer Encoder output. It calculates CTC loss against the unpadded labels (excluding SOS/EOS).
    *   `Total_Loss = 0.7 * CE_Loss + 0.3 * CTC_Loss`. This forces the encoder to learn perfect monotonic alignment.
*   **Decoding:** Uses Beam Search (width=5) or Greedy Decode.

---

## 5. Decoding Constraint (`telugu_mask.py`)
Because Telugu ligatures follow strict grammatical rules, the model includes a Data-Derived Transition Matrix.
*   During `greedy_decode` or `beam_decode`, before the argmax or top-k selection is made, `telugu_mask.py` inspects the previously predicted character.
*   It checks the `valid_matrix` (built from the training set). If the model is predicting a character that legally cannot follow the previous character, a `constrain_penalty` (e.g., 10.0) is subtracted from that character's logit.
*   This acts as a soft constraint, gently forcing the model to pick grammatically valid Telugu ligatures.

---

## 6. Evaluation Metrics
*   **CER (Character Error Rate) & WER (Word Error Rate):** Based on Levenshtein distance.
*   **NFC Normalization:** Before calculating CER/WER, all predicted strings and ground truth strings are passed through `unicodedata.normalize("NFC", text)`. This is critical for Indic scripts to prevent visually identical strings with different byte representations from being penalized.

---

## 7. Recently Fixed Bugs (Do Not Flag These)
The previous agent should know that the following tricky bugs were *already* identified and fixed in the current codebase:
1.  **Padding Mask Calculation:** Fixed `input_widths // 8` to use `input_widths // self.cnn_encoder.width_downsample` to support `high_res_temporal` properly.
2.  **CTC Eval Crash:** `input_widths` was added to `CTCModel.greedy_decode()`.
3.  **Config Propagation:** `num_encoder_layers` and `high_res_temporal` are now properly passed from `ar_config.yaml` to the model constructors in both `train_ar.py` and `evaluate.py`.
4.  **Metric Consistency:** NFC normalization was added to the validation loops in the training scripts so that validation CER perfectly matches evaluation CER.
5.  **Modern PyTorch AMP:** `torch.cuda.amp` was replaced with `torch.amp` everywhere.

---

## 8. Mission for the Reviewing Agent
Given the complete context above, your task is to review the code with extreme prejudice. Look for:
1.  **Mathematical Flaws:** Are the loss calculations (Joint CTC + CE) mathematically sound and scaled correctly? Are log_probs and logits being mixed up anywhere?
2.  **Tensor Shape Mismatches:** Does the beam search implementation in `ar_model.py` correctly handle the expanding tensor shapes? Does the attention mask broadcasting work perfectly for all batch sizes?
3.  **Memory Leaks:** Are tensors building up computational graphs during evaluation or metric calculation?
4.  **Data Leakage:** Is there any way the test set or validation set is leaking into the training pipeline or the vocabulary transition matrix?
5.  **Architectural Bottlenecks:** Is the ResNet-18 outputting a spatial resolution that is too small for the Transformer Encoder to be useful? 

Please analyze the codebase and report any critical flaws.
