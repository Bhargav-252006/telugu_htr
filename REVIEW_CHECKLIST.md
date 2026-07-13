# Formal Code-Review Checklist for Telugu HTR

This checklist represents the highest-priority checks for implementation correctness in the Multi-Task AR HTR system.

## 1. Loss & Training Dynamics
- [ ] **Loss Scaling & Logging:** Verify that CTC and CE losses are not overpowering each other. Ensure both loss magnitudes are logged separately to TensorBoard during training for empirical verification.
- [ ] **CE Target Padding:** Verify `nn.CrossEntropyLoss` correctly uses `ignore_index` for the `<PAD>` token so padded regions don't skew the loss.
- [ ] **CTC Target Construction:** Verify CTC targets strictly exclude Autoregressive-only tokens (`<SOS>` and `<EOS>`).
- [ ] **Checkpoint Selection Fairness:** Verify whether the validation loop selects the `best.pt` checkpoint using *unconstrained* decoding to ensure the baseline AR metrics are fair before constraints are applied.

## 2. Masking & Tensor Logic
- [ ] **Decoder Masks:** Verify the Transformer Decoder receives all three masks correctly during both teacher-forcing (training) and autoregressive generation (inference):
  - `tgt_mask` (Causal Mask)
  - `tgt_key_padding_mask` (Target Padding Mask)
  - `memory_key_padding_mask` (Encoder Padding Mask)
- [ ] **Beam Search Shape Logic:** Verify tensor broadcasting and score bookkeeping after beam expansion (e.g., when `batch_size` is multiplied by `beam_size`).
- [ ] **Beam Score Normalization:** Verify that finished beams and active beams are scored consistently, and length-normalization is applied to prevent bias against longer Telugu words.

## 3. Data Leakage & Evaluation Rigor
- [ ] **Transition Matrix Integrity:** Verify `valid_matrix` in `vocab.py` is built **strictly** from the training split, and never updated during validation or test evaluations.
- [ ] **Unicode Normalization Mismatch:** Verify `unicodedata.normalize("NFC", ...)` is applied identically during both training validation (checkpoint selection) and final test-set evaluation.
- [ ] **Computational Graph Build-up:** Verify evaluation scripts run completely under `torch.no_grad()` to prevent memory leaks and graph accumulation.
- [ ] **Width-Downsample Factor:** Verify the CTC head input lengths are dynamically computed from the actual padded image widths, not the static `max_width`.
