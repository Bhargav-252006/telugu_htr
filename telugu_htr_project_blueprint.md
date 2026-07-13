# Telugu Handwritten Text Recognition Project Blueprint

## Overview
This project targets **word-level Telugu handwritten text recognition** using the public IIIT-HW-Telugu dataset from IIIT Hyderabad, which provides Telugu handwritten word images with train, validation, and test splits and paired ground-truth text files.[cite:207][cite:209] The intended research direction is to compare a standard handwritten text recognition baseline against a Transformer-based autoregressive recognizer and then extend that recognizer with a Telugu-script-aware decoding constraint.[cite:180][cite:223]

## Core research question
Can a **Telugu-script-aware autoregressive decoder** improve recognition quality over standard CTC-based Telugu handwritten text recognition pipelines on public Telugu benchmarks, especially for modifier-heavy words, ligatures, and difficult handwriting styles?[cite:192][cite:223]

## Main method
The work should be executed in three stages:

1. **Baseline model**: build or reuse a CNN/BiLSTM/CTC or ResNet/CTC baseline for Telugu handwritten word recognition, following existing Indic HTR baselines.[cite:223]
2. **Transformer Encoder-Decoder**: replace the CTC decoding stage with a modern TrOCR-style architecture (ResNet + Transformer Encoder + Transformer Decoder) that predicts the Telugu transcription character by character from globally contextualized image features.[cite:180][cite:182]
3. **Proposed model**: empirically evaluate a **data-derived Telugu-aware constraint mechanism** in decoding (via a tunable soft penalty) so the model respects observed Telugu character composition patterns without hard-blocking valid but rare sequences.[cite:192][cite:228]

## Exact work order

### Phase 1: Dataset preparation
- Download the IIIT-HW-Telugu dataset and inspect the official split files and ground-truth annotations.[cite:207][cite:209]
- Create a parser for `train_gt.txt`, `val_gt.txt`, and `test_gt.txt` so each sample maps `image_path -> Telugu text label`.[cite:209]
- Build a Telugu Unicode vocabulary from the training split only, then add special tokens such as `<pad>`, `<bos>`, and `<eos>` for the autoregressive system.[cite:223]
- Normalize images to a fixed height while preserving aspect ratio, then pad width as needed for batching.

### Phase 2: Baseline pipeline
- Train a strong baseline first using a standard Indic HTR implementation or a custom CNN/BiLSTM/CTC pipeline.[cite:223]
- Use Character Error Rate (CER) and Word Error Rate (WER) as the main evaluation metrics, since these are standard in handwritten text recognition benchmarking.[cite:223][cite:216]
- Save predictions on the validation set and manually inspect frequent Telugu-specific errors such as ligature confusion and modifier confusion.[cite:192]

### Phase 3: Plain autoregressive recognizer
- Build an image encoder using a lightweight CNN such as ResNet-18 or MobileNet-style visual feature extraction, followed by optional Transformer encoder layers.[cite:180][cite:182]
- Build a Transformer decoder that predicts the next Telugu character using previous output characters and cross-attention over image features.[cite:180]
- Train this model as a plain sequence-to-sequence recognizer without script-specific rules first, so it becomes the main comparison point for the proposed method.[cite:180][cite:182]

### Phase 4: Proposed Telugu-aware decoder
- Add a data-derived Telugu-script-aware decoding prior:
  - Extract transition validity from the training set labels.
  - Apply a **soft masking penalty** for illegal next-character combinations (e.g. subtract 10.0 from logits) rather than a hard `-inf` mask. This prevents overblocking valid but rare unseen ligatures.
- Keep the first version simple and testable: do not combine too many ideas in one model.
- Compare this model directly against the plain autoregressive decoder, not only against CTC.

### Phase 5: Experiments and ablations
- Run three core models: baseline CTC, plain autoregressive decoder, and Telugu-aware autoregressive decoder.[cite:223][cite:180]
- Add ablations such as:
  - with vs without Telugu-aware mask,
  - greedy decoding vs beam search,
  - encoder-only changes vs decoder-only changes.
- Evaluate difficult subsets manually, such as long words, rare symbols, and words containing multiple modifiers.[cite:192]

### Phase 6: Analysis and writing
- Report CER and WER on validation and test sets.[cite:223][cite:216]
- Include qualitative visual examples of correct predictions, failure cases, and cases improved by the Telugu-aware decoder.
- Write the project report using the structure: problem, related work, dataset, method, experiments, results, error analysis, conclusion.

## Model design recommendation

| Component | Recommended starting choice | Reason |
|---|---|---|
| Image encoder | ResNet-18 | Stable, lightweight, enough for word-level HTR |
| Optional visual sequence encoder | 2–4 Transformer encoder layers | Adds context over visual tokens without being too heavy |
| Decoder | 4-layer Transformer decoder | Enough to model sequence dependency at student-project scale |
| Tokenization | Character-level Telugu Unicode | Most practical for HTR and CER-based evaluation |
| Baseline | CNN/BiLSTM/CTC | Strong and standard comparison point [cite:223] |
| Metrics | CER, WER | Standard HTR metrics [cite:216][cite:223] |

## Recommended software stack
- Python 3.10+
- PyTorch
- torchvision
- Pillow or OpenCV
- NumPy and pandas
- TensorBoard or Weights & Biases
- Git
- Optional: existing Indic HTR baseline repository for faster setup.[cite:223]

## Hardware requirements
A **RTX 3090 Ti with 24 GB VRAM** is sufficient for this project because the target task is word-level handwritten text recognition, not large-scale language-model pretraining.[cite:180][cite:182] A practical system target is:

- GPU: RTX 3090 Ti 24 GB
- RAM: 32 GB recommended
- CPU: modern multi-core CPU
- Storage: at least 30–50 GB free for dataset copies, checkpoints, logs, and ablations

## Practical training plan
- Start with image height 32 or 48 and moderate batch size.
- Use mixed precision training to improve throughput.
- Save checkpoints every epoch.
- Track CER on validation after every epoch and stop based on best validation CER.
- Keep experiment names clean, for example:
  - `baseline_ctc_v1`
  - `ar_decoder_v1`
  - `ar_decoder_telugu_mask_v1`

## Resource list

### Datasets
- IIIT-HW-Telugu dataset page and split details.[cite:207][cite:209]
- Indic HTR competition/benchmark datasets if later extension is needed.[cite:216][cite:210]

### Baseline and reference papers
- TrOCR: Transformer-based OCR.[cite:180]
- A Light Transformer-Based Architecture for Handwritten Text Recognition.[cite:182]
- Indic handwritten text recognition baseline implementation and pipeline.[cite:223]
- Recent Telugu handwritten recognition papers highlighting script difficulty and data scarcity.[cite:192][cite:184]

## Expected output of the project
By the end of the project, the deliverables should be:
- a cleaned dataset loader,
- Telugu vocabulary builder,
- baseline CTC training pipeline,
- plain autoregressive recognizer,
- Telugu-aware autoregressive recognizer,
- CER/WER evaluation scripts,
- qualitative prediction examples,
- and a final report or paper draft.

## Suggested timeline

| Week | Work |
|---|---|
| 1 | Dataset setup, vocabulary creation, dataloader |
| 2 | Baseline CNN/BiLSTM/CTC training |
| 3 | Plain autoregressive model implementation |
| 4 | Telugu-aware constraint implementation |
| 5 | Ablations and benchmarking |
| 6 | Report writing and final cleanup |

## Safe novelty statement
A safe project claim is:

**“This work presents a rigorous empirical study of Telugu handwritten word recognition, evaluating a TrOCR-style autoregressive encoder-decoder and a data-derived soft-penalty decoding constraint against standard CTC baselines on the IIIT-HW-Telugu benchmark.”**[cite:207][cite:209][cite:180]

This wording is safer than claiming absolute novelty, because prior Telugu and Indic handwritten recognition work already exists. It frames the contribution around the evaluation of a specific architectural and constraining methodology.[cite:184][cite:192][cite:223]
