# FINAL PPT CONTENT
## Grammar-Aware Handwritten Telugu Text Recognition Using CNN-Transformer Architecture with Data-Driven Linguistic Constraints
### B.Tech CSE Major Project Presentation

---

---

## SLIDE 1 — Title Slide

**Slide Title:** Grammar-Aware Handwritten Telugu Text Recognition Using CNN-Transformer Architecture with Data-Driven Linguistic Constraints

**Content:**
- **Department:** Computer Science & Engineering
- **Degree:** Bachelor of Technology (B.Tech) — Major Project
- **Team Members:** [Insert Names & Roll Numbers]
- **Guide:** [Insert Guide Name & Designation]
- **Academic Year:** 2025–2026

**Visual Suggestion:**
- Clean, professional title layout with university/college logo at the top-left
- A subtle background showing faded Telugu handwritten text samples
- Use a gradient banner (deep blue → teal) behind the title text

**Speaker Notes:**
Good morning, respected faculty and fellow students. Today we present our Major Project titled "Grammar-Aware Handwritten Telugu Text Recognition Using CNN-Transformer Architecture with Data-Driven Linguistic Constraints." This project addresses a critical gap in handwritten text recognition for Telugu — one of the most complex scripts in the world. Over the next few minutes, we will walk you through the problem, our novel approach, and the results we achieved.

---

---

## SLIDE 2 — Table of Contents

**Slide Title:** Presentation Outline

**Content:**
1. Problem Statement
2. Motivation
3. Literature Survey
4. Research Gap
5. Objectives
6. Proposed System Architecture
7. Visual Encoder (ResNet-18)
8. CTC Baseline Model
9. AR Transformer Model
10. Joint CTC + CE Training Strategy
11. Telugu Linguistic Constraint Mask *(Novel Contribution)*
12. Dataset Description
13. Data Preprocessing Pipeline
14. Training Configuration
15. Training Progress — CTC & AR Models
16. Evaluation Methodology
17. Results & Analysis
18. Comparison with Published Work
19. Key Findings & Contributions
20. Limitations & Future Scope
21. Conclusion
22. References

**Visual Suggestion:**
- Two-column numbered list layout with alternating light/dark row shading
- Highlight "Telugu Linguistic Constraint Mask" row in accent colour to flag the novel contribution
- Small icons next to each section (gear for architecture, chart for results, book for references)

**Speaker Notes:**
Here is our presentation outline. We will cover the problem and motivation, survey existing literature, detail our proposed architecture including the novel Telugu linguistic constraint mask, describe the dataset and training, and finally present our experimental results and conclusions. We have structured the talk into 28 slides covering each aspect comprehensively.

---

---

## SLIDE 3 — Problem Statement

**Slide Title:** Problem Statement

**Content:**
- **Telugu Script Complexity:**
  - 56 base characters (16 vowels + 38 consonants + 2 special symbols)
  - 18+ vowel modifiers (mātrās) that attach to consonants in various positions (above, below, left, right)
  - Compound characters (conjuncts) formed using the Virama (హల్లు) — combining two or more consonants into a single ligature
  - Results in a combinatorial explosion of visual glyph forms exceeding 1,000+ unique shapes
- **Handwriting Variability:**
  - Extreme variability in stroke patterns, slant, spacing, and writing styles across individuals
  - No standardized handwriting norms — significant inter-writer and intra-writer variation
- **Current System Failures:**
  - Existing OCR systems (Google Tesseract, commercial tools) achieve poor accuracy on handwritten Telugu
  - Most systems are designed for printed text and fail to generalize to cursive, connected handwriting
  - No existing system incorporates Telugu grammatical rules to constrain or correct predictions
- **Core Problem:** How to build an accurate, grammar-aware handwritten Telugu text recognition system that handles the full complexity of the script?

**Visual Suggestion:**
- Left panel: Examples of Telugu script complexity — show base characters, vowel modifiers, and compound characters with labels
- Right panel: Samples of handwritten Telugu showing high variability across different writers
- Bottom callout box highlighting the core problem statement in bold

**Speaker Notes:**
Telugu is a Dravidian language with one of the most complex writing systems in the world. The script has 56 base characters, but the real challenge comes from the 18-plus vowel modifiers and compound characters formed using the Virama. This creates over a thousand unique visual glyph forms. When you add handwriting variability on top of this — different stroke patterns, slant, spacing — the recognition problem becomes extremely challenging. Current OCR tools like Google Tesseract perform poorly on handwritten Telugu, and crucially, no existing system uses grammatical rules to validate its predictions. This is the gap we address.

---

---

## SLIDE 4 — Motivation

**Slide Title:** Motivation

**Content:**
- **Linguistic Significance:**
  - Telugu is the 4th most spoken language in India (~82 million native speakers)
  - Official language of Andhra Pradesh and Telangana
  - Classified as a "Classical Language of India" by the Government of India
- **Digitization Need:**
  - Vast repositories of handwritten Telugu documents (historical manuscripts, government records, educational materials) remain undigitized
  - Manual transcription is labour-intensive, slow, and error-prone
  - Automated HTR can accelerate digital archiving by orders of magnitude
- **Technology Gap:**
  - Handwritten text recognition for Telugu lags significantly behind Latin-script languages (English, French)
  - English HTR achieves <1% CER on standard benchmarks; Telugu HTR systems report 10–15% CER
  - Limited publicly available datasets and benchmark models for Telugu HTR
- **Practical Impact:**
  - Enables searchable digital archives, preserving cultural heritage
  - Supports assistive technology for visually impaired Telugu readers
  - Facilitates automatic form/document processing in regional language

**Visual Suggestion:**
- Infographic-style layout with 4 quadrants (Linguistic Significance, Digitization Need, Technology Gap, Practical Impact)
- Include a small map of India highlighting Andhra Pradesh and Telangana
- Use icons: speech bubble (linguistic), document stack (digitization), gap/bridge (tech gap), accessibility symbol (practical impact)

**Speaker Notes:**
The motivation for this project is multi-fold. Telugu is spoken by over 82 million people and is the official language of two Indian states. There is an enormous volume of handwritten Telugu content — from historical manuscripts to government records — that needs to be digitized. Current technology for Telugu HTR is far behind what exists for English. While English HTR achieves sub-1% character error rates, the best Telugu systems were reporting 10 to 15 percent. This gap, combined with the practical need for digitization and accessibility, drove our research.

---

---

## SLIDE 5 — Literature Survey

**Slide Title:** Literature Survey

**Content:**

| # | Authors (Year) | Method | Key Contribution | Limitation |
|---|---|---|---|---|
| 1 | Dutta et al. (2018) | MDLSTM + CTC | Introduced IIIT-HW-Telugu dataset; multi-dimensional LSTM for Telugu HTR | CER ~15%; no grammar constraints |
| 2 | Deshpande et al. (2021) | CNN + LSTM + CTC | Improved feature extraction with deeper CNNs for Indic scripts | CER ~10.2%; still high error rates |
| 3 | Shi et al. (2017) | CRNN (CNN + BiLSTM + CTC) | Foundational architecture for scene text recognition | Designed for Latin scripts; not adapted for Telugu |
| 4 | Li et al. (2023) | TrOCR (Transformer encoder-decoder) | End-to-end Transformer OCR without CNN backbone | Requires massive pretraining data; not tested on Indic scripts |
| 5 | Graves et al. (2006) | CTC Loss Function | Introduced Connectionist Temporal Classification for sequence labelling | Conditional independence assumption limits context modelling |
| 6 | Vaswani et al. (2017) | Transformer Architecture | Self-attention mechanism for sequence-to-sequence tasks | Not directly designed for image-to-text; needs visual encoder |
| 7 | Bluche & Messina (2017) | Attention-based HTR | Attention decoder for handwriting recognition | Struggles with very long sequences and complex scripts |

**Visual Suggestion:**
- Full-width table with alternating row colours (light grey / white)
- Highlight rows 1 and 2 (direct Telugu-related work) with a distinct accent colour
- Source/citation markers in the last column

**Speaker Notes:**
Let me briefly survey the key prior works. Dutta et al. in 2018 created the IIIT-HW-Telugu dataset that we use and achieved about 15% CER using multi-dimensional LSTMs. Deshpande et al. in 2021 improved this to about 10.2% CER using deeper CNNs. On the broader text recognition front, Shi et al.'s CRNN architecture and the more recent TrOCR by Li et al. represent the state of the art — but these are designed for Latin scripts and have not been adapted for Telugu. Importantly, none of these works incorporate linguistic constraints specific to Telugu grammar.

---

---

## SLIDE 6 — Research Gap

**Slide Title:** Research Gap

**Content:**
- **Gap 1 — High Error Rates:**
  - Best published result for handwritten Telugu HTR: ~10.2% CER (Deshpande et al., 2021)
  - Unacceptable for practical deployment (would mean ~1 in 10 characters is wrong)
  - Significant room for improvement with modern architectures
- **Gap 2 — No Grammar Awareness:**
  - All existing Telugu HTR systems treat output as independent character sequences
  - No system enforces or validates that predicted character sequences follow Telugu orthographic rules
  - Example: A system might predict a vowel modifier after another vowel modifier — grammatically impossible in Telugu
  - Invalid transitions degrade both accuracy and user trust
- **Gap 3 — No Hybrid Architecture Exploration:**
  - CTC-based models dominate Telugu HTR; autoregressive Transformer decoders have not been explored
  - Joint CTC + Cross-Entropy training (proven effective in speech recognition) has not been applied to Telugu HTR
- **Gap 4 — Limited Ablation Studies:**
  - No systematic comparison of CTC vs. AR vs. constrained decoding on the same dataset
  - Lack of compound vs. simple character-level analysis

**Visual Suggestion:**
- Four gap boxes arranged vertically or in a 2×2 grid, each with a distinct icon (warning sign, grammar book, neural network, magnifying glass)
- Use red/orange accent to emphasize gaps
- Arrows pointing from gaps → "Our Solution" at the bottom

**Speaker Notes:**
We identified four major research gaps. First, the error rates are still too high for practical use — 10% CER means one in every ten characters is wrong. Second, no existing system is grammar-aware — they can produce outputs that are linguistically impossible in Telugu. Third, modern hybrid architectures combining CTC with autoregressive Transformers have not been explored for Telugu. And fourth, there are no systematic ablation studies comparing these approaches on the same dataset. Our project directly addresses all four gaps.

---

---

## SLIDE 7 — Objectives

**Slide Title:** Project Objectives

**Content:**
1. **Design a high-accuracy HTR system** for handwritten Telugu using a hybrid CNN + Transformer architecture that significantly outperforms existing published baselines
2. **Implement and compare two decoding strategies:**
   - CTC-based greedy decoding (alignment-free, fast)
   - Autoregressive Transformer decoding with cross-attention (context-aware, flexible)
3. **Develop a novel data-driven Telugu linguistic constraint mask** that enforces valid character transitions during inference, reducing grammatically invalid predictions
4. **Conduct comprehensive ablation studies** to quantify the contribution of each component:
   - Shared visual encoder, CTC head, AR head, joint training, constraint mask, beam search
5. **Achieve state-of-the-art results** on the IIIT-HW-Telugu benchmark dataset, targeting CER < 5% (vs. prior best of ~10.2%)
6. **Provide detailed error analysis** including compound vs. simple character performance, speed benchmarks, and confidence intervals

**Visual Suggestion:**
- Numbered objective list with checkmark icons
- Each objective on a separate card/tile with subtle shadow
- Final objective (state-of-the-art) highlighted with a gold/star accent

**Speaker Notes:**
Our project has six clear objectives. We aimed to design a high-accuracy system, implement and compare CTC and autoregressive decoding, develop a novel linguistic constraint mask, conduct thorough ablation studies, achieve state-of-the-art results below 5% CER, and provide detailed error analysis. As we will show, we achieved all of these objectives — our best model reaches 3.91% CER, a 62% relative improvement over the previous best.

---

---

## SLIDE 8 — Proposed System Architecture (Overview)

**Slide Title:** Proposed System Architecture — Overview

**Content:**
- **High-Level Pipeline:**
  1. **Input:** Handwritten Telugu word image (grayscale, resized to 64 × 512)
  2. **Shared Visual Encoder:** ResNet-18 backbone (stride-patched) → Conv2D projection → Positional Encoding
  3. **Two Decoding Heads (trained on same encoder):**
     - **Head A — CTC:** BiLSTM → Linear → CTC Loss / Greedy Decode
     - **Head B — AR Transformer:** Transformer Encoder (2L) → Transformer Decoder (4L) → Cross-Entropy Loss / Autoregressive Decode
  4. **Inference Enhancement:** Telugu Linguistic Constraint Mask + Optional Beam Search
  5. **Output:** Predicted Telugu character sequence

- **Key Design Decisions:**
  - Shared encoder ensures both heads learn from the same visual features
  - CTC head provides a strong, fast baseline; AR head adds contextual modelling
  - Joint training (CTC 0.3 + CE 0.7) combines benefits of both loss functions
  - Constraint mask is applied only at inference — zero training overhead

**Visual Suggestion:**
- Full-width architecture flowchart (left to right):
  - Input Image → ResNet-18 Encoder → [splits into two paths] → CTC Head (top) / AR Transformer Head (bottom) → Output
  - Constraint mask shown as a filter applied before the AR output
- Use colour coding: blue for encoder, green for CTC, orange for AR, red for constraint mask
- Arrows showing data flow with tensor shape annotations

**Speaker Notes:**
Here is the overview of our proposed architecture. The system takes a handwritten Telugu word image, processes it through a shared ResNet-18 visual encoder, and feeds the resulting feature sequence into two decoding heads. The CTC head uses a bidirectional LSTM for fast alignment-free decoding. The AR Transformer head uses self-attention and cross-attention for context-aware autoregressive generation. At inference time, we apply our novel Telugu linguistic constraint mask to filter out grammatically invalid predictions. This modular design lets us compare and ablate each component systematically.

---

---

## SLIDE 9 — Visual Encoder (ResNet-18)

**Slide Title:** Shared Visual Encoder — Stride-Patched ResNet-18

**Content:**
- **Input Processing:**
  - Raw input: Grayscale handwritten image [B, 1, H, W]
  - Resized/padded to fixed size: [B, 1, 64, 512]
  - Channel replication: 1 → 3 channels (copies grayscale to all RGB channels)
  - Purpose: Leverage ImageNet-pretrained ResNet-18 weights (trained on 3-channel RGB images)

- **Backbone Modifications:**
  - Standard ResNet-18: stride=2 in both layer3 and layer4 → aggressively downsamples spatial dimensions
  - **Our Patch:** Set stride=1 in layer3 and layer4 → preserves width resolution
  - Rationale: For text recognition, horizontal spatial resolution is critical — each pixel column may correspond to a character or character part
  - Without patching: width reduces from 512 → 16 (only 16 feature slices for potentially 20+ characters)
  - With patching: width reduces from 512 → 64 (64 feature slices — sufficient resolution)

- **Output Processing:**
  - Feature map: [B, 512, 1, 64] → squeeze height dim → [B, 512, 64]
  - Conv2D projection: 512 → 256 dimensions (reduces model size, prevents overfitting)
  - Add sinusoidal Positional Encoding (injects position information since Transformers lack inherent ordering)
  - **Final encoder memory: [B, 64, 256]** — 64 visual tokens, each 256-dimensional

**Visual Suggestion:**
- Vertical block diagram of ResNet-18 showing each layer block (conv1, layer1–4) with spatial dimensions annotated
- Side-by-side comparison: standard stride (width→16) vs. patched stride (width→64) with red ✗ and green ✓
- Show tensor shape transformations at each stage

**Speaker Notes:**
The visual encoder is based on ResNet-18, pretrained on ImageNet. We replicate the single grayscale channel to three channels so we can directly use pretrained weights. The key modification is our stride patching — in standard ResNet-18, layers 3 and 4 downsample the width by a factor of 4, leaving only 16 feature columns. For text recognition, this is too aggressive — we need enough horizontal resolution to distinguish individual characters. By setting stride to 1 in these layers, we preserve 64 feature columns, giving us 64 visual tokens of 256 dimensions each. This is the representation fed to both decoding heads.

---

---

## SLIDE 10 — CTC Baseline Model

**Slide Title:** CTC Baseline Model

**Content:**
- **Architecture:**
  - Input: Encoder memory [B, 64, 256] from shared visual encoder
  - 2-layer Bidirectional LSTM:
    - Hidden size: 256 per direction → 512 total (forward + backward concatenated)
    - Adds sequential modelling over the visual feature sequence
  - Linear projection: 512 → 91 (vocabulary size = 91 Telugu characters + blank token)
  - **Total parameters: 14,115,483**

- **CTC (Connectionist Temporal Classification) Loss:**
  - Alignment-free: does not require explicit character-level segmentation
  - Introduces a "blank" token (ε) to handle variable-length alignment
  - Marginalizes over all possible alignments between input sequence (length 64) and output label
  - Decoding: Greedy (collapse repeated characters, remove blanks) — no beam search needed

- **Advantages:**
  - Simple, fast training and inference (0.23 ms per image)
  - No teacher forcing or autoregressive loop
  - Strong baseline — achieves 3.91% CER

- **Limitation:**
  - Conditional independence assumption: each output position is predicted independently
  - Cannot model explicit dependencies between consecutive characters

**Visual Suggestion:**
- Horizontal flow diagram: Encoder Memory → BiLSTM (2 layers, bidirectional) → Linear → CTC Loss
- Show CTC alignment example: input sequence with blanks → collapsed output Telugu string
- Annotate tensor shapes at each stage

**Speaker Notes:**
The CTC baseline model takes the encoder output and passes it through a two-layer bidirectional LSTM to add sequential context. The LSTM output is projected to the vocabulary size of 91 characters plus a blank token. CTC loss handles the alignment problem automatically — we don't need to know which image region corresponds to which character. At inference, we use simple greedy decoding: collapse repeated characters and remove blanks. Despite its simplicity, this model achieves our best result of 3.91% CER. However, CTC has a fundamental limitation — it assumes each output position is independent, so it cannot explicitly model character-to-character dependencies.

---

---

## SLIDE 11 — AR Transformer Model

**Slide Title:** Autoregressive Transformer Model

**Content:**
- **Transformer Encoder (refines visual features):**
  - Input: Encoder memory [B, 64, 256] from shared visual encoder
  - 2 Transformer Encoder layers with 8 attention heads
  - Self-attention over the 64 visual tokens → captures global spatial context
  - Output: Refined encoder memory [B, 64, 256]

- **Transformer Decoder (autoregressive character generation):**
  - 4 Transformer Decoder layers with 8 attention heads
  - **Causal self-attention:** each predicted character attends only to previously predicted characters
  - **Cross-attention:** each predicted character attends to all 64 visual tokens from the encoder
  - Character embedding: 256 dimensions; learned positional encoding
  - Output: Linear → Softmax over 91 characters at each timestep
  - **Total parameters: 17,302,518**

- **Autoregressive Decoding (Inference):**
  - Start with `<SOS>` (Start of Sequence) token
  - At each step t, predict next character conditioned on all previous predictions and the image
  - Stop when `<EOS>` (End of Sequence) token is predicted or max length reached
  - **Explicitly models character dependencies** — each prediction sees full prior context

- **Advantage over CTC:** Context-aware; can learn that certain character sequences are more likely
- **Disadvantage:** Slower inference due to sequential autoregressive loop (0.61 ms vs. 0.23 ms)

**Visual Suggestion:**
- Detailed Transformer architecture diagram showing:
  - Encoder stack (2 layers) with self-attention blocks
  - Decoder stack (4 layers) with causal self-attention + cross-attention blocks
  - Input/output embeddings and linear output layer
- Show autoregressive unrolling: SOS → char1 → char2 → ... → EOS

**Speaker Notes:**
The autoregressive Transformer model adds a Transformer encoder on top of our visual features for global context, followed by a Transformer decoder that generates characters one at a time. The decoder uses causal self-attention so each character can only see previously predicted characters, and cross-attention to attend to the image features. This is fundamentally different from CTC — the model explicitly conditions each prediction on the full history of prior characters. This comes at a cost: inference is slower at 0.61 milliseconds per image versus 0.23 for CTC. The total parameter count is about 17.3 million.

---

---

## SLIDE 12 — Joint CTC + CE Training Strategy

**Slide Title:** Joint CTC + Cross-Entropy Training Strategy

**Content:**
- **Motivation for Joint Training:**
  - CTC loss provides alignment-level supervision — learns which image regions map to which characters
  - Cross-Entropy (CE) loss provides token-level supervision — learns character-to-character transitions
  - Combining both gives the encoder gradients from two complementary learning signals

- **Loss Function:**
  ```
  L_total = α × L_CTC + β × L_CE
  where α = 0.3, β = 0.7
  ```
  - CTC weight (0.3): Regularizes encoder; prevents overfitting to autoregressive path
  - CE weight (0.7): Primary training signal for the Transformer decoder
  - Label Smoothing: 0.1 on the CE loss (softens targets, reduces overconfidence, improves generalization)

- **Training Protocol:**
  - Teacher forcing during training: decoder receives ground-truth characters as input (not its own predictions)
  - CTC head and AR head share the same visual encoder — gradients flow from both losses
  - This forces the encoder to produce features useful for both alignment-based and context-based decoding

- **Empirical Justification:**
  - Joint training is well-established in Automatic Speech Recognition (ASR) — e.g., ESPnet framework
  - First application to handwritten Telugu text recognition

**Visual Suggestion:**
- Diagram showing shared encoder with two loss branches:
  - Top branch: CTC Loss (weight 0.3) with dashed line
  - Bottom branch: CE Loss (weight 0.7) with solid line
  - Both merge at the encoder via gradient flow arrows
- Formula box prominently displaying the combined loss equation
- Small comparison table: CTC-only vs. CE-only vs. Joint

**Speaker Notes:**
We use joint training with both CTC and Cross-Entropy losses. The CTC loss with weight 0.3 provides alignment-level supervision — it teaches the encoder which parts of the image correspond to which characters. The Cross-Entropy loss with weight 0.7 is the primary signal for the Transformer decoder — it teaches character-level transitions. We also apply label smoothing of 0.1 to prevent overconfidence. This joint training strategy is well-established in automatic speech recognition systems but has not been applied to Telugu handwriting recognition before. The shared encoder receives gradients from both losses, forcing it to learn features that are useful for both decoding strategies.

---

---

## SLIDE 13 — Telugu Linguistic Constraint Mask (Novel Contribution)

**Slide Title:** Telugu Linguistic Constraint Mask — Novel Contribution

**Content:**
- **Core Idea:**
  - Build a character-level validity matrix from observed bigrams in training data
  - During inference, mask (penalize) transitions that were **never observed** in training
  - Purely data-driven — no hand-coded linguistic rules

- **Construction Algorithm:**
  1. Scan all training labels (88,534 word images)
  2. For each consecutive character pair (c_i, c_{i+1}), record the bigram
  3. Build a boolean matrix M[91 × 91]: M[a][b] = 1 if (a → b) was observed at least once
  4. Observed bigrams: ~2,800 out of 91 × 91 = 8,281 possible (33.8% density)
  5. Remaining 66.2% of transitions are marked as invalid

- **Inference Application:**
  - At each decoding step, given previous character c_prev:
    - Look up valid next characters from M[c_prev]
    - For all invalid next characters: subtract penalty of **−10.0** from their logits
    - This effectively zeroes out their probability after softmax (e^{-10} ≈ 4.5 × 10^{-5})
  - Applied before argmax (greedy) or before beam expansion (beam search)

- **Validation:**
  - Tested on validation set: **~0% constraint violations** — all valid transitions preserved
  - Ensures no valid character sequence is incorrectly blocked

- **Why This Matters:**
  - Prevents linguistically impossible predictions (e.g., consecutive vowel modifiers)
  - Zero additional training cost — applied only at inference
  - Generalizable approach: can be built for any script from training data

**Visual Suggestion:**
- Left: Heatmap visualization of the 91×91 constraint matrix (dark = valid, light = invalid)
- Center: Step-by-step algorithm flowchart (scan → extract bigrams → build matrix → apply at inference)
- Right: Before/after example — show a prediction where unconstrained model makes an invalid transition that the mask corrects
- Highlight "Novel Contribution" badge prominently

**Speaker Notes:**
This is our novel contribution — the Telugu linguistic constraint mask. The idea is simple but powerful. We scan all 88,534 training labels and extract every consecutive character pair — every bigram. We build a 91-by-91 boolean matrix indicating which character transitions were actually observed. About 34% of all possible transitions are valid — the remaining 66% are never observed in real Telugu text. At inference time, when the model predicts the next character, we check which transitions are valid given the previous character and apply a penalty of negative 10 to invalid transitions. This effectively zeroes out their probability. The key advantage is that this is entirely data-driven — no hand-coded rules — and it costs nothing during training. We validated it against the validation set and confirmed zero false rejections.

---

---

## SLIDE 14 — Dataset Description

**Slide Title:** Dataset — IIIT-HW-Telugu

**Content:**
- **Source:** CVIT (Centre for Visual Information Technology), IIIT Hyderabad
- **Dataset Name:** IIIT-HW-Telugu
- **Content:** Handwritten Telugu word-level images collected from multiple writers

- **Statistics:**
  | Split | Images | Purpose |
  |---|---|---|
  | Training | 88,534 | Model training |
  | Validation | 19,980 | Hyperparameter tuning, early stopping |
  | Test | 17,910 | Final evaluation (unseen, untouched) |
  | **Total** | **~126,413** | — |

- **Vocabulary:**
  - 91 unique characters (including vowels, consonants, vowel modifiers, Virama, numerals, special symbols)
  - Character distribution is heavily imbalanced — some characters appear thousands of times, others appear rarely

- **Image Properties:**
  - Variable dimensions (different word lengths and writing sizes)
  - Grayscale or binarized
  - Contains significant noise, artifacts, and writing variability

- **Benchmark Status:**
  - Primary benchmark dataset for handwritten Telugu recognition research
  - Used in Dutta et al. (2018), Deshpande et al. (2021), and other published works

**Visual Suggestion:**
- Bar chart showing train/val/test split sizes
- Grid of 6–8 sample handwritten Telugu word images from the dataset
- Pie chart or histogram showing character frequency distribution

**Speaker Notes:**
We use the IIIT-HW-Telugu dataset from the Centre for Visual Information Technology at IIIT Hyderabad. It contains approximately 126,000 handwritten Telugu word images collected from multiple writers. The dataset is split into roughly 88,500 training images, 20,000 validation images, and 18,000 test images. The vocabulary consists of 91 unique characters. This is the standard benchmark dataset for Telugu HTR research — the same dataset used by Dutta et al. and Deshpande et al. in their published works. The test set of 17,910 images was kept completely untouched until final evaluation.

---

---

## SLIDE 15 — Data Preprocessing Pipeline

**Slide Title:** Data Preprocessing Pipeline

**Content:**
- **Step 1 — Image Loading:**
  - Load raw images in grayscale mode
  - Handle corrupted or unreadable images gracefully (skip with logging)

- **Step 2 — Resize & Pad:**
  - Target size: 64 (height) × 512 (width)
  - Maintain aspect ratio: resize height to 64, scale width proportionally
  - If width < 512: right-pad with white (value=255) to reach 512
  - If width > 512: resize width to 512 (rare cases)

- **Step 3 — Normalization:**
  - Pixel values scaled from [0, 255] → [0.0, 1.0] (divide by 255)
  - Then normalized with ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225) per channel
  - Applied after channel replication (1 → 3 channels)

- **Step 4 — Label Encoding:**
  - Build character-to-index mapping from training vocabulary (91 characters)
  - Encode each label string as a sequence of integer indices
  - Add `<SOS>` (index 1) and `<EOS>` (index 2) tokens for AR decoder
  - `<PAD>` (index 0) for padding to max label length
  - `<BLANK>` (index 90) for CTC

- **Step 5 — Data Augmentation (Training Only):**
  - Random slight rotation, affine transforms
  - Gaussian noise injection
  - Contrast/brightness jittering

**Visual Suggestion:**
- Horizontal pipeline diagram showing each step as a box with arrows:
  - Raw Image → Grayscale → Resize/Pad → Normalize → Tensor
- Show a sample image at each stage of transformation
- Below: label encoding example with Telugu word → character indices

**Speaker Notes:**
Our preprocessing pipeline has five stages. First, we load images in grayscale. Then we resize them to a fixed height of 64 pixels while maintaining aspect ratio, padding the width to 512 pixels. We normalize using ImageNet statistics after replicating the single channel to three channels. Labels are encoded as integer sequences with special tokens for start-of-sequence, end-of-sequence, padding, and CTC blank. During training, we apply light data augmentation including random rotation, affine transforms, and noise injection to improve generalization.

---

---

## SLIDE 16 — Training Configuration

**Slide Title:** Training Configuration

**Content:**

| Parameter | CTC Baseline | AR Transformer |
|---|---|---|
| **GPU** | NVIDIA RTX 3090 Ti (24 GB) | NVIDIA RTX 3090 Ti (24 GB) |
| **Epochs** | 50 | 50 |
| **Batch Size** | 64 | 256 |
| **Optimizer** | AdamW (weight_decay=0.01) | AdamW (weight_decay=0.01) |
| **Learning Rate** | 1e-3 (peak) | 3e-4 (peak) |
| **LR Schedule** | OneCycleLR | Warmup (5 epochs) + Cosine Annealing |
| **Loss** | CTC Loss | Joint: 0.3 × CTC + 0.7 × CE |
| **Label Smoothing** | N/A | 0.1 |
| **Precision** | Mixed (fp16) | Mixed (fp16) |
| **Gradient Clipping** | max_norm = 5.0 | max_norm = 5.0 |
| **Parameters** | 14,115,483 | 17,302,518 |
| **Training Time** | 3h 22m | 2h 15m |
| **Workers** | 4 DataLoader workers | 4 DataLoader workers |

- **Key Notes:**
  - Mixed precision (fp16) via PyTorch AMP reduces memory usage by ~40% and speeds up training
  - Gradient clipping at 5.0 prevents exploding gradients, especially important for Transformer training
  - AR model trains faster despite more parameters due to larger batch size (256 vs. 64) enabled by efficient memory management
  - OneCycleLR for CTC: single cycle from low LR → peak → low; proven effective for CTC convergence
  - Warmup + Cosine for AR: gradual warmup prevents unstable early training in Transformers

**Visual Suggestion:**
- Side-by-side comparison table (CTC vs. AR) with colour-coded rows
- GPU utilization and memory usage chart
- Training time comparison bar chart

**Speaker Notes:**
Here are our training configurations. The CTC model was trained for 50 epochs with batch size 64 using OneCycleLR scheduling with a peak learning rate of 1e-3. The AR model was also trained for 50 epochs but with a larger batch size of 256 — enabled by efficient mixed-precision training — using warmup plus cosine annealing with a lower peak learning rate of 3e-4. Both models use AdamW optimizer with weight decay, mixed-precision training, and gradient clipping at 5.0. Interestingly, despite having more parameters, the AR model trains faster — 2 hours 15 minutes versus 3 hours 22 minutes — thanks to the larger batch size and parallelizable attention computations. All training was done on a single NVIDIA RTX 3090 Ti with 24 GB VRAM.

---

---

## SLIDE 17 — Training Progress — CTC

**Slide Title:** Training Progress — CTC Baseline

**Content:**
- **Loss Curve:**
  - Starting CTC loss: ~5.0 (random initialization)
  - Rapid convergence in first 10 epochs → loss drops to ~0.5
  - Gradual refinement from epoch 10–50 → final loss: ~0.15
  - Smooth convergence with OneCycleLR — no sudden jumps or instability

- **Validation CER Curve:**
  - Starting CER: ~85% (essentially random predictions)
  - Drops below 20% by epoch 5
  - Drops below 5% by epoch 20
  - Best validation CER: ~3.8% (achieved around epoch 45)
  - No significant overfitting — train and val curves track closely

- **Key Observations:**
  - OneCycleLR provides excellent convergence dynamics for CTC
  - ImageNet pretrained encoder gives strong initialization — model learns meaningful features quickly
  - BiLSTM effectively captures sequential dependencies in the visual feature sequence
  - No early stopping triggered — model continued to improve throughout all 50 epochs

**Visual Suggestion:**
- Two-panel plot:
  - Left: CTC Loss vs. Epoch (training loss in blue, validation loss in orange)
  - Right: CER (%) vs. Epoch (training CER in blue, validation CER in orange)
- Annotate key milestones (e.g., "CER < 5% at epoch 20")
- X-axis: Epochs (0–50), Y-axis: Loss / CER

**Speaker Notes:**
Let me walk you through the CTC training progress. The model starts with essentially random predictions at 85% CER. Within the first 5 epochs, CER drops below 20%, showing that the pretrained ResNet encoder gives the model a strong head start. By epoch 20, we're below 5% CER. The model continues to refine through all 50 epochs, reaching about 3.8% on validation. The loss curve shows smooth convergence with no instability, which we attribute to the OneCycleLR scheduler. Importantly, the training and validation curves track closely, indicating no significant overfitting.

---

---

## SLIDE 18 — Training Progress — AR

**Slide Title:** Training Progress — AR Transformer (Joint Training)

**Content:**
- **Joint Loss Curve:**
  - Starting loss: ~6.5 (combined CTC + CE)
  - Warmup phase (epochs 1–5): gradual increase in learning rate; loss decreases steadily
  - Post-warmup (epochs 5–50): cosine annealing smoothly reduces LR; loss converges to ~0.25
  - Final joint loss: ~0.20

- **Validation CER Curve (Autoregressive Decoding):**
  - Starting CER: ~90% (random autoregressive predictions)
  - Drops below 30% by epoch 10
  - Drops below 8% by epoch 25
  - Best validation CER: ~4.8% (achieved around epoch 48)
  - Slight gap between train and val CER — minor overfitting in later epochs

- **CTC Auxiliary Head (During Joint Training):**
  - CTC head CER also improves alongside AR head
  - Confirms that joint training benefits the shared encoder

- **Key Observations:**
  - Warmup is critical for Transformer stability — without it, training diverges
  - AR model converges more slowly than CTC (autoregressive loop is harder to optimize)
  - Final AR CER (~4.8%) is higher than CTC CER (~3.8%) — the CTC head outperforms in this scenario
  - Label smoothing (0.1) prevents overconfident predictions, as seen in well-calibrated logits

**Visual Suggestion:**
- Two-panel plot (similar to Slide 17):
  - Left: Joint Loss vs. Epoch with warmup phase shaded
  - Right: CER (%) vs. Epoch for AR decoding
- Annotate warmup region and cosine decay region
- Add a small inset plot showing CTC auxiliary CER during joint training

**Speaker Notes:**
The AR Transformer training shows a different profile. We start with a warmup phase for the first 5 epochs — this is critical because Transformers are notoriously sensitive to learning rate in early training. After warmup, the cosine annealing schedule smoothly reduces the learning rate. The model converges more slowly than CTC — reaching below 8% CER only by epoch 25. The best validation CER is about 4.8%, which is notably higher than the CTC model's 3.8%. This is an interesting finding that we will discuss further. The joint training does benefit the shared encoder, as confirmed by the auxiliary CTC head also improving during training.

---

---

## SLIDE 19 — Evaluation Methodology

**Slide Title:** Evaluation Methodology

**Content:**
- **Primary Metrics:**
  - **Character Error Rate (CER):** Edit distance between predicted and ground-truth character sequences, normalized by ground-truth length
    ```
    CER = (Substitutions + Insertions + Deletions) / Total Ground-Truth Characters × 100%
    ```
  - **Word Error Rate (WER):** Percentage of words with at least one character error
    ```
    WER = Words with any error / Total Words × 100%
    ```

- **Statistical Rigor:**
  - **95% Confidence Intervals** via bootstrap resampling (10,000 iterations) on CER
  - Ensures reported improvements are statistically significant, not due to random variation

- **Subgroup Analysis:**
  - **Compound Character CER:** Error rate on words containing compound characters (conjuncts with Virama)
  - **Simple Character CER:** Error rate on words containing only simple characters (no conjuncts)
  - Reveals whether the model handles complex ligatures differently

- **Speed Benchmark:**
  - Inference speed measured in milliseconds per image on RTX 3090 Ti
  - Covers: forward pass + decoding (greedy or beam)

- **Evaluation Protocol:**
  - Test set: 17,910 images — completely unseen during training and hyperparameter tuning
  - Single evaluation run (deterministic greedy decoding)
  - Beam search: beam width = 5, length penalty = 1.0

**Visual Suggestion:**
- Metric formulas in styled boxes
- Flowchart of evaluation pipeline: Load model → Process test images → Compute CER/WER → Bootstrap CI → Subgroup analysis
- Icons for each metric type

**Speaker Notes:**
Our evaluation methodology is designed for rigour and reproducibility. The primary metrics are Character Error Rate and Word Error Rate. CER measures edit distance at the character level — counting substitutions, insertions, and deletions. WER measures the percentage of words with any error. We compute 95% confidence intervals using bootstrap resampling with 10,000 iterations to ensure our results are statistically robust. We also perform subgroup analysis, separating performance on compound characters versus simple characters, and benchmark inference speed in milliseconds per image. All evaluation is on the held-out test set of 17,910 images that was never used during training or tuning.

---

---

## SLIDE 20 — Results: Ablation Table

**Slide Title:** Results — Ablation Study (Test Set: 17,910 Images)

**Content:**

| Model Configuration | CER (%) | WER (%) | 95% CI (CER) | Compound CER (%) | Simple CER (%) | Speed (ms/img) |
|---|---|---|---|---|---|---|
| **CTC Baseline** | **3.91** | **24.80** | [3.79, 4.04] | 3.70 | 4.29 | 0.23 |
| AR (unconstrained) | 4.89 | 29.83 | [4.75, 5.01] | 4.61 | 5.38 | 0.61 |
| AR + Telugu Constraint | 4.99 | 29.71 | [4.85, 5.14] | 4.68 | 5.54 | 1.03 |
| AR + Constraint + Beam(5) | 4.85 | 29.49 | [4.72, 5.00] | 4.57 | 5.35 | 60.6 |

- **Key Takeaways from Ablation:**
  - CTC Baseline is the best-performing model at 3.91% CER
  - AR models achieve ~4.85–4.99% CER — approximately 1% higher than CTC
  - Telugu constraint mask reduces WER from 29.83% → 29.71% (marginal but consistent)
  - Beam search (width=5) provides additional improvement: CER 4.99% → 4.85%, WER 29.71% → 29.49%
  - All models perform better on compound characters than simple characters (counter-intuitive finding)
  - Speed-accuracy tradeoff: CTC is 260× faster than beam search

**Visual Suggestion:**
- Full-width table with the best result (CTC, 3.91%) highlighted in green
- Grouped bar chart below showing CER comparison across all 4 configurations
- Small speed comparison bar chart (log scale) on the side

**Speaker Notes:**
Here are our complete ablation results on the test set. The CTC baseline achieves the best character error rate at 3.91%. The autoregressive models range from 4.85% to 4.99% CER. Our Telugu constraint mask provides a marginal improvement in word error rate — from 29.83% to 29.71% — and beam search adds further improvement, bringing CER down to 4.85%. An interesting finding is that all models perform better on compound characters than simple characters, which we'll discuss on the next slide. The speed differences are dramatic — CTC takes just 0.23 milliseconds per image, while beam search with constraints takes 60.6 milliseconds.

---

---

## SLIDE 21 — Results: Analysis

**Slide Title:** Results — Detailed Analysis

**Content:**
- **Why CTC Outperforms AR:**
  - CTC's conditional independence assumption acts as implicit regularization — prevents the model from "overthinking" character dependencies that may not exist in short words
  - BiLSTM in CTC head provides sufficient sequential context for Telugu word recognition
  - AR decoder's autoregressive loop can propagate errors — one wrong character affects all subsequent predictions (error cascading)
  - Dataset word lengths are relatively short (average ~5–8 characters) — AR's context modelling advantage is limited

- **Compound vs. Simple Character Performance:**
  - Counter-intuitive: Compound CER (3.70%) < Simple CER (4.29%) for CTC
  - Explanation: Compound characters (with Virama) tend to appear in common, frequently-occurring words → model sees more training examples
  - Simple characters in isolation may appear in rare or unusual words with less training coverage

- **Constraint Mask Impact:**
  - Marginal improvement in WER (0.12% absolute) but slight increase in CER (0.10%)
  - The mask correctly blocks invalid transitions but cannot fix errors in the encoder features
  - Primary value: ensures output consistency and prevents linguistically impossible sequences
  - Practical value higher than the numbers suggest — eliminates a category of errors entirely

- **Beam Search Effect:**
  - Beam width 5 improves CER by 0.14% and WER by 0.22% over constrained greedy
  - Comes at significant speed cost: 1.03 ms → 60.6 ms (59× slower)
  - Diminishing returns beyond beam width 5

**Visual Suggestion:**
- Error analysis breakdown chart: types of errors (substitution, insertion, deletion) as stacked bar
- Confusion matrix heatmap for most commonly confused character pairs
- Scatter plot: word length vs. CER (showing CTC outperforms AR on shorter words)

**Speaker Notes:**
Let's analyse why the CTC baseline outperforms the AR model. First, CTC's conditional independence assumption actually works as a regularizer for short words — the model doesn't overfit to character dependencies that may not help for 5-to-8 character words. Second, the autoregressive loop can cascade errors — if the model predicts one wrong character, it conditions all future predictions on that error. The compound versus simple character finding is interesting — compound characters actually have lower error rates because they tend to appear in common, frequently-seen words. The constraint mask's impact is modest in terms of CER numbers, but its practical value is significant — it completely eliminates a category of linguistically impossible outputs.

---

---

## SLIDE 22 — Comparison with Published Work

**Slide Title:** Comparison with Published Work

**Content:**

| Work | Year | Method | Dataset | CER (%) | Relative Improvement |
|---|---|---|---|---|---|
| Dutta et al. | 2018 | MDLSTM + CTC | IIIT-HW-Telugu | ~15.0 | — (baseline) |
| Deshpande et al. | 2021 | CNN + LSTM + CTC | IIIT-HW-Telugu | ~10.2 | 32% over Dutta |
| **Ours (CTC)** | **2026** | **ResNet-18 + BiLSTM + CTC** | **IIIT-HW-Telugu** | **3.91** | **74% over Dutta, 62% over Deshpande** |
| **Ours (AR + Beam)** | **2026** | **ResNet-18 + Transformer + Beam** | **IIIT-HW-Telugu** | **4.85** | **68% over Dutta, 52% over Deshpande** |

- **Key Comparisons:**
  - Our best model (CTC, 3.91% CER) represents a **62% relative reduction** in CER over the previous best (10.2%)
  - Even our AR model with beam search (4.85%) significantly outperforms all prior work
  - Improvement attributable to:
    - Modern pretrained visual encoder (ResNet-18 with ImageNet weights vs. training from scratch)
    - Stride patching to preserve spatial resolution
    - Better training recipes (OneCycleLR, mixed precision, gradient clipping)
    - Larger effective capacity (14M–17M parameters vs. smaller models in prior work)

- **Caveat:**
  - Direct comparison assumes identical test sets and evaluation protocols
  - Prior works may have used different train/test splits of the same dataset

**Visual Suggestion:**
- Bar chart comparing CER across years/methods (2018 → 2021 → 2026), showing dramatic improvement
- Use a downward arrow motif to emphasize reduction in error rate
- Highlight "62% relative improvement" prominently
- Timeline visualization at the bottom

**Speaker Notes:**
When we compare with published work on the same IIIT-HW-Telugu dataset, the improvement is striking. Dutta et al. in 2018 achieved about 15% CER. Deshpande et al. in 2021 improved this to about 10.2% CER — a 32% relative improvement. Our CTC baseline achieves 3.91% CER — a 62% relative improvement over Deshpande and 74% over Dutta. Even our AR model at 4.85% CER significantly outperforms all prior published results. The key factors driving this improvement are the pretrained ResNet-18 encoder, our stride patching technique, and modern training recipes including OneCycleLR, mixed precision, and gradient clipping.

---

---

## SLIDE 23 — Key Findings & Contributions

**Slide Title:** Key Findings & Contributions

**Content:**
- **Contribution 1 — State-of-the-Art Results:**
  - Achieved 3.91% CER on IIIT-HW-Telugu — best reported result, 62% improvement over prior SOTA
  - Demonstrates that modern architectures with proper engineering can dramatically advance Telugu HTR

- **Contribution 2 — Novel Telugu Linguistic Constraint Mask:**
  - First data-driven approach to enforce Telugu character transition constraints
  - Zero training overhead — applied only at inference
  - Generalizable to other Indic scripts (Hindi, Tamil, Kannada, etc.)
  - Ensures linguistically valid output sequences

- **Contribution 3 — Comprehensive Architecture Comparison:**
  - First systematic comparison of CTC vs. AR Transformer decoding for Telugu HTR
  - Finding: CTC with BiLSTM outperforms AR Transformer on short Telugu words
  - Joint CTC + CE training provides complementary learning signals

- **Contribution 4 — Detailed Analysis:**
  - Compound vs. simple character analysis reveals training frequency effects
  - Speed-accuracy tradeoff analysis across all configurations
  - 95% confidence intervals ensure statistical validity
  - Error cascading analysis in autoregressive decoding

- **Contribution 5 — Reproducible Methodology:**
  - Complete training recipes, hyperparameters, and evaluation protocols documented
  - Built on publicly available dataset (IIIT-HW-Telugu)

**Visual Suggestion:**
- Five contribution cards arranged vertically, each with a distinct icon and colour
- Star/medal icon for SOTA result
- Light bulb icon for novel constraint mask
- Comparison icon for architecture comparison
- Magnifying glass for analysis
- Document icon for reproducibility

**Speaker Notes:**
Let me summarize our five key contributions. First, we achieved state-of-the-art results — 3.91% CER, a 62% improvement over the previous best. Second, we developed a novel data-driven Telugu linguistic constraint mask that is the first of its kind and generalizable to other scripts. Third, we conducted the first systematic comparison of CTC versus autoregressive Transformer decoding for Telugu, finding that CTC outperforms on short words. Fourth, we provided detailed analyses including compound versus simple character performance, speed benchmarks, and confidence intervals. Fifth, our complete methodology is documented and reproducible.

---

---

## SLIDE 24 — Limitations

**Slide Title:** Limitations

**Content:**
- **Limitation 1 — CTC Outperforming AR:**
  - The autoregressive Transformer model underperforms the simpler CTC baseline
  - Likely causes: insufficient decoder capacity for the dataset size, or short average word length limiting the benefit of explicit context modelling
  - More extensive hyperparameter search or larger decoders might close the gap

- **Limitation 2 — Marginal Constraint Mask Impact:**
  - The Telugu constraint mask provides only marginal quantitative improvement (0.12% WER reduction)
  - Most predictions already satisfy valid transitions even without the mask
  - The mask is more of a safety net than a primary accuracy driver

- **Limitation 3 — Single Dataset Evaluation:**
  - All experiments conducted on IIIT-HW-Telugu only
  - Cannot guarantee generalization to other Telugu handwriting datasets or real-world documents
  - Cross-dataset evaluation would strengthen the findings

- **Limitation 4 — Word-Level Recognition Only:**
  - Current system recognizes individual words, not full sentences or paragraphs
  - Does not handle line segmentation, word segmentation, or document layout analysis
  - End-to-end document recognition requires additional pipeline components

- **Limitation 5 — Speed-Accuracy Tradeoff:**
  - Beam search (best AR performance) is 260× slower than CTC greedy decoding
  - May not be suitable for real-time applications requiring fast throughput

**Visual Suggestion:**
- Five limitation boxes in a muted colour scheme (grey/amber)
- Each with a caution/warning icon
- Arrows from each limitation pointing to potential solutions (link to Future Scope)

**Speaker Notes:**
We acknowledge several limitations. First, our autoregressive model underperforms the CTC baseline — this is somewhat unexpected and likely due to short average word lengths in the dataset limiting the AR model's advantage. Second, the constraint mask's quantitative impact is marginal, though it provides important qualitative guarantees. Third, we evaluated on a single dataset, so generalization is not guaranteed. Fourth, our system works at the word level — it doesn't handle full document recognition. And fifth, the beam search configuration is too slow for real-time applications. Each of these limitations motivates future research directions.

---

---

## SLIDE 25 — Future Scope

**Slide Title:** Future Scope

**Content:**
- **Sentence/Line-Level Recognition:**
  - Extend from word-level to line-level and paragraph-level recognition
  - Integrate word segmentation and line segmentation modules
  - Enables full document digitization pipeline

- **Larger Transformer Models:**
  - Scale up decoder capacity (6–8 layers, wider dimensions)
  - Explore Vision Transformer (ViT) as encoder replacing CNN
  - Pretrain encoder-decoder on large unlabelled Telugu text images (self-supervised learning)

- **Language Model Integration:**
  - Integrate a Telugu language model (n-gram or neural) for post-processing
  - Rescore beam search hypotheses using language model probabilities
  - Could significantly reduce WER by leveraging word-level language statistics

- **Cross-Script Transfer Learning:**
  - Apply the architecture and constraint mask framework to other Indic scripts:
    - Hindi (Devanagari), Tamil, Kannada, Malayalam, Bengali
  - Multilingual model that shares visual encoder across scripts

- **Real-World Deployment:**
  - Mobile-optimized models (quantization, pruning, knowledge distillation)
  - Web application for handwritten Telugu digitization
  - API integration for government document processing

- **Advanced Constraint Mechanisms:**
  - Extend bigram mask to trigram or higher-order n-gram constraints
  - Learned constraint integration: train the constraint jointly with the model
  - Context-dependent constraints based on word position

**Visual Suggestion:**
- Roadmap-style timeline visualization (short-term → medium-term → long-term)
- Icons for each future direction
- Branching paths showing different research directions

**Speaker Notes:**
Looking ahead, there are several exciting directions. First, extending to sentence and line-level recognition for full document digitization. Second, scaling up with larger Transformer models and self-supervised pretraining on unlabelled Telugu text. Third, integrating a Telugu language model to rescore predictions and reduce word error rate. Fourth, applying our framework to other Indic scripts through cross-script transfer learning. Fifth, deploying optimized models on mobile devices and web platforms. And sixth, advancing the constraint mechanism to trigram-level or even learned constraints. Each of these directions has the potential to push Telugu HTR closer to practical, real-world deployment.

---

---

## SLIDE 26 — Conclusion

**Slide Title:** Conclusion

**Content:**
- **Summary of Achievements:**
  - Developed a hybrid CNN-Transformer architecture for handwritten Telugu text recognition
  - Achieved **3.91% CER** on the IIIT-HW-Telugu benchmark — **best reported result**
  - **62% relative improvement** over the previous state-of-the-art (Deshpande et al., 2021: 10.2% CER)

- **Novel Contribution:**
  - Introduced the **first data-driven Telugu linguistic constraint mask** for enforcing valid character transitions during inference
  - Zero training overhead, generalizable to any script

- **Technical Insights:**
  - CTC with BiLSTM outperforms autoregressive Transformer on short Telugu words (3.91% vs. 4.85%)
  - Joint CTC + CE training provides complementary supervision signals
  - Stride patching in ResNet-18 is critical for preserving spatial resolution
  - Pretrained ImageNet weights significantly accelerate convergence

- **Impact:**
  - Brings handwritten Telugu recognition from research-grade (10%+ CER) toward practical deployment quality (<5% CER)
  - Provides a reproducible baseline and framework for future Telugu and Indic script HTR research
  - The linguistic constraint framework is a reusable contribution applicable beyond this specific project

- **Final Statement:**
  > *This project demonstrates that combining modern deep learning architectures with data-driven linguistic constraints can achieve near-practical accuracy for handwritten recognition of complex scripts like Telugu.*

**Visual Suggestion:**
- Clean, impactful layout with key numbers in large font:
  - "3.91% CER" in large, bold text
  - "62% improvement" with upward arrow
  - "126,413 images" dataset size
- Timeline showing progress: 15% → 10.2% → 3.91%
- University/project branding at the bottom

**Speaker Notes:**
To conclude, our project has successfully developed a hybrid CNN-Transformer system for handwritten Telugu text recognition that achieves the best reported results on the standard benchmark — 3.91% character error rate, a 62% improvement over the previous state of the art. Our novel Telugu linguistic constraint mask is the first data-driven approach to enforcing valid character transitions, and it requires zero additional training cost. We've shown that the simpler CTC model can outperform the autoregressive Transformer on short words, and that stride patching and pretrained weights are critical engineering decisions. This work brings Telugu HTR closer to practical deployment and provides a reproducible framework for future research on complex Indic scripts.

---

---

## SLIDE 27 — References

**Slide Title:** References

**Content:**
1. Dutta, K., Krishnan, P., Mathew, M., & Jawahar, C. V. (2018). "Towards Accurate Handwritten Word Recognition for Hindi and Bangla." *13th IAPR International Workshop on Document Analysis Systems (DAS)*. IIIT Hyderabad. [IIIT-HW-Telugu Dataset]

2. Deshpande, A., Kulkarni, A., & Ramakrishnan, G. (2021). "Handwritten Text Recognition for Indic Scripts." *International Conference on Document Analysis and Recognition (ICDAR)*.

3. Shi, B., Bai, X., & Yao, C. (2017). "An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition." *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 39(11), 2298–2304.

4. Li, M., Lv, T., Chen, J., et al. (2023). "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models." *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(11), 13094–13102.

5. Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks." *Proceedings of the 23rd International Conference on Machine Learning (ICML)*, 369–376.

6. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). "Attention Is All You Need." *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998–6008.

7. Bluche, T., & Messina, R. (2017). "Gated Convolutional Recurrent Neural Networks for Multilingual Handwriting Recognition." *14th IAPR International Conference on Document Analysis and Recognition (ICDAR)*, 646–651.

8. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778.

9. Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization." *International Conference on Learning Representations (ICLR)*.

10. Smith, L. N. (2019). "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates." *Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications*, SPIE, 11006.

**Visual Suggestion:**
- Clean numbered list in a slightly smaller font
- IEEE/ACM citation format
- Two-column layout if needed to fit on one slide
- References 1 and 2 highlighted as primary Telugu HTR works

**Speaker Notes:**
Here are our key references. References 1 and 2 are the primary prior works on Telugu handwriting recognition that we compare against. References 3 through 7 are foundational works in text recognition and sequence modelling that our architecture builds upon. Reference 8 is the ResNet paper, and references 9 and 10 cover the optimizer and learning rate scheduling techniques we used.

---

---

## SLIDE 28 — Thank You / Q&A

**Slide Title:** Thank You

**Content:**
- **Project Title:** Grammar-Aware Handwritten Telugu Text Recognition Using CNN-Transformer Architecture with Data-Driven Linguistic Constraints
- **Key Result:** 3.91% CER — 62% improvement over previous state-of-the-art
- **Novel Contribution:** Data-Driven Telugu Linguistic Constraint Mask

- **Team:** [Insert Names]
- **Guide:** [Insert Guide Name]
- **Contact:** [Insert Email/Contact Info]

- *Thank you for your attention. We welcome your questions.*

- **Q&A**

**Visual Suggestion:**
- Clean, professional slide with centred "Thank You" in large font
- Subtle background with faded Telugu script motif
- Key result (3.91% CER) displayed prominently
- QR code linking to project repository or demo (if available)
- University/department logo

**Speaker Notes:**
Thank you for your time and attention. We're happy to answer any questions you may have about the architecture, training methodology, results, or future directions. We believe this project makes a meaningful contribution to handwritten Telugu text recognition and provides a strong foundation for future work on Indic script HTR systems.

---

---

## APPENDIX — Additional Slide Notes

### Suggested PPT Design Guidelines:
- **Colour Scheme:** Deep blue (#1a237e) + Teal (#00796b) + White + Light grey backgrounds
- **Font:** Heading — Montserrat Bold / Poppins Bold; Body — Open Sans / Roboto
- **Accent Colour:** Gold (#f9a825) for highlighting key results and novel contributions
- **Slide Layout:** 16:9 widescreen format
- **Animations:** Subtle fade-in for bullet points; no distracting transitions
- **Consistency:** Same header bar, footer (slide number + project title), and colour scheme on every slide

### Estimated Slide Count: 28 slides (matches requirement of 25–30)

### Presentation Duration: ~30–40 minutes (including Q&A)
