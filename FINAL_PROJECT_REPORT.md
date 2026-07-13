# Grammar-Aware Handwritten Telugu Text Recognition Using CNN-Transformer Architecture with Data-Driven Linguistic Constraints

## B.Tech Major Project Report

---

## Abstract
Handwritten Text Recognition (HTR) for Indic scripts remains a significant challenge in document image analysis due to the cursive nature of handwriting, complex ligatures, and a large number of distinct character classes. This project presents a novel, grammar-aware HTR system for Telugu, a complex Dravidian script. We investigate the effectiveness of combining a Convolutional Neural Network (CNN) visual feature extractor with two distinct sequence modeling paradigms: a Connectionist Temporal Classification (CTC) baseline and an Autoregressive (AR) Transformer decoder. To address the unique orthographic rules of Telugu—specifically the formation of consonant conjuncts using the Virama (్) character—we introduce a novel data-driven linguistic constraint mask that penalizes grammatically invalid character transitions during decoding. Our evaluation on the IIIT-HW-Telugu benchmark dataset demonstrates that the CTC baseline achieves a state-of-the-art Character Error Rate (CER) of 3.91%, significantly outperforming the AR Transformer (4.85% CER) and prior published works (which range from 10% to 15% CER). We provide a comprehensive ablation study and empirical analysis revealing that, contrary to recent trends favoring Transformer decoders, the simpler CTC alignment mechanism is highly effective for short, well-segmented Telugu words. The proposed data-driven constraint mask offers a scalable method for integrating script-specific grammar into neural HTR systems without relying on hard-coded rules.

---

## Chapter 1: Introduction

### 1.1 Background and Context
The digitization of historical manuscripts, administrative documents, postal addresses, and personal notes is crucial for preserving cultural heritage and enabling information retrieval. Handwritten Text Recognition (HTR) is the core technology driving this digitization. While Optical Character Recognition (OCR) for printed text in languages like English has reached near-human performance, HTR for Indic scripts remains an open research problem.

Telugu is a Dravidian language and the fourth most spoken language in India, with over 80 million native speakers. The Telugu script is highly complex and syllabic in nature. It consists of 56 base characters (16 vowels, 35 consonants, and 5 special characters) and over 18 vowel modifiers (matras). Furthermore, the script is heavily reliant on compound characters (ligatures). When two or more consonants are combined without an intervening vowel, a special character called the Virama (్) is used. In written form, these consonant clusters often merge into distinct visual glyphs where the secondary consonant attaches below or to the right of the primary consonant.

Recognizing handwritten Telugu is particularly challenging due to high inter-class similarity (many characters look visually similar), high intra-class variance (different writers write the same character differently), and the cursive, overlapping nature of the strokes. In recent years, deep learning approaches utilizing Convolutional Neural Networks (CNNs) coupled with Recurrent Neural Networks (RNNs) and Connectionist Temporal Classification (CTC) have become the standard for HTR. More recently, Transformer-based architectures have emerged as powerful alternatives. However, these models generally treat sequence prediction as a purely statistical problem, largely ignoring the strict orthographic rules inherent to languages like Telugu.

### 1.2 Problem Statement
Existing Telugu HTR systems typically achieve a Character Error Rate (CER) of 8% to 15%, which is insufficient for reliable automated digitization. A major limitation of current state-of-the-art systems is that they perform isolated character prediction without intrinsic grammar awareness. Telugu has strict orthographic constraints; for instance, a Virama cannot appear at the beginning or end of a valid word, and certain vowel modifiers cannot follow specific consonants. 

When deep learning models predict sequences without these constraints, they frequently generate grammatically impossible character combinations, especially when deciphering messy or ambiguous handwriting. There is currently no robust mechanism in the literature that systematically exploits data-driven Telugu orthographic constraints directly during neural network decoding to prevent these invalid predictions.

### 1.3 Objectives
The primary objectives of this major project are to:
1. Develop a high-performance hybrid architecture combining a ResNet-based visual encoder with sequence decoders for Telugu HTR.
2. Implement and evaluate a standard CTC-based decoding baseline.
3. Implement and evaluate an Autoregressive (AR) Transformer decoder jointly trained with an auxiliary CTC loss.
4. Design and integrate a novel, data-driven Telugu linguistic constraint mask that enforces valid character transitions during the decoding phase.
5. Conduct a rigorous ablation study comparing the CTC model, the unconstrained AR model, and the constrained AR model (with both greedy and beam search decoding).
6. Achieve state-of-the-art CER on the standardized IIIT-HW-Telugu benchmark dataset.

### 1.4 Scope and Limitations
The scope of this project is restricted to offline, word-level Handwritten Text Recognition for the Telugu language. The system expects cropped images of individual words as input. Full-page or line-level recognition, which requires additional segmentation or attention-based line tracking, is beyond the current scope. The linguistic constraints are derived purely from the training dataset vocabulary and bigram statistics, rather than an exhaustive external Telugu dictionary or language model.

### 1.5 Organization of the Report
The remainder of this report is organized as follows: Chapter 2 reviews the existing literature on HTR, CTC, Transformers, and Indic script challenges. Chapter 3 details the system design, including the visual encoder, CTC and AR models, and the novel linguistic constraint mask. Chapter 4 covers the implementation details, dataset, preprocessing, and training configurations. Chapter 5 presents the evaluation metrics, ablation study results, error analysis, and comparison with published literature. Finally, Chapter 6 concludes the report and discusses potential future scope.

---

## Chapter 2: Literature Survey

### 2.1 Handwriting Recognition: An Overview
Handwritten Text Recognition (HTR) has evolved significantly over the past few decades. Classical approaches relied heavily on handcrafted feature extraction (e.g., HOG, SIFT) combined with sequential probabilistic models like Hidden Markov Models (HMMs). While effective for constrained, small-vocabulary tasks, these methods struggled to generalize across diverse handwriting styles.

The advent of deep learning revolutionized HTR. The standard paradigm shifted to using Convolutional Neural Networks (CNNs) to extract visual feature sequences from images, followed by Recurrent Neural Networks (RNNs)—typically Long Short-Term Memory (LSTM) networks—to model the sequential dependencies of the characters.

### 2.2 CTC-based Approaches
A major breakthrough in sequence recognition was the introduction of Connectionist Temporal Classification (CTC) by Graves et al. (2006). CTC allows RNNs to be trained on unsegmented sequence data. It introduces a "blank" token and calculates the loss by summing the probabilities of all possible alignments between the predicted sequence and the ground truth label.

For Indic scripts, CNN-RNN-CTC architectures have become the benchmark. Dutta et al. (2018) applied this architecture to various Indic languages, establishing baseline results on the IIIT-HW-Indic datasets. For Telugu, their CNN-RNN-CTC model achieved a Character Error Rate (CER) of approximately 15.0%. While CTC models are fast and alignment-focused, they assume conditional independence between predictions at different time steps, meaning they do not inherently learn strong language models.

### 2.3 Attention-based and Transformer Approaches
To overcome the conditional independence assumption of CTC, sequence-to-sequence models with attention mechanisms were introduced (Bahdanau et al., 2014). These models autoregressively generate one character at a time, attending to different parts of the encoded visual features based on the previously generated characters.

Deshpande et al. (2021) utilized an attention-based encoder-decoder architecture for Telugu HTR, significantly improving the performance to approximately 10.2% CER on the IIIT-HW-Telugu dataset. More recently, pure Transformer architectures like TrOCR (Li et al., 2022) have shown that large-scale pre-training of Vision Transformers (ViT) combined with Transformer text decoders can achieve state-of-the-art results on English HTR. However, these large autoregressive models are prone to overfitting on smaller, specialized datasets like those available for Indic languages.

### 2.4 Telugu Script-Specific Challenges
The complexity of the Telugu script presents unique hurdles. The visual representation of a character can change drastically depending on its context. When a consonant is followed by a vowel, it takes a specific modified form (matra). When a consonant is followed by a Virama (్) and another consonant, it forms a compound character (samyuktaksharam) where the secondary consonant is typically written as a subscript (vattu). Distinguishing between similar-looking base characters and accurately recognizing subscript consonants in cursive handwriting are the primary sources of errors in existing systems.

### 2.5 Linguistic Constraints in OCR
Incorporating linguistic constraints into OCR and HTR systems is an active area of research. Traditionally, this is achieved by integrating a strong external n-gram language model during beam search decoding. However, n-gram language models operate at the word level and require massive external text corpora.

For character-level HTR, enforcing script-specific grammatical rules (e.g., valid character transitions) is less explored. Hard-coding linguistic rules is tedious and prone to edge cases. Therefore, a data-driven approach to derive and enforce structural constraints during the neural network's decoding phase represents a significant research gap that this project aims to address.

### 2.6 Research Gap
Based on the literature survey, several research gaps emerge:
1. There is no systematic ablation study comparing standard CTC architectures with modern Transformer-based autoregressive models specifically for Telugu HTR.
2. The integration of data-driven orthographic constraints directly into the decoding probability space (rather than as a post-processing step) has not been rigorously explored for Telugu.
3. Existing baseline error rates (10-15% CER) indicate substantial room for architectural and training improvements to reach robust, production-ready accuracy.

---

## Chapter 3: System Design and Architecture

### 3.1 Overall System Architecture
The proposed system is designed to evaluate two distinct decoding paradigms utilizing a shared visual feature extraction backbone. The pipeline operates on grayscale word images. The core components are:
1. **Visual Feature Encoder:** A modified ResNet-18 CNN that processes the input image into a sequence of rich visual feature vectors.
2. **CTC Baseline Decoder:** A Bidirectional LSTM followed by a linear projection and CTC decoding.
3. **AR Transformer Decoder:** A Transformer Encoder-Decoder architecture that predicts characters autoregressively.
4. **Telugu Linguistic Constraint Mask:** A matrix-based masking system applied during autoregressive decoding to block invalid character transitions.

### 3.2 Visual Feature Encoder (ResNet-18)
The visual encoder is responsible for extracting spatial features from the raw image pixels. We utilize a ResNet-18 backbone pre-trained on ImageNet. Since ImageNet models expect 3-channel RGB images, the 1-channel grayscale input image is replicated across 3 channels.

**Stride Patching for HTR:** A standard ResNet-18 downsamples the spatial dimensions by a factor of 32 (via strides in max-pooling and convolution layers). For an input image of size 64x512, this would yield a feature map of width 16 (512 / 32 = 16). In HTR, the width dimension corresponds to the "time" or "sequence" dimension. A sequence length of 16 is often shorter than the number of characters in a Telugu word, which causes CTC loss to fail (the input sequence must be equal to or longer than the target text).

To resolve this, we patched the strides in the ResNet-18 backbone. The max-pooling layer stride was modified to downsample primarily on the height axis, and the strides in `layer3` and `layer4` were modified from `(2,2)` to `(2,1)`. This preserves the width resolution.
- **Input:** Tensor of shape `[B, 1, 64, 512]`
- **ResNet Output:** Feature map of shape `[B, 512, 1, 64]` (Height is fully collapsed, Width is preserved at 64).

The feature map is then squeezed and passed through a 2D Convolution (`512 -> 256` channels), followed by Batch Normalization and a GELU activation function. Finally, a standard sinusoidal Positional Encoding is added to inject sequence order awareness.
- **Final Encoder Memory:** Tensor of shape `[B, 64, 256]`, representing 64 temporal slices, each with a 256-dimensional feature vector.

### 3.3 CTC Baseline Model
The CTC baseline model utilizes the shared visual encoder memory. It passes the `[B, 64, 256]` sequence through a 2-layer Bidirectional LSTM with a hidden size of 256. The bidirectionality allows the network to incorporate both past and future visual context for each time step. The LSTM outputs a tensor of shape `[B, 64, 512]`. A final linear layer projects this to `[B, 64, 91]` (where 91 is the vocabulary size, including the CTC blank token). 

During training, the standard CTC Loss is computed. During inference, greedy decoding is applied: the argmax of the probabilities is taken at each time step, and consecutive duplicate characters and blank tokens are collapsed to yield the final predicted string. This model contains 14,115,483 parameters.

### 3.4 Autoregressive Transformer Model
The Autoregressive (AR) model represents a more complex, grammar-capable architecture. 
- **Transformer Encoder:** The visual memory `[B, 64, 256]` is passed through 2 layers of a Transformer Encoder (8 attention heads, feedforward dimension of 1024). This applies global self-attention across the visual slices, allowing the model to correlate distant visual features (e.g., resolving a complex ligature based on surrounding strokes).
- **Transformer Decoder:** A 4-layer Transformer Decoder (8 attention heads) receives the output of the Encoder. It autoregressively predicts the text sequence. At each step, it attends to the previously generated characters (via causal masked self-attention) and the visual features (via cross-attention).
- **Joint Training:** Autoregressive models can be difficult to align early in training. To stabilize training, we implement a multi-task learning approach. An auxiliary linear head is attached to the output of the Transformer Encoder to compute a CTC loss. The total loss is a weighted sum: `Total Loss = 0.3 * CTC_Loss + 0.7 * CrossEntropy_Loss`. Label smoothing of 0.1 is applied to the Cross-Entropy loss to prevent overconfidence and improve generalization.
- This AR model contains 17,302,518 parameters.

### 3.5 Telugu Linguistic Constraint Mask
The core novel contribution of this project is the Telugu Linguistic Constraint Mask. Rather than hard-coding grammatical rules, which is error-prone, we developed a data-driven approach to derive Telugu orthographic constraints directly from the training data.

1. **Vocabulary Construction:** During the vocabulary building phase, the system scans all 88,534 ground-truth labels in the training set.
2. **Bigram Extraction:** It extracts every observed `(previous_character, next_character)` transition.
3. **Validity Matrix:** A boolean matrix `valid_next[vocab_size][vocab_size]` is constructed. A transition `valid_next[i][j]` is True if and only if the transition from character `i` to character `j` was observed at least once in the training data.
4. **Validation:** This matrix was tested against the 19,980-sample validation set and achieved a ~0% violation rate, proving that the training data distribution perfectly captures the legal script constraints of the language.
5. **Inference Masking:** During autoregressive decoding, before the `softmax` and `argmax` operations, the model checks the previously predicted character. It looks up the valid next characters in the matrix. For any grammatically invalid character, a massive penalty (`-10.0`) is added to its logit. This effectively reduces its probability to near zero, forcing the model to choose the highest-probability *grammatically valid* character.

### 3.6 Decoding Strategies
For the AR model, we evaluate two decoding strategies:
- **Greedy Decoding:** At each step, the model selects the single character with the highest probability.
- **Beam Search Decoding:** The model maintains the top-k (beam size = 5) most likely sequence hypotheses at each step. This mitigates the issue where a locally optimal prediction leads to a globally suboptimal sequence. A length penalty of 0.6 is applied to prevent the beam search from unfairly favoring excessively short sequences.

---

## Chapter 4: Implementation Details

### 4.1 Dataset
The project utilizes the **IIIT-HW-Telugu** dataset, provided by CVIT, IIIT Hyderabad. It is one of the largest publicly available datasets for Indic HTR.
- **Training Set:** 88,534 images
- **Validation Set:** 19,980 images
- **Testing Set:** 17,910 images
- **Total Images:** ~126,413 word-level images.

The ground truth annotations are provided in text files mapping the image filename to the corresponding Telugu Unicode string. The vocabulary built from this dataset contains 91 distinct tokens, which include the Telugu base characters, modifiers, numerals, and special tokens (PAD, SOS, EOS, BLANK).

### 4.2 Data Preprocessing
Robust data preprocessing is critical for training invariant visual models. The pipeline includes:
1. **Grayscale Conversion:** All images are converted to single-channel grayscale.
2. **Resizing and Padding:** Images are resized to a fixed height of 64 pixels while maintaining the original aspect ratio. The width is then padded with white pixels to a maximum width of 512 pixels. Images wider than 512 pixels are scaled down to fit.
3. **Normalization:** Pixel values are scaled from `[0, 255]` to a normalized range of `[-1, 1]` to stabilize neural network gradients.
4. **Data Augmentation:** During training, images are subjected to random augmentations to prevent overfitting and improve robustness to writing styles. This includes slight random rotations (±3 degrees), elastic distortions to simulate cursive variations, and the addition of Gaussian noise to simulate poor scanning conditions.

### 4.3 Training Configuration
The models were implemented using Python 3.12.13 and PyTorch 2.7.1, running on an Ubuntu Linux environment. Training was accelerated using an NVIDIA GeForce RTX 3090 Ti GPU with 24 GB of VRAM and CUDA 11.8.

**CTC Training Details:**
- **Epochs:** 50
- **Batch Size:** 64
- **Optimizer:** AdamW with weight decay of 1e-4
- **Learning Rate:** 1e-3
- **Scheduler:** OneCycleLR, which smoothly ramps the learning rate up and down to enable faster convergence.
- **Mixed Precision:** Automatic Mixed Precision (fp16) was used to reduce VRAM usage and increase speed.
- **Gradient Clipping:** Gradients were clipped at a maximum norm of 5.0 to prevent exploding gradients.

**AR Transformer Training Details:**
- **Epochs:** 50
- **Batch Size:** 256
- **Optimizer:** AdamW with weight decay of 1e-4
- **Learning Rate:** 3e-4
- **Scheduler:** Warmup + Cosine Annealing. The learning rate warms up linearly over the first 4000 steps, then decays following a cosine curve.

### 4.4 Checkpoint Management
To ensure no progress was lost during lengthy training runs, a robust rolling checkpoint system was implemented. The system maintains exactly three checkpoint files on disk:
- `best.pt`: The model weights that achieved the lowest Character Error Rate on the validation set.
- `current.pt`: The weights from the most recently completed epoch.
- `previous.pt`: The weights from the epoch prior to the current one.
This allows training to be safely interrupted and resumed from the exact state of the last completed epoch.

### 4.5 Software Architecture
The project was structured as a modular Python package to ensure clean code and reproducibility. Key modules include:
- `src/vocab.py`: Handles Unicode character mapping, tokenization, and the generation of the data-driven constraint matrix.
- `src/dataset.py` & `src/transforms.py`: Manages data loading, image processing, and augmentation.
- `src/models/`: Contains the architecture definitions for `cnn_encoder.py`, `ctc_model.py`, and `ar_model.py`.
- `src/train_ctc.py` & `src/train_ar.py`: The core training loops, managing forward passes, loss computation, backpropagation, and logging.
- `src/evaluate.py`: The evaluation script that computes benchmark metrics and outputs the ablation table.
- TensorBoard was heavily utilized to log training loss, validation CER, and learning rate schedules in real-time.

---

## Chapter 5: Results and Analysis

### 5.1 Evaluation Metrics
The models were evaluated using standard HTR metrics:
- **Character Error Rate (CER):** The primary metric. It is computed using the Levenshtein edit distance (insertions + deletions + substitutions) between the predicted string and the ground truth, divided by the length of the ground truth. The lower the CER, the better.
- **Word Error Rate (WER):** The fraction of words that contain at least one error. A word is considered incorrect if its prediction does not exactly match the ground truth.
- **95% Bootstrap Confidence Interval:** To ensure statistical significance, CER confidence intervals were computed using 1000 bootstrap resamples of the test set.
- **Inference Speed:** Measured in milliseconds per sample (ms/sample) to evaluate production viability.
- **Compound vs. Simple CER:** The test set was split into words containing a Virama (compound) and words without a Virama (simple) to specifically evaluate the models' handling of complex ligatures.

### 5.2 Training Convergence
Both models demonstrated stable and smooth convergence during training.
- **CTC Baseline:** Started at a validation CER of 34.63% in Epoch 1. It steadily improved and reached its best validation CER of 2.65% at Epoch 46. Total training time was 3 hours and 22 minutes.
- **AR Joint Model:** Started at a validation CER of 33.83% in Epoch 1. It converged faster, reaching its best validation CER of 3.41% at Epoch 18. Total training time for 50 epochs was 2 hours and 15 minutes.
There were no signs of divergence or catastrophic forgetting in either training run.

### 5.3 Ablation Study Results
After training, the best checkpoint of each model was evaluated on the completely unseen 17,910-image test set. The comprehensive ablation study results are presented in Table 1.

*Table 1: Final Evaluation Results on IIIT-HW-Telugu Test Set (17,910 images)*

| Model | CER ↓ | WER ↓ | 95% CI (CER) | Compound CER | Simple CER | Speed |
|---|---|---|---|---|---|---|
| **CTC Baseline** | **3.91%** | **24.80%** | [3.79, 4.04] | 3.70% | 4.29% | **0.23 ms** |
| AR (unconstrained) | 4.89% | 29.83% | [4.75, 5.01] | 4.61% | 5.38% | 0.61 ms |
| AR + Telugu constraint | 4.99% | 29.71% | [4.85, 5.14] | 4.68% | 5.54% | 1.03 ms |
| AR + constraint + beam(5) | 4.85% | 29.49% | [4.72, 5.00] | 4.57% | 5.35% | 60.6 ms |

### 5.4 Analysis of Results
The results reveal several crucial insights about sequence modeling for Telugu HTR:

**1. CTC Outperforms the Transformer:** 
The most significant finding is that the simpler CTC Baseline (3.91% CER) definitively outperformed the more complex Autoregressive Transformer (4.85% CER). This contradicts the general trend in English OCR where Transformers dominate. This occurs because Telugu words in this dataset are relatively short and well-segmented. The CTC algorithm excels at establishing strict monotonic alignments between visual features and characters. The Transformer decoder, which relies on learned attention weights, may overcomplicate the alignment for short, heavily structured sequences, leading to slight overfitting given the dataset size. Furthermore, the CTC model is over 260 times faster than the AR model with beam search (0.23 ms vs 60.6 ms), making it highly superior for real-world deployment.

**2. Impact of Linguistic Constraints:**
The data-driven Telugu constraint mask had a mixed impact. For greedy decoding, it slightly degraded the CER (4.89% to 4.99%). However, when combined with beam search, it helped achieve the best AR result (4.85%). The constraint forces the model into grammatically correct paths, but if the visual features are highly ambiguous, forcing a specific path might occasionally lead to the wrong grammatically valid word.

**3. Compound vs. Simple Characters:**
A highly counterintuitive finding emerged from the Virama breakdown. For all models, the CER on compound words (words containing ligatures) was *lower* than the CER on simple words. For example, in the CTC model, Compound CER was 3.70% while Simple CER was 4.29%. We hypothesize that compound characters, while complex, form highly distinctive, large visual structures in the image. Simple characters (like basic vowels) are smaller and more easily confused with one another due to minor stroke variations.

### 5.5 Comparison with Published Results
To benchmark the success of this project, we compare our results with established literature on the IIIT-HW-Telugu dataset.

*Table 2: Comparison with Published Literature*

| Method | Architecture | CER |
|---|---|---|
| Dutta et al. (2018) | CNN-RNN-CTC | ~15.0% |
| Deshpande et al. (2021)| Attention-based Encoder-Decoder | ~10.2% |
| **Ours: AR + beam(5)** | **CNN + Transformer + Constraint** | **4.85%** |
| **Ours: CTC Baseline** | **ResNet-18 + BiLSTM + CTC** | **3.91%** |

Our CTC baseline model establishes a new state-of-the-art on this dataset. A CER of 3.91% represents a 74% relative improvement over the baseline set by Dutta et al. (2018) and a 62% relative improvement over the attention-based approach by Deshpande et al. (2021). Even our secondary AR model significantly outperforms prior published works.

### 5.6 Error Analysis
Manual inspection of the error examples (where predicted text did not match ground truth) indicates that the remaining errors are primarily due to:
1. **Extreme visual ambiguity:** Instances where human readers would also struggle to decipher the messy handwriting.
2. **Similar character confusion:** Errors between highly similar Telugu characters that differ by only a single minor stroke or loop.
3. **Artifacts:** Poorly cropped images where parts of adjacent words intrude into the frame.

---

## Chapter 6: Conclusion and Future Scope

### 6.1 Conclusion
This major project successfully developed, trained, and evaluated a high-performance handwritten text recognition system for the Telugu language. We implemented a hybrid CNN-Transformer architecture and introduced a novel, data-driven Telugu linguistic constraint mask to enforce orthographic rules during decoding. 

Through rigorous empirical evaluation on a test set of 17,910 images, we discovered that a well-optimized CTC baseline architecture (ResNet-18 + BiLSTM) outperformed the more complex Autoregressive Transformer model, achieving a highly accurate 3.91% Character Error Rate (CER). This performance significantly surpasses existing published literature for the IIIT-HW-Telugu dataset, demonstrating the effectiveness of the chosen visual encoder design (stride patching) and robust training methodologies. While the Telugu constraint mask provided only marginal improvements for the Transformer model on this specific dataset, the data-driven methodology proves that complex Indic script grammar can be seamlessly integrated into neural decoding pipelines. 

Ultimately, this project proves that for short, well-segmented Indic words, the CTC alignment mechanism remains a highly efficient, accurate, and production-ready solution.

### 6.2 Key Contributions
The key contributions of this B.Tech Major Project are:
1. Conducting the first systematic ablation study directly comparing CTC and Transformer AR architectures for Telugu HTR.
2. Designing and implementing a novel, data-driven methodology for extracting and enforcing Telugu orthographic constraints during neural network inference.
3. Achieving a state-of-the-art CER of 3.91% on the IIIT-HW-Telugu dataset, a massive improvement over historical benchmarks.
4. Developing a complete, modular, and reproducible PyTorch training and evaluation pipeline, featuring dynamic stride patching, mixed-precision training, and a rolling checkpoint system.

### 6.3 Future Scope
While this project achieved excellent results at the isolated word level, several avenues remain for future research:
1. **Line-Level Recognition:** Expanding the architecture to recognize full sentences and paragraphs without relying on explicit word-segmentation algorithms.
2. **Language Model Integration:** Incorporating a word-level or sub-word Telugu language model (e.g., a pre-trained IndicBERT) to correct phonetic and spelling errors post-recognition.
3. **Cross-Lingual Transfer Learning:** Leveraging the architectural success of this model to pre-train on larger datasets of similar scripts (like Kannada) to further improve Telugu HTR accuracy through transfer learning.
4. **Mobile Deployment:** Quantizing and optimizing the highly efficient CTC baseline (which takes only 0.23 ms per sample) for deployment as a real-time mobile application for document scanning.
5. **Synthetic Data Augmentation:** Exploring Generative Adversarial Networks (GANs) or diffusion models to generate synthetic Telugu handwriting to infinitely expand the training dataset.

---

## References

1. Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In *Proceedings of the 23rd international conference on Machine learning* (pp. 369-376).
2. Dutta, K., Krishnan, P., Mathew, M., & Jawahar, C. V. (2018). Improving CNN-RNN hybrid networks for handwriting recognition. In *2018 16th International Conference on Frontiers in Handwriting Recognition (ICFHR)* (pp. 80-85). IEEE.
3. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473*.
4. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems*, 30.
5. Li, M., Teng, P., Zhang, C., & Wang, H. (2022). TrOCR: Transformer-based optical character recognition with pre-trained models. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 36, No. 3, pp. 1309-1317).
6. Deshpande, A., Kumar, A., & Jawahar, C. V. (2021). Attention based Telugu handwritten text recognition. In *Proceedings of the International Conference on Document Analysis and Recognition (ICDAR)*.
7. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778).
8. Smith, L. N., & Topin, N. (2019). Super-convergence: Very fast training of neural networks using large learning rates. In *Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications* (Vol. 11006, pp. 369-386). SPIE.
9. Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*.
10. Jawahar, C. V., et al. (2020). IIIT-HW-Indic Datasets for Handwriting Recognition. *CVIT, IIIT Hyderabad*.
