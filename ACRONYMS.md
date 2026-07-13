# Glossary of Acronyms and Technical Terms

This document provides a comprehensive list of all acronyms and abbreviations used throughout this project's code, documentation, and methodology.

## Deep Learning & Architecture
* **AI** — Artificial Intelligence.
* **CNN** — Convolutional Neural Network (Used for extracting visual features from images, e.g., ResNet).
* **RNN** — Recurrent Neural Network (A type of network for sequential data).
* **BiLSTM** — Bidirectional Long Short-Term Memory (A type of RNN that processes data in both forward and backward directions; used in the baseline model).
* **CTC** — Connectionist Temporal Classification (A loss function used to train models like BiLSTM when the alignment between the input image and output text is unknown).
* **AR** — Autoregressive (A model that predicts the next character in a sequence based on the characters it has previously predicted; used in the Transformer Decoder).
* **ResNet** — Residual Network (A specific type of CNN with "skip connections" that make it easier to train deep networks. We use ResNet-18).
* **TrOCR** — Transformer-based Optical Character Recognition (A modern AI architecture that uses Transformers for both looking at the image and generating the text).
* **GELU** — Gaussian Error Linear Unit (An activation function used inside the Transformer models to help them learn complex patterns).
* **SOTA** — State of the Art (The highest level of general development or accuracy achieved at the present time).

## Metrics & Evaluation
* **CER** — Character Error Rate (The percentage of individual characters the model guessed wrong. Lower is better).
* **WER** — Word Error Rate (The percentage of entire words the model guessed wrong. Lower is better).
* **NFC** — Normalization Form C (A Unicode standard that combines characters and modifiers into a single pre-composed character when possible. Used to ensure fair CER scoring for Telugu).

## Vocabulary & Data Processing
* **HTR** — Handwritten Text Recognition (The specific task of reading handwritten text).
* **OCR** — Optical Character Recognition (The general task of reading text from images, usually printed text).
* **SOS / BOS** — Start Of Sequence / Beginning Of Sequence (A special token `<SOS>` used to tell the model to start predicting text).
* **EOS** — End Of Sequence (A special token `<EOS>` used by the model to indicate it has finished predicting the word).
* **PAD** — Padding (A special token `<PAD>` used to fill empty space so that all words in a batch are the exact same length).
* **UNK** — Unknown (A special token `<UNK>` used if the model encounters a character it has never seen before).
* **GT** — Ground Truth (The actual correct label or text for an image).

## Hardware & Engineering
* **GPU** — Graphics Processing Unit (The hardware used to train the AI models much faster than a standard processor).
* **VRAM** — Video Random Access Memory (The memory on the GPU. Models and batches must fit within this limit).
* **OOM** — Out Of Memory (An error that occurs when the batch size or model is too large to fit in the GPU's VRAM or the system's regular RAM).
* **IIIT** — International Institute of Information Technology (The institute in Hyderabad that created the `IIIT-HW-Telugu` dataset).
