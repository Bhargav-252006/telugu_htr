# Telugu Handwriting Reader (HTR)

Welcome! This project is an Artificial Intelligence (AI) system designed to do one very difficult thing: **Read handwritten Telugu words from images and turn them into typed text.**

Telugu is a beautiful language, but it is notoriously difficult for computers to read because the letters often merge together into complex shapes (ligatures) and have marks added to the top, bottom, and sides. This project uses modern AI to solve that problem.

---

## 🧠 How Does It Work? (Explained Simply)

Instead of trying to teach the computer exactly what a "K" or an "M" looks like, we show it thousands of examples and let it learn on its own. Our AI has four main parts:

1. **The Eye (ResNet-18)**: First, a standard image scanner (a Convolutional Neural Network) looks at the picture of the word and breaks it down into visual patterns.
2. **The Big Picture (Transformer Encoder)**: Next, a powerful AI mechanism looks at all those patterns *at the same time*. It connects the dots—like realizing that a small loop on the far left belongs to a letter on the far right.
3. **The Speaker (Transformer Decoder)**: Then, a text-generator starts guessing the word, one character at a time, by paying attention to the visual patterns.
4. **The Grammar Coach (Soft Penalty Constraint)**: Finally, as the Speaker guesses letters, the Coach checks a dictionary of "allowed Telugu letter combinations" that we built from the training data. If the Speaker tries to guess a combination of letters that doesn't exist in Telugu, the Coach gently penalizes that guess and forces it to try a real Telugu combination.

---

## 🛠️ How to Set Up the Project

If you are new to programming, don't worry! Just follow these steps in your terminal (command prompt).

### 1. Install the requirements
First, make sure you have Python installed. Then, install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Download the Data
You need the **IIIT-HW-Telugu** dataset, which contains thousands of pictures of handwritten Telugu words. 
You can download it from the [CVIT IIIT Hyderabad website](http://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data).

Once downloaded, organize the folders exactly like this:
```text
major/
 └── data/
      └── raw/
           ├── train/       <-- Put training images and labels.txt here
           ├── val/         <-- Put validation images and labels.txt here
           └── test/        <-- Put test images and labels.txt here
```

---

## 🚀 How to Run the AI

Once your data is in place, you can train the AI by running these commands one by one.

### Step 1: Teach the Grammar Coach
First, we need to tell the AI what valid Telugu letter combinations look like by scanning the training data.
```bash
python -m src.vocab data/raw/train/labels.txt checkpoints/vocab.pkl data/raw/val/labels.txt
```

### Step 2: Train the "Basic" AI (CTC Baseline)
We first train a simple, standard AI so we have a baseline to compare against. This takes about 1 hour if you have a good GPU (like an RTX 3090).
```bash
python -m src.train_ctc --config configs/ctc_config.yaml
```

### Step 3: Train the "Smart" AI (Autoregressive Model)
Now, we train our advanced, highly-intelligent model (The Eye + Big Picture + Speaker). This takes about 3 hours on a good GPU.
```bash
python -m src.train_ar --config configs/ar_config.yaml
```
*(Note: If your computer crashes or turns off, you can just run this exact same command again. The AI will automatically resume from where it left off!)*

### Step 4: Test How Smart It Is
Finally, we want to test the AI on new images it has never seen before to see how many words it gets right.

Run the basic AI test:
```bash
python -m src.evaluate --model_type ctc --checkpoint checkpoints/ctc/best.pt --config configs/ctc_config.yaml --split test
```

Run the smart AI test (without the Grammar Coach):
```bash
python -m src.evaluate --model_type ar --checkpoint checkpoints/ar/best.pt --config configs/ar_config.yaml --split test --no_constrain
```

Run the smartest AI test (WITH the Grammar Coach):
```bash
python -m src.evaluate --model_type ar --checkpoint checkpoints/ar/best.pt --config configs/ar_config.yaml --split test
```

---

## 📊 What Do The Results Mean?
When you run Step 4, you will see a score called **CER** (Character Error Rate). 
* **0.00** means the AI made zero mistakes (Perfect!).
* **0.50** means the AI got 50% of the characters wrong. 

The goal of this project is to show that our "Smartest AI" (with the Grammar Coach) gets a much lower (better) CER than the Basic AI!
