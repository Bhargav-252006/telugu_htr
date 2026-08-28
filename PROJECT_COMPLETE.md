# Telugu Handwritten Word Recognition — Complete Project Reference

## 1. Project Overview
- Brief description of what this project does
- Architecture: CNN (ResNet-18) + CTC baseline vs Autoregressive Transformer with joint CTC-Attention
- Dataset: IIIT-HW-Telugu (80,693 train / 20,048 val / 17,910 test)
- Results: AR Transformer 3.66% CER vs CTC 3.91% CER (best lexicon-free result)

## 2. Directory Structure
```text
Folder PATH listing for volume Windows-SSD
Volume serial number is 00000240 8296:E6AA
C:\USERS\BHARGAV\DESKTOP\MAJOR
¦   .gitignore
¦   BUGS_FOUND.txt
¦   Dataset.gz
¦   generator.py
¦   lexicon.txt
¦   Major(b4).pptx
¦   README.md
¦   Readme.txt
¦   requirements.txt
¦   sanity_output.txt
¦   setup_server.sh
¦   technical_deep_dive.md
¦   
+---checkpoints
¦   ¦   vocab.pkl
¦   ¦   
¦   +---ar
¦   ¦       best.pt
¦   ¦       current.pt
¦   ¦       previous.pt
¦   ¦       
¦   +---ar_no_ctc
¦   ¦       best.pt
¦   ¦       current.pt
¦   ¦       previous.pt
¦   ¦       
¦   +---ctc
¦           best.pt
¦           current.pt
¦           previous.pt
¦           
+---configs
¦       ar_config.yaml
¦       ar_no_ctc_config.yaml
¦       ar_v2_config.yaml
¦       ctc_config.yaml
¦       
+---logs
¦   ¦   ar_v2_training.log
¦   ¦   full_evaluation.log
¦   ¦   
¦   +---ar
¦   ¦       events.out.tfevents.1783769690.bvrit-csm-gpu.1083842.0
¦   ¦       events.out.tfevents.1783774084.bvrit-csm-gpu.1226518.0
¦   ¦       training_20260711_113450.json
¦   ¦       training_20260711_113450.log
¦   ¦       training_20260711_124804.json
¦   ¦       training_20260711_124804.log
¦   ¦       
¦   +---ar_no_ctc
¦   ¦       events.out.tfevents.1783789050.bvrit-csm-gpu.1773156.0
¦   ¦       events.out.tfevents.1783789728.bvrit-csm-gpu.1798166.0
¦   ¦       training_20260711_165729.json
¦   ¦       training_20260711_165729.log
¦   ¦       training_20260711_170848.json
¦   ¦       training_20260711_170848.log
¦   ¦       
¦   +---ar_v2
¦   ¦       events.out.tfevents.1785504307.bvrit-csm-gpu.3426951.0
¦   ¦       events.out.tfevents.1785504811.bvrit-csm-gpu.3444324.0
¦   ¦       events.out.tfevents.1785504932.bvrit-csm-gpu.3448680.0
¦   ¦       events.out.tfevents.1785505486.bvrit-csm-gpu.3466878.0
¦   ¦       training_20260731_132503.json
¦   ¦       training_20260731_132503.log
¦   ¦       training_20260731_132702.log
¦   ¦       training_20260731_133330.json
¦   ¦       training_20260731_133330.log
¦   ¦       training_20260731_133532.json
¦   ¦       training_20260731_133532.log
¦   ¦       training_20260731_134445.json
¦   ¦       training_20260731_134445.log
¦   ¦       
¦   +---ctc
¦           events.out.tfevents.1783757530.bvrit-csm-gpu.671646.0
¦           training_20260711_075914.log
¦           training_20260711_080951.json
¦           training_20260711_080951.log
¦           training_20260711_081210.json
¦           training_20260711_081210.log
¦           
+---notebooks
¦       01_data_exploration.ipynb
¦       02_baseline_results.ipynb
¦       03_ar_results_comparison.ipynb
¦       
+---paper
¦   ¦   main.tex
¦   ¦   
¦   +---figures
¦           ablation_comparison.png
¦           confusion_matrix_ar_v2.png
¦           data_pipeline.jpg
¦           prediction_examples.jpg
¦           system_architecture.jpg
¦           training_curves.png
¦           
+---results
¦   ¦   ablation_results.json
¦   ¦   error_examples_ctc.json
¦   ¦   final_results_analysis_v1_STALE.md
¦   ¦   
¦   +---paper_figures
¦           ablation_comparison.pdf
¦           ablation_comparison.png
¦           all_results.json
¦           confusion_matrix_ar_v2.png
¦           
+---scripts
¦   ¦   evaluate_benchmark.py
¦   ¦   generate_architecture_diagram.py
¦   ¦   generate_paper_figures.py
¦   ¦   generate_synthetic_dataset.py
¦   ¦   prepare_iiit_benchmark.py
¦   ¦   run_full_evaluation.py
¦   ¦   sanity_check.py
¦   ¦   
¦   +---__pycache__
¦           generate_synthetic_dataset.cpython-313.pyc
¦           sanity_check.cpython-313.pyc
¦           
+---src
¦   ¦   checkpoint_manager.py
¦   ¦   dataset.py
¦   ¦   evaluate.py
¦   ¦   training_logger.py
¦   ¦   train_ar.py
¦   ¦   train_ctc.py
¦   ¦   transforms.py
¦   ¦   vocab.py
¦   ¦   __init__.py
¦   ¦   
¦   +---decoding
¦   ¦   ¦   telugu_mask.py
¦   ¦   ¦   __init__.py
¦   ¦   ¦   
¦   ¦   +---__pycache__
¦   ¦           telugu_mask.cpython-313.pyc
¦   ¦           __init__.cpython-313.pyc
¦   ¦           
¦   +---models
¦   ¦   ¦   ar_model.py
¦   ¦   ¦   cnn_encoder.py
¦   ¦   ¦   ctc_model.py
¦   ¦   ¦   __init__.py
¦   ¦   ¦   
¦   ¦   +---__pycache__
¦   ¦           ar_model.cpython-313.pyc
¦   ¦           cnn_encoder.cpython-313.pyc
¦   ¦           ctc_model.cpython-313.pyc
¦   ¦           __init__.cpython-313.pyc
¦   ¦           
¦   +---results
¦   ¦       ablation_results.json
¦   ¦       
¦   +---__pycache__
¦           checkpoint_manager.cpython-313.pyc
¦           dataset.cpython-313.pyc
¦           evaluate.cpython-313.pyc
¦           train_ar.cpython-313.pyc
¦           train_ctc.cpython-313.pyc
¦           transforms.cpython-313.pyc
¦           vocab.cpython-313.pyc
¦           __init__.cpython-313.pyc
¦           
+---validating
    ¦   lexicon.txt
    ¦   Readme.txt
    ¦   requirements.txt
    ¦   sanity_output.txt
    ¦   setup_server.sh
    ¦   telugu_vocab.txt
    ¦   test.txt
    ¦   train.txt
    ¦   val.txt
    ¦   
    +---checkpoints
    ¦       vocab.pkl
    ¦       
    +---configs
    ¦       ar_config.yaml
    ¦       ar_no_ctc_config.yaml
    ¦       ctc_config.yaml
    ¦       
    +---logs
    +---notebooks
    ¦       01_data_exploration.ipynb
    ¦       02_baseline_results.ipynb
    ¦       03_ar_results_comparison.ipynb
    ¦       
    +---scripts
    ¦   ¦   evaluate_benchmark.py
    ¦   ¦   generate_synthetic_dataset.py
    ¦   ¦   prepare_iiit_benchmark.py
    ¦   ¦   sanity_check.py
    ¦   ¦   
    ¦   +---__pycache__
    ¦           generate_synthetic_dataset.cpython-313.pyc
    ¦           sanity_check.cpython-313.pyc
    ¦           
    +---src
        ¦   checkpoint_manager.py
        ¦   dataset.py
        ¦   evaluate.py
        ¦   training_logger.py
        ¦   train_ar.py
        ¦   train_ctc.py
        ¦   transforms.py
        ¦   vocab.py
        ¦   __init__.py
        ¦   
        +---decoding
        ¦   ¦   telugu_mask.py
        ¦   ¦   __init__.py
        ¦   ¦   
        ¦   +---__pycache__
        ¦           telugu_mask.cpython-313.pyc
        ¦           __init__.cpython-313.pyc
        ¦           
        +---models
        ¦   ¦   ar_model.py
        ¦   ¦   cnn_encoder.py
        ¦   ¦   ctc_model.py
        ¦   ¦   __init__.py
        ¦   ¦   
        ¦   +---__pycache__
        ¦           ar_model.cpython-312.pyc
        ¦           ar_model.cpython-313.pyc
        ¦           cnn_encoder.cpython-312.pyc
        ¦           cnn_encoder.cpython-313.pyc
        ¦           ctc_model.cpython-312.pyc
        ¦           ctc_model.cpython-313.pyc
        ¦           __init__.cpython-312.pyc
        ¦           __init__.cpython-313.pyc
        ¦           
        +---__pycache__
                checkpoint_manager.cpython-313.pyc
                dataset.cpython-312.pyc
                dataset.cpython-313.pyc
                evaluate.cpython-313.pyc
                train_ar.cpython-313.pyc
                train_ctc.cpython-313.pyc
                transforms.cpython-312.pyc
                transforms.cpython-313.pyc
                vocab.cpython-312.pyc
                vocab.cpython-313.pyc
                __init__.cpython-312.pyc
                __init__.cpython-313.pyc
                
```

## 3. Results Summary

### results/paper_figures/all_results.json
```json
{
  "AR v2 (greedy, unconstrained)": {
    "cer": 0.03667368933344433,
    "wer": 0.23333333333333334,
    "cer_ci": [
      0.03549118880448668,
      0.0379339030451939
    ],
    "speed_ms": 1.3923258038621453,
    "avg_pred_len": 8.729089893914015,
    "virama_breakdown": {
      "compound": {
        "count": 10129,
        "cer": 0.03316089136155778,
        "wer": 0.23467272188764932
      },
      "simple": {
        "count": 7781,
        "cer": 0.042917615253272624,
        "wer": 0.23158976995244826
      }
    }
  },
  "AR v2 (greedy, constrained)": {
    "cer": 0.037378090560383194,
    "wer": 0.2332216638749302,
    "cer_ci": [
      0.03603692617855835,
      0.038769425667313305
    ],
    "speed_ms": 1.767636303925634,
    "avg_pred_len": 8.73249581239531,
    "virama_breakdown": {
      "compound": {
        "count": 10129,
        "cer": 0.03332099222510181,
        "wer": 0.2326981933063481
      },
      "simple": {
        "count": 7781,
        "cer": 0.044589499146272055,
        "wer": 0.23390309728826628
      }
    }
  },
  "AR v2 (beam=5, unconstrained)": {
    "cer": 0.03660324921075044,
    "wer": 0.23344500279173647,
    "cer_ci": [
      0.03536166940071429,
      0.03786695112995781
    ],
    "speed_ms": 77.0026061602773,
    "avg_pred_len": 8.729480737018426,
    "virama_breakdown": {
      "compound": {
        "count": 10129,
        "cer": 0.03310085353772877,
        "wer": 0.23467272188764932
      },
      "simple": {
        "count": 7781,
        "cer": 0.04282868525896414,
        "wer": 0.2318468063230947
      }
    }
  },
  "AR v2 (beam=5, constrained)": {
    "cer": 0.036667285685926705,
    "wer": 0.2327749860413177,
    "cer_ci": [
      0.03550183531809931,
      0.037880106215369025
    ],
    "speed_ms": 82.54657121825525,
    "avg_pred_len": 8.731881630374092,
    "virama_breakdown": {
      "compound": {
        "count": 10129,
        "cer": 0.03296076528212774,
        "wer": 0.2322045611610228
      },
      "simple": {
        "count": 7781,
        "cer": 0.043255549231644846,
        "wer": 0.2335175427322966
      }
    }
  },
  "CTC Baseline": {
    "cer": 0.039113479037659854,
    "wer": 0.24796203238414294,
    "cer_ci": [
      0.037866528335890104,
      0.040428171187581546
    ],
    "speed_ms": 0.2817223685736898,
    "avg_pred_len": 8.720938023450586,
    "virama_breakdown": {
      "compound": {
        "count": 10129,
        "cer": 0.036983299478671565,
        "wer": 0.2571823477144832
      },
      "simple": {
        "count": 7781,
        "cer": 0.04289982925441093,
        "wer": 0.23595938825343787
      }
    }
  }
}
```

### results/ablation_results.json
```json
{
  "CTC Baseline": {
    "cer": 0.039113479037659854,
    "wer": 0.24796203238414294,
    "cer_ci": [
      0.03785770154300436,
      0.04033726820330558
    ],
    "inference_time_ms_per_sample": 2.6851100671364567,
    "virama_breakdown": {
      "compound": {
        "count": 10129,
        "cer": 0.036983299478671565,
        "wer": 0.2571823477144832
      },
      "simple": {
        "count": 7781,
        "cer": 0.04289982925441093,
        "wer": 0.23595938825343787
      }
    },
    "avg_pred_len": 8.720938023450586
  },
  "AR (no CTC aux)": {
    "cer": 0.05428372000691594,
    "wer": 0.3236739251814629,
    "cer_ci": [
      0.05284870292849873,
      0.05584563830097187
    ],
    "inference_time_ms_per_sample": 1.7023902043225978,
    "virama_breakdown": {
      "compound": {
        "count": 10129,
        "cer": 0.0498714189939662,
        "wer": 0.3329055188073847
      },
      "simple": {
        "count": 7781,
        "cer": 0.06212649402390438,
        "wer": 0.31165659940881635
      }
    },
    "avg_pred_len": 8.715801228364043
  },
  "AR (unconstrained)": {
    "cer": 0.048853426911969054,
    "wer": 0.29826912339475153,
    "cer_ci": [
      0.047475316798779955,
      0.050162922950764924
    ],
    "inference_time_ms_per_sample": 1.7083059802925966,
    "virama_breakdown": {
      "compound": {
        "count": 10129,
        "cer": 0.04605901718082392,
        "wer": 0.3146411294303485
      },
      "simple": {
        "count": 7781,
        "cer": 0.05382043255549231,
        "wer": 0.27695668937154605
      }
    },
    "avg_pred_len": 8.729201563372417
  },
  "AR + Telugu constraint": {
    "cer": 0.04991643239989498,
    "wer": 0.29709659408151873,
    "cer_ci": [
      0.048512885552275904,
      0.05128995237328073
    ],
    "inference_time_ms_per_sample": 40.24139122880417,
    "virama_breakdown": {
      "compound": {
        "count": 10129,
        "cer": 0.04683950889060108,
        "wer": 0.3118767894165268
      },
      "simple": {
        "count": 7781,
        "cer": 0.05538560045532157,
        "wer": 0.2778563166688086
      }
    },
    "avg_pred_len": 8.733109994416527
  },
  "AR + constraint + beam(5)": {
    "cer": 0.048520437241052504,
    "wer": 0.29491903964265775,
    "cer_ci": [
      0.04719833645765049,
      0.049988130564684145
    ],
    "inference_time_ms_per_sample": 327.4522206425201,
    "virama_breakdown": {
      "compound": {
        "count": 10129,
        "cer": 0.04571880284579285,
        "wer": 0.3097048079770955
      },
      "simple": {
        "count": 7781,
        "cer": 0.053500284575981785,
        "wer": 0.2756715075183138
      }
    },
    "avg_pred_len": 8.73283082077052
  },
  "_metadata": {
    "timestamp": "2026-07-13T08:29:14.464577",
    "device": "cuda"
  }
}
```

## 4. Configuration Files

### configs/ctc_config.yaml
```yaml
model:
  vocab_size: null
  d_model: 256
  lstm_hidden: 256
  lstm_layers: 2
  dropout: 0.1
  pretrained: true
data:
  train_annotation: data/raw/train/labels.txt
  val_annotation: data/raw/val/labels.txt
  train_image_root: .
  val_image_root: .
  image_height: 64
  max_image_width: 512
  max_label_len: 32
  use_elastic: false
  num_workers: 2
training:
  batch_size: 64
  epochs: 50
  lr: 0.0003
  weight_decay: 0.0001
  grad_clip: 5.0
  mixed_precision: true
  log_interval: 50
  val_interval: 1
  save_dir: checkpoints/ctc
  log_dir: logs/ctc
  vocab_path: checkpoints/vocab.pkl
  warmup_steps: 4000

```

### configs/ar_config.yaml
```yaml
model:
  vocab_size: null
  d_model: 256
  num_encoder_layers: 2
  nhead: 4
  num_decoder_layers: 4
  dim_feedforward: 1024
  dropout: 0.2
  high_res_temporal: false
  max_label_len: 34
  label_smoothing: 0.1
  pretrained: true
  ctc_weight: 0.3
data:
  train_annotation: data/raw/train/labels.txt
  val_annotation: data/raw/val/labels.txt
  train_image_root: .
  val_image_root: .
  image_height: 64
  max_image_width: 512
  max_label_len: 32
  use_elastic: true
  num_workers: 8
training:
  batch_size: 256
  epochs: 50
  lr: 0.0003
  weight_decay: 0.0001
  warmup_steps: 4000
  grad_clip: 5.0
  mixed_precision: true
  log_interval: 50
  val_interval: 1
  greedy_val: true
  beam_size: 5
  constrain_decode: true
  constrain_penalty: 10.0
  save_dir: checkpoints/ar
  log_dir: logs/ar
  vocab_path: checkpoints/vocab.pkl

```

### configs/ar_v2_config.yaml
```yaml
# ═══════════════════════════════════════════════════════════════════
# AR Transformer v2 — Improved Configuration
# ═══════════════════════════════════════════════════════════════════
# Fixes applied:
#   1. d_model: 256 → 384 (more capacity)
#   2. nhead: 4 → 8
#   3. num_decoder_layers: 4 → 6
#   4. dim_feedforward: 1024 → 1536 (4×d_model)
#   5. num_encoder_layers: 2 → 3
#   6. batch_size: 256 → 64 (avoid sharp minima)
#   7. epochs: 50 → 80 (more training with smaller batch)
#   8. label_smoothing: 0.1 → 0.05
#   9. dropout: 0.2 → 0.15
# ═══════════════════════════════════════════════════════════════════

model:
  vocab_size: null
  d_model: 384
  num_encoder_layers: 3
  nhead: 8
  num_decoder_layers: 6
  dim_feedforward: 1536
  dropout: 0.15
  high_res_temporal: false
  max_label_len: 34
  label_smoothing: 0.05
  pretrained: true
  ctc_weight: 0.3

data:
  train_annotation: data/raw/train/labels.txt
  val_annotation: data/raw/val/labels.txt
  train_image_root: .
  val_image_root: .
  image_height: 64
  max_image_width: 512
  max_label_len: 32
  use_elastic: true
  num_workers: 8

training:
  batch_size: 64
  epochs: 80
  lr: 0.0003
  weight_decay: 0.0001
  warmup_steps: 4000
  grad_clip: 5.0
  mixed_precision: true
  log_interval: 50
  val_interval: 1
  greedy_val: true
  beam_size: 5
  constrain_decode: true
  constrain_penalty: 10.0
  save_dir: checkpoints/ar_v2
  log_dir: logs/ar_v2
  vocab_path: checkpoints/vocab.pkl

```

## 5. Source Code

### src/__init__.py
```python

```

### src/vocab.py
```python
"""
src/vocab.py

Telugu character vocabulary and DATA-DRIVEN transition validity matrix.

Design philosophy
─────────────────
Instead of hard-coding linguistic rules from intuition, we:
  1. Scan ALL training labels and extract every observed (prev, next) bigram.
  2. Build the validity matrix from those OBSERVED transitions only.
  3. Optionally layer on hard linguistic rules AFTER verifying they do not
     conflict with the observed data.
  4. Log a conflict report so you can audit every blocked transition.

This guarantees the constraint matrix never blocks a transition that
actually appears in the training set.

Usage
─────
  # First run (no data yet): use a permissive default vocab
  vocab = TeluguVocab()

  # After dataset is available: build from data
  vocab = TeluguVocab.from_annotation_files(["data/raw/train/labels.txt"])
  vocab.build_data_driven_matrix(["data/raw/train/labels.txt"])

  # Inspect what was learned
  vocab.print_transition_stats()

  # Save / load
  vocab.save("checkpoints/vocab.pkl")
  vocab = TeluguVocab.load("checkpoints/vocab.pkl")
"""

from __future__ import annotations

import os
import pickle
import json
import unicodedata
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════
# Telugu Unicode character sets
# ═══════════════════════════════════════════════════════════════════

# Independent vowels  U+0C05–U+0C14
TELUGU_VOWELS: Set[str] = set(
    "\u0C05\u0C06\u0C07\u0C08\u0C09\u0C0A\u0C0B\u0C0C"
    "\u0C0E\u0C0F\u0C10\u0C12\u0C13\u0C14"
)

# Consonants  U+0C15–U+0C39, plus extras
TELUGU_CONSONANTS: Set[str] = set(
    "\u0C15\u0C16\u0C17\u0C18\u0C19"
    "\u0C1A\u0C1B\u0C1C\u0C1D\u0C1E"
    "\u0C1F\u0C20\u0C21\u0C22\u0C23"
    "\u0C24\u0C25\u0C26\u0C27\u0C28"
    "\u0C2A\u0C2B\u0C2C\u0C2D\u0C2E"
    "\u0C2F\u0C30\u0C31\u0C32\u0C33"
    "\u0C35\u0C36\u0C37\u0C38\u0C39"
    "\u0C3D\u0C58\u0C59\u0C5A"
)

# Dependent vowel signs (matras)  U+0C3E–U+0C4C, U+0C55, U+0C56
TELUGU_VOWEL_SIGNS: Set[str] = set(
    "\u0C3E\u0C3F\u0C40"
    "\u0C41\u0C42\u0C43\u0C44"
    "\u0C46\u0C47\u0C48"
    "\u0C4A\u0C4B\u0C4C"
    "\u0C55\u0C56"
)

# Virama / halant — joins consonants into conjuncts
VIRAMA: str = "\u0C4D"

# Anusvara, Visarga, Nukta
TELUGU_ANUSVARA: str = "\u0C02"
TELUGU_VISARGA:  str = "\u0C03"
TELUGU_NUKTA:    str = "\u0C00"
TELUGU_MODIFIERS: Set[str] = {TELUGU_ANUSVARA, TELUGU_VISARGA, TELUGU_NUKTA}

# Telugu digits
TELUGU_DIGITS: Set[str] = set("\u0C66\u0C67\u0C68\u0C69\u0C6A\u0C6B\u0C6C\u0C6D\u0C6E\u0C6F")

# Full set
ALL_TELUGU_CHARS: Set[str] = (
    TELUGU_VOWELS | TELUGU_CONSONANTS | TELUGU_VOWEL_SIGNS
    | {VIRAMA} | TELUGU_MODIFIERS | TELUGU_DIGITS
)

# ═══════════════════════════════════════════════════════════════════
# Special tokens
# ═══════════════════════════════════════════════════════════════════

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# ═══════════════════════════════════════════════════════════════════
# Vocabulary
# ═══════════════════════════════════════════════════════════════════

class TeluguVocab:
    """
    Character-level vocabulary for Telugu HTR with a data-driven
    transition validity matrix.

    The validity matrix `valid_next[i][j]` = True means token j is
    allowed as the next prediction after token i during constrained
    decoding. It is built by observing actual label bigrams.
    """

    def __init__(self, chars: Optional[List[str]] = None):
        """
        Parameters
        ----------
        chars : list of Telugu character strings to include.
                If None, uses the full predefined Telugu character set.
        """
        if chars is None:
            chars = sorted(ALL_TELUGU_CHARS)

        self._idx2char: List[str] = SPECIAL_TOKENS + chars
        self._char2idx: Dict[str, int] = {c: i for i, c in enumerate(self._idx2char)}

        self.pad_id = self._char2idx[PAD_TOKEN]
        self.sos_id = self._char2idx[SOS_TOKEN]
        self.eos_id = self._char2idx[EOS_TOKEN]
        self.unk_id = self._char2idx[UNK_TOKEN]

        # Validity matrix: initialise as fully permissive (all True)
        # Will be overwritten by build_data_driven_matrix()
        V = len(self)
        self._valid_next: List[List[bool]] = [[True] * V for _ in range(V)]

        # Statistics recorded during matrix build
        self._observed_bigrams: Dict[Tuple[int, int], int] = {}   # (prev_id, next_id) → count
        self._blocked_bigrams:  List[Tuple[int, int]]       = []  # pairs blocked by rules
        self._matrix_source: str = "permissive_default"

    # ─────────────────────────────────────────────────────────────
    # Size / lookup
    # ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._idx2char)

    def char2idx(self, ch: str) -> int:
        return self._char2idx.get(ch, self.unk_id)

    def idx2char(self, idx: int) -> str:
        if 0 <= idx < len(self._idx2char):
            return self._idx2char[idx]
        return UNK_TOKEN

    def encode(self, text: str) -> List[int]:
        text = unicodedata.normalize("NFC", text)
        return [self.char2idx(ch) for ch in text]

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        chars = []
        for i in ids:
            if strip_special and i == self.eos_id:
                break
            if strip_special and i in (self.pad_id, self.sos_id):
                continue
            chars.append(self.idx2char(i))
        return "".join(chars)

    def category(self, idx: int) -> str:
        ch = self.idx2char(idx)
        if ch == PAD_TOKEN:              return "PAD"
        if ch == SOS_TOKEN:              return "SOS"
        if ch == EOS_TOKEN:              return "EOS"
        if ch == UNK_TOKEN:              return "UNK"
        if ch in TELUGU_VOWELS:          return "VOWEL"
        if ch in TELUGU_CONSONANTS:      return "CONSONANT"
        if ch in TELUGU_VOWEL_SIGNS:     return "VOWEL_SIGN"
        if ch == VIRAMA:                 return "VIRAMA"
        if ch in TELUGU_MODIFIERS:       return "MODIFIER"
        if ch in TELUGU_DIGITS:          return "DIGIT"
        return "OTHER"

    # ─────────────────────────────────────────────────────────────
    # Data-driven transition matrix
    # ─────────────────────────────────────────────────────────────

    def build_data_driven_matrix(
        self,
        annotation_files: List[str],
        soft_mode: bool = False,
        soft_penalty: float = -10.0,
        verbose: bool = True,
    ):
        """
        Build the transition validity matrix from OBSERVED label bigrams.

        Algorithm
        ─────────
        1. Parse every label in the annotation files.
        2. For each label, generate synthetic bigrams:
             SOS → first_char
             char[i] → char[i+1]   for all consecutive pairs
             last_char → EOS
        3. Record all observed (prev_id, next_id) pairs.
        4. Set valid_next[i][j] = True  iff  (i,j) was observed in data.
           Special cases:
             - PAD → nothing  (always blocked)
             - EOS → nothing  (always blocked)
             - UNK → everything  (always allowed, it's a fallback)
             - everything → UNK  (always allowed)
        5. Store the set of blocked pairs for audit.

        Parameters
        ----------
        annotation_files : list of annotation .txt paths (train only — DO NOT
                           include val/test to avoid data leakage).
        soft_mode        : if True, don't fully block — return a float penalty
                           tensor instead (useful if matrix is exported to GPU).
        verbose          : print audit report.
        """
        V = len(self)
        observed: Dict[Tuple[int, int], int] = defaultdict(int)

        total_labels = 0
        for path in annotation_files:
            if not os.path.exists(path):
                print(f"[Vocab] WARNING: annotation file not found: {path}")
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    label = unicodedata.normalize("NFC", parts[1].strip())
                    if not label:
                        continue

                    total_labels += 1
                    ids = self.encode(label)

                    # SOS → first token
                    if ids:
                        observed[(self.sos_id, ids[0])] += 1

                    # Consecutive pairs
                    for p, n in zip(ids, ids[1:]):
                        observed[(p, n)] += 1

                    # Last token → EOS
                    if ids:
                        observed[(ids[-1], self.eos_id)] += 1

        if verbose:
            print(f"\n[Vocab] Scanned {total_labels:,} labels from {len(annotation_files)} file(s).")
            print(f"[Vocab] Observed {len(observed):,} unique (prev, next) token bigrams.")

        # ── Build validity matrix ──────────────────────────────
        # Default: block everything, then allow what was observed
        valid = [[False] * V for _ in range(V)]

        for (prev_id, next_id), count in observed.items():
            if 0 <= prev_id < V and 0 <= next_id < V:
                valid[prev_id][next_id] = True

        # Always allow transitions to/from UNK (safety fallback)
        for i in range(V):
            valid[i][self.unk_id] = True
            valid[self.unk_id][i] = True

        # Always block PAD and EOS as previous tokens
        for j in range(V):
            valid[self.pad_id][j] = False
            valid[self.eos_id][j] = False

        # PAD as next is always blocked (PAD only pads sequences, never predicted)
        for i in range(V):
            valid[i][self.pad_id] = False

        self._valid_next       = valid
        self._observed_bigrams = dict(observed)
        self._matrix_source    = "data_driven"

        if verbose:
            self._audit_report(V)

    def _audit_report(self, V: int):
        """Print a human-readable summary of the transition matrix."""
        total_pairs   = V * V
        allowed_pairs = sum(self._valid_next[i][j] for i in range(V) for j in range(V))
        blocked_pairs = total_pairs - allowed_pairs
        coverage_pct  = 100.0 * allowed_pairs / total_pairs

        print(f"\n[Vocab] ── Transition Matrix Audit ─────────────────────────")
        print(f"  Vocabulary size   : {V}")
        print(f"  Total token pairs : {total_pairs:,}")
        print(f"  Allowed pairs     : {allowed_pairs:,}  ({coverage_pct:.1f}%)")
        print(f"  Blocked pairs     : {blocked_pairs:,}  ({100-coverage_pct:.1f}%)")
        print(f"  Matrix source     : {self._matrix_source}")

        # Per-category summary
        cats = set(self.category(i) for i in range(V))
        print(f"\n  Allowed transitions by (prev_category → next_category):")
        cat_list = ["SOS", "VOWEL", "CONSONANT", "VOWEL_SIGN", "VIRAMA",
                    "MODIFIER", "DIGIT", "EOS", "UNK"]
        header = f"  {'PREV':>12} |" + "".join(f"{c:>12}" for c in cat_list)
        print(header)
        print("  " + "-" * (13 + 12 * len(cat_list)))

        for prev_cat in cat_list:
            prev_ids = [i for i in range(V) if self.category(i) == prev_cat]
            row = f"  {prev_cat:>12} |"
            for nxt_cat in cat_list:
                nxt_ids = [j for j in range(V) if self.category(j) == nxt_cat]
                allowed = sum(
                    1 for p in prev_ids for n in nxt_ids if self._valid_next[p][n]
                )
                total = len(prev_ids) * len(nxt_ids)
                if total == 0:
                    row += f"{'—':>12}"
                else:
                    row += f"{allowed:>5}/{total:<6}"
            print(row)
        print()

    # ─────────────────────────────────────────────────────────────
    # Constrained decoding interface
    # ─────────────────────────────────────────────────────────────

    def get_valid_next_mask(self, prev_token_id: int) -> List[bool]:
        """
        Return a boolean list of length V.
        True  → token at that index is valid after prev_token_id.
        False → blocked.
        """
        if 0 <= prev_token_id < len(self._valid_next):
            return self._valid_next[prev_token_id]
        return [True] * len(self)   # fallback: allow everything

    def get_valid_next_tensor(
        self, prev_token_id: int, device: str = "cpu"
    ):
        """
        Return a boolean torch.Tensor of shape [V] on the given device.
        Useful for direct logit masking in the decoder.
        """
        import torch
        mask = self._valid_next[prev_token_id]
        return torch.tensor(mask, dtype=torch.bool, device=device)

    def is_valid_transition(self, prev_id: int, next_id: int) -> bool:
        return bool(self._valid_next[prev_id][next_id])

    # ─────────────────────────────────────────────────────────────
    # Validation helper — run BEFORE training to catch rule issues
    # ─────────────────────────────────────────────────────────────

    def validate_against_split(
        self, annotation_file: str, split_name: str = "val"
    ) -> Dict:
        """
        Run through a split (val or test) and count how many label
        transitions are blocked by the current matrix.

        A blocked val/test transition means the constraint will HURT
        decoding on that sample. This number should be 0 or very close to 0.

        Returns a dict with violation statistics.
        """
        total_transitions  = 0
        blocked_count      = 0
        blocked_examples   = []   # (label, prev_char, next_char)

        with open(annotation_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                label = unicodedata.normalize("NFC", parts[1].strip())
                ids   = self.encode(label)
                full  = [self.sos_id] + ids + [self.eos_id]

                for p, n in zip(full, full[1:]):
                    total_transitions += 1
                    if not self._valid_next[p][n]:
                        blocked_count += 1
                        if len(blocked_examples) < 20:
                            blocked_examples.append(
                                (label, self.idx2char(p), self.idx2char(n))
                            )

        violation_rate = blocked_count / max(total_transitions, 1)
        print(f"\n[Vocab] Validation against [{split_name}]:")
        print(f"  Total transitions   : {total_transitions:,}")
        print(f"  Blocked transitions : {blocked_count:,}  ({violation_rate*100:.3f}%)")

        if blocked_examples:
            print(f"  First blocked examples:")
            for label, prev_ch, next_ch in blocked_examples[:5]:
                print(f"    label='{label}'  prev='{prev_ch}'  next='{next_ch}'")

        if violation_rate > 0.001:
            print(f"  ⚠️  WARNING: >0.1% transitions blocked on {split_name}.")
            print(f"     Consider expanding training data coverage or relaxing rules.")
        else:
            print(f"  ✓  Constraint matrix is safe on {split_name}.")

        return {
            "split": split_name,
            "total_transitions": total_transitions,
            "blocked_count": blocked_count,
            "violation_rate": violation_rate,
            "blocked_examples": blocked_examples,
        }

    # ─────────────────────────────────────────────────────────────
    # Factory: build from annotation files
    # ─────────────────────────────────────────────────────────────

    @classmethod
    def from_annotation_files(
        cls, annotation_files: List[str], build_matrix: bool = True
    ) -> "TeluguVocab":
        """
        1. Scan labels to collect observed characters.
        2. Build vocab from those characters only.
        3. Optionally build data-driven transition matrix from train labels.

        Parameters
        ----------
        annotation_files : TRAIN split annotation files only.
        build_matrix     : if True, also call build_data_driven_matrix().
        """
        from collections import Counter
        char_counter: Counter = Counter()

        for path in annotation_files:
            if not os.path.exists(path):
                print(f"[Vocab] WARNING: {path} not found — skipping.")
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        char_counter.update(unicodedata.normalize("NFC", parts[1]))

        # Keep only chars that are actual Telugu (filter noise)
        chars = sorted(ch for ch in char_counter if ch in ALL_TELUGU_CHARS)
        # Include any extra unknown chars (handle dirty labels gracefully)
        extra = sorted(ch for ch in char_counter
                       if ch not in ALL_TELUGU_CHARS and ch not in SPECIAL_TOKENS)
        if extra:
            print(f"[Vocab] Found {len(extra)} non-Telugu chars in labels: {extra[:10]}")

        vocab = cls(chars + extra)
        print(f"[Vocab] Built vocabulary: {len(vocab)} tokens "
              f"({len(chars)} Telugu + {len(extra)} extra + {len(SPECIAL_TOKENS)} special)")

        if build_matrix:
            vocab.build_data_driven_matrix(annotation_files)

        return vocab

    # ─────────────────────────────────────────────────────────────
    # Serialisation
    # ─────────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Vocab] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TeluguVocab":
        with open(path, "rb") as f:
            vocab = pickle.load(f)
        print(f"[Vocab] Loaded from {path} (size={len(vocab)})")
        return vocab

    def __repr__(self) -> str:
        return (
            f"TeluguVocab(size={len(self)}, "
            f"matrix_source='{self._matrix_source}')"
        )


# ═══════════════════════════════════════════════════════════════════
# Module-level default vocab (permissive, no matrix built yet)
# ═══════════════════════════════════════════════════════════════════
DEFAULT_VOCAB = TeluguVocab()


# ═══════════════════════════════════════════════════════════════════
# CLI entry point for building & saving vocab from dataset
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.vocab <train_annotation.txt> <output_vocab.pkl> [val_annotation.txt]")
        sys.exit(1)

    train_ann  = sys.argv[1]
    output_pkl = sys.argv[2]
    val_ann    = sys.argv[3] if len(sys.argv) > 3 else None

    # Build from training data
    vocab = TeluguVocab.from_annotation_files([train_ann], build_matrix=True)

    # Validate against val split (should be 0 violations)
    if val_ann and os.path.exists(val_ann):
        vocab.validate_against_split(val_ann, split_name="val")

    # Save
    vocab.save(output_pkl)

    # Quick round-trip test
    word = next(
        (parts[1] for line in open(train_ann, encoding="utf-8")
         for parts in [line.strip().split(maxsplit=1)] if len(parts) == 2),
        "కాలం"
    )
    ids  = vocab.encode(word)
    back = vocab.decode(ids)
    print(f"\nRound-trip: '{word}' → {ids} → '{back}'")
    print(f"\nDone. Vocab saved to: {output_pkl}")

```

### src/transforms.py
```python
"""
src/transforms.py
Image transforms for Telugu HTR.

All images are:
  • Converted to grayscale
  • Resized to a fixed height (H=64), width kept proportional
  • Width-padded (white background) to max_width (512)
  • Normalised to [0, 1] then standardised with ImageNet-compatible stats

Augmentation (training only):
  • Small random rotation  (±5°)
  • Random perspective distortion
  • Random brightness / contrast jitter
  • Optional elastic distortion (via grid_distort)
"""

from __future__ import annotations
import random
import math
from typing import Tuple, Optional

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageFilter
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_HEIGHT   = 64          # fixed height for all images
MAX_WIDTH       = 512         # maximum width; wider images are clipped/padded
MEAN            = [0.5]       # grayscale single-channel
STD             = [0.5]       # maps [0,1] → [-1, 1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_keep_aspect(img: Image.Image, target_h: int, max_w: int) -> Image.Image:
    """Resize PIL image to target_h, keep aspect ratio, crop if wider than max_w."""
    w, h = img.size
    scale = target_h / h
    new_w = int(w * scale)
    resized = img.resize((new_w, target_h), getattr(Image, "Resampling", Image).BICUBIC)
    if new_w > max_w:
        return resized.crop((0, 0, max_w, target_h))
    return resized


def _pad_to_width(img: Image.Image, target_w: int, fill: int = 255) -> Image.Image:
    """Pad image on the right to target_w with fill colour (white by default)."""
    w, h = img.size
    if w >= target_w:
        return img.crop((0, 0, target_w, h))    # crop if accidentally wider
    padded = Image.new(img.mode, (target_w, h), fill)
    padded.paste(img, (0, 0))
    return padded


def _elastic_distort(img: Image.Image, alpha: float = 10.0, sigma: float = 3.0) -> Image.Image:
    """
    Lightweight elastic distortion using a random displacement grid.
    alpha  – magnitude of displacement (pixels)
    sigma  – smoothing radius
    """
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Random displacement fields
    dx = (np.random.rand(h, w) * 2 - 1) * alpha
    dy = (np.random.rand(h, w) * 2 - 1) * alpha

    # Smooth with a simple box filter
    from scipy.ndimage import gaussian_filter   # lazy import; only used if called
    dx = gaussian_filter(dx, sigma)
    dy = gaussian_filter(dy, sigma)

    # Build mapping
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)

    import cv2   # lazy import
    distorted = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR)
    return Image.fromarray(distorted.astype(np.uint8))


def _morphological_ops(img: Image.Image) -> Image.Image:
    """Randomly apply erosion or dilation to simulate varying pen thickness.
    Assuming dark text on light background:
    MinFilter (dilation of dark strokes) -> thicker pen
    MaxFilter (erosion of dark strokes) -> thinner pen
    """
    op = random.choice(["thicken", "thin", "none"])
    if op == "thicken":
        # MinFilter makes dark pixels expand (thickens strokes)
        return img.filter(ImageFilter.MinFilter(3))
    elif op == "thin":
        # MaxFilter makes light pixels expand (thins strokes)
        return img.filter(ImageFilter.MaxFilter(3))
    return img


# ---------------------------------------------------------------------------
# Transform classes
# ---------------------------------------------------------------------------

class TrainTransform:
    """
    Augmentation + normalisation pipeline for training images.

    Returns a tuple of (float tensor [1, H, W] in [-1, 1], scaled_width).
    """

    def __init__(
        self,
        target_h: int  = TARGET_HEIGHT,
        max_w: int     = MAX_WIDTH,
        rotation: float = 5.0,      # ± degrees
        use_elastic: bool = False,  # requires scipy + cv2; off by default
    ):
        self.target_h    = target_h
        self.max_w       = max_w
        self.rotation    = rotation
        self.use_elastic = use_elastic

        self.jitter = T.ColorJitter(brightness=0.3, contrast=0.3)
        self.normalize = T.Normalize(mean=MEAN, std=STD)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # 1. Grayscale
        img = ImageOps.grayscale(img)

        # 2. Random elastic distort (optional)
        if self.use_elastic and random.random() < 0.3:
            try:
                img = _elastic_distort(img)
            except ImportError:
                pass  # scipy / cv2 not installed — skip silently

        # 2b. Random stroke thickness variation
        if random.random() < 0.3:
            img = _morphological_ops(img)

        # 3. Random rotation
        if self.rotation > 0 and random.random() < 0.5:
            angle = random.uniform(-self.rotation, self.rotation)
            img = TF.rotate(img, angle, fill=255)

        # 4. Random perspective
        if random.random() < 0.3:
            w, h = img.size
            startpoints = [
                [0,     0    ],
                [w - 1, 0    ],
                [w - 1, h - 1],
                [0,     h - 1],
            ]
            endpoints = self._rand_perspective_pts(img)
            img = TF.perspective(
                img,
                startpoints=startpoints,
                endpoints=endpoints,
                fill=255,
            )

        # 5. Resize + pad
        img = _resize_keep_aspect(img, self.target_h, self.max_w)
        scaled_w = img.size[0]  # width before padding
        img = _pad_to_width(img, self.max_w, fill=255)

        # 6. Brightness / contrast jitter (applied on PIL then converted)
        img = img.convert("RGB")   # ColorJitter needs 3-ch
        img = self.jitter(img)
        img = ImageOps.grayscale(img)

        # 7. To tensor [1, H, W], then normalize
        tensor = TF.to_tensor(img)          # [1, H, W] in [0, 1]
        tensor = self.normalize(tensor)     # → [-1, 1]
        return tensor, scaled_w

    @staticmethod
    def _rand_perspective_pts(img: Image.Image, jitter: float = 0.05):
        """Return four jittered corners for use as perspective endpoints."""
        w, h = img.size
        d = int(min(w, h) * jitter)

        def rnd(lo, hi):
            return random.randint(lo, hi) if lo < hi else lo

        return [
            [rnd(0, d),       rnd(0, d)],
            [rnd(w-d-1, w-1), rnd(0, d)],
            [rnd(w-d-1, w-1), rnd(h-d-1, h-1)],
            [rnd(0, d),       rnd(h-d-1, h-1)],
        ]


class ValTransform:
    """
    Deterministic resize + pad + normalise for validation / test / inference.

    Returns a tuple of (float tensor [1, H, W] in [-1, 1], scaled_width).
    """

    def __init__(self, target_h: int = TARGET_HEIGHT, max_w: int = MAX_WIDTH):
        self.target_h  = target_h
        self.max_w     = max_w
        self.normalize = T.Normalize(mean=MEAN, std=STD)

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, int]:
        img = ImageOps.grayscale(img)
        img = _resize_keep_aspect(img, self.target_h, self.max_w)
        scaled_w = img.size[0]  # width before padding
        img = _pad_to_width(img, self.max_w, fill=255)
        tensor = TF.to_tensor(img)       # [1, H, W] in [0, 1]
        tensor = self.normalize(tensor)  # → [-1, 1]
        return tensor, scaled_w


# ---------------------------------------------------------------------------
# Collate helper (used in DataLoader)
# ---------------------------------------------------------------------------

def collate_fn_pad(batch):
    """
    Pads a batch of (image_tensor, original_width, label_ids, label_len) tuples.
    Images are already fixed-width from the transform, so only labels need padding.
    Returns:
        images   : [B, 1, H, W]
        labels   : [B, T_max]  (0-padded)
        lengths  : [B]         (original label lengths, LongTensor)
        widths   : [B]         (scaled image widths before padding, LongTensor)
    """
    images, widths, labels, lengths = zip(*batch)
    images  = torch.stack(images, dim=0)             # [B, 1, H, W]
    max_len = max(len(l) for l in labels)
    padded  = torch.zeros(len(labels), max_len, dtype=torch.long)
    for i, lab in enumerate(labels):
        t = torch.tensor(lab, dtype=torch.long)
        padded[i, :len(lab)] = t
    lengths = torch.tensor(lengths, dtype=torch.long)
    widths  = torch.tensor(widths, dtype=torch.long)
    return images, padded, lengths, widths

```

### src/dataset.py
```python
"""
src/dataset.py
PyTorch Dataset for IIIT-HW-Telugu.

Annotation file format (one per split):
    <image_filename>  <ground_truth_label>

e.g.
    train/word_00001.png  కాలం
    train/word_00002.png  పూజ

Images live at:  <image_root>/<image_filename>
"""

from __future__ import annotations
import os
import unicodedata
from typing import List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.vocab import TeluguVocab, DEFAULT_VOCAB
from src.transforms import TrainTransform, ValTransform, collate_fn_pad


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TeluguHTRDataset(Dataset):
    """
    Word-level handwritten Telugu dataset.

    Parameters
    ----------
    annotation_file : str
        Path to the split's annotation .txt file.
    image_root : str
        Root directory that contains the images (images paths in the
        annotation file are relative to this root).
    vocab : TeluguVocab
        Character vocabulary.
    transform : callable
        Image transform (TrainTransform or ValTransform).
    max_label_len : int
        Labels longer than this are skipped during loading.
    add_sos_eos : bool
        If True, prepend SOS and append EOS to every label sequence.
        Required for autoregressive training; not needed for CTC.
    """

    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        vocab: TeluguVocab,
        transform: Callable,
        max_label_len: int = 32,
        add_sos_eos: bool = False,
    ):
        self.image_root    = image_root
        self.vocab         = vocab
        self.transform     = transform
        self.max_label_len = max_label_len
        self.add_sos_eos   = add_sos_eos

        self.samples: List[Tuple[str, str]] = []   # (image_path, label)
        self._load_annotations(annotation_file)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_annotations(self, annotation_file: str):
        skipped = 0
        with open(annotation_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    skipped += 1
                    continue
                img_rel, label = parts
                label = unicodedata.normalize("NFC", label.strip())
                if not label or len(label) > self.max_label_len:
                    skipped += 1
                    continue
                img_path = os.path.join(self.image_root, img_rel)
                self.samples.append((img_path, label))

        print(
            f"[Dataset] Loaded {len(self.samples)} samples "
            f"(skipped {skipped}) from {annotation_file}"
        )

        # Validate that all image files exist (check first 50)
        missing = []
        for img_path, label in self.samples[:min(50, len(self.samples))]:
            if not os.path.exists(img_path):
                missing.append(img_path)
        if missing:
            raise FileNotFoundError(
                f"[Dataset] {len(missing)} image files not found (showing first 5):\n"
                + "\n".join(missing[:5])
                + f"\nCheck that image_root='{self.image_root}' matches your labels.txt format."
            )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        # Load image (no silent fallback — crash on missing files)
        img = Image.open(img_path).convert("RGB")

        # Apply transform → (tensor [1, H, W], scaled_width)
        img_tensor, img_width = self.transform(img)

        # Encode label
        label_ids = self.vocab.encode(label)
        if self.add_sos_eos:
            label_ids = [self.vocab.sos_id] + label_ids + [self.vocab.eos_id]

        label_len = len(label_ids)
        return img_tensor, img_width, label_ids, label_len

    # ------------------------------------------------------------------
    # Extra utilities
    # ------------------------------------------------------------------

    def get_label(self, idx: int) -> str:
        return self.samples[idx][1]

    def sample_batch(self, n: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Utility: fetch a small batch for debugging."""
        indices = list(range(min(n, len(self))))
        items   = [self[i] for i in indices]
        return collate_fn_pad(items)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    annotation_file: str,
    image_root: str,
    vocab: TeluguVocab,
    split: str = "train",                  # "train" | "val" | "test"
    batch_size: int = 64,
    num_workers: int = 4,
    max_label_len: int = 32,
    add_sos_eos: bool = False,
    use_elastic: bool = False,
) -> DataLoader:
    """
    Build a DataLoader for a given split.

    Parameters
    ----------
    split : 'train' uses TrainTransform (with augmentation),
            'val' or 'test' uses ValTransform (deterministic).
    add_sos_eos : set True for autoregressive models, False for CTC.
    """
    if split == "train":
        transform = TrainTransform(use_elastic=use_elastic)
        shuffle   = True
    else:
        transform = ValTransform()
        shuffle   = False

    dataset = TeluguHTRDataset(
        annotation_file = annotation_file,
        image_root      = image_root,
        vocab           = vocab,
        transform       = transform,
        max_label_len   = max_label_len,
        add_sos_eos     = add_sos_eos,
    )

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        collate_fn  = collate_fn_pad,
        pin_memory  = True,
        drop_last   = (split == "train"),
    )
    return loader


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.dataset <annotation_file> <image_root>")
        sys.exit(1)

    ann_file   = sys.argv[1]
    image_root = sys.argv[2]

    vocab  = DEFAULT_VOCAB
    loader = build_dataloader(
        ann_file, image_root, vocab,
        split       = "train",
        batch_size  = 4,
        num_workers = 0,
        add_sos_eos = True,
    )

    images, labels, lengths, widths = next(iter(loader))
    print(f"Batch images shape : {images.shape}")    # [4, 1, 64, 512]
    print(f"Batch labels shape : {labels.shape}")    # [4, T_max]
    print(f"Label lengths      : {lengths}")
    print(f"Image widths       : {widths}")
    for i in range(len(lengths)):
        decoded = vocab.decode(labels[i].tolist())
        print(f"  Sample {i}: '{decoded}'  (len={lengths[i]}, w={widths[i]})")

```

### src/checkpoint_manager.py
```python
"""
src/checkpoint_manager.py

Rolling 2-slot checkpoint manager.

DESIGN
──────
At most 3 files exist on disk at any time:

    checkpoints/<run>/
        best.pt        ← best val CER ever seen
        current.pt     ← end of most recent epoch
        previous.pt    ← end of the epoch before that

Algorithm on each save:
    1. If current.pt exists  →  rename it to previous.pt  (overwrites previous.pt)
    2. Write new current.pt
    3. If new val CER < best CER  →  copy current.pt to best.pt

DISK COST
─────────
For a ~200 MB model (fp32) or ~100 MB (fp16):
    3 files × 200 MB = 600 MB total  — fixed forever, no growth.

ROLLBACK
─────────
    - 1 epoch back  : load previous.pt
    - Best ever     : load best.pt
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """
    Manages rolling 2-slot checkpoints: best / current / previous.

    Parameters
    ----------
    save_dir  : directory where checkpoint files are stored.
    model     : the nn.Module to save.
    optimizer : the optimizer.
    scheduler : the LR scheduler.
    """

    BEST_NAME     = "best.pt"
    CURRENT_NAME  = "current.pt"
    PREVIOUS_NAME = "previous.pt"

    def __init__(
        self,
        save_dir:  str,
        model,
        optimizer,
        scheduler = None,
    ):
        self.save_dir  = save_dir
        self.model     = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        os.makedirs(save_dir, exist_ok=True)

        self.best_cer: float = float("inf")

        # Paths
        self.best_path     = os.path.join(save_dir, self.BEST_NAME)
        self.current_path  = os.path.join(save_dir, self.CURRENT_NAME)
        self.previous_path = os.path.join(save_dir, self.PREVIOUS_NAME)

    # ── Core save ─────────────────────────────────────────────────

    def save(self, epoch: int, val_cer: float, extra: Optional[Dict] = None) -> str:
        """
        Save at the end of an epoch.

        Steps:
          1. Promote current → previous  (if current exists)
          2. Write new current
          3. If val_cer < best_cer: update best

        Parameters
        ----------
        epoch   : current epoch number.
        val_cer : validation CER (used to decide whether to update best).
        extra   : any extra keys to include in the checkpoint dict.

        Returns
        -------
        Path of the file that was just written (current_path).
        """
        # Step 1: promote current → previous
        if os.path.exists(self.current_path):
            shutil.move(self.current_path, self.previous_path)

        # Step 2: update best_cer BEFORE building dict so checkpoint has correct value
        improved = False
        if val_cer < self.best_cer:
            self.best_cer = val_cer
            improved = True

        # Step 3: write new current (now contains the updated best_cer)
        ckpt = self._build_dict(epoch, val_cer, extra)
        torch.save(ckpt, self.current_path)

        # Step 4: copy to best if improved
        if improved:
            shutil.copy2(self.current_path, self.best_path)

        self._print_status(epoch, val_cer, improved)
        return self.current_path


    # ── Load helpers ──────────────────────────────────────────────

    def load(self, path: str, device: str = "cpu") -> Dict[str, Any]:
        """
        Load a checkpoint from any path (current, previous, best, or thermal).
        Restores model, optimizer, and scheduler weights in-place.

        Returns the full checkpoint dict (contains 'epoch', 'val_cer', etc).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=device, weights_only=False)

        self.model.load_state_dict(ckpt["model"])

        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

        if self.scheduler is not None and "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])

        self.best_cer = ckpt.get("best_cer", float("inf"))

        print(
            f"[CheckpointManager] Loaded '{os.path.basename(path)}' — "
            f"epoch={ckpt.get('epoch', '?')}  "
            f"val_cer={ckpt.get('val_cer', '?')}"
        )
        return ckpt

    def load_best(self, device: str = "cpu") -> Dict:
        return self.load(self.best_path, device)

    def load_current(self, device: str = "cpu") -> Dict:
        return self.load(self.current_path, device)

    def load_previous(self, device: str = "cpu") -> Dict:
        """Roll back one epoch."""
        return self.load(self.previous_path, device)

    # ── Status ───────────────────────────────────────────────────

    def disk_usage(self) -> str:
        """Return a human-readable summary of files on disk."""
        files = [
            (self.best_path,     "best"),
            (self.current_path,  "current"),
            (self.previous_path, "previous"),
        ]
        lines = []
        total_bytes = 0
        for path, label in files:
            if os.path.exists(path):
                size = os.path.getsize(path)
                total_bytes += size
                lines.append(f"  {label:>10}.pt  {size / 1024 / 1024:>8.1f} MB")
            else:
                lines.append(f"  {label:>10}.pt  {'—':>8}")
        lines.append(f"  {'TOTAL':>10}       {total_bytes / 1024 / 1024:>8.1f} MB")
        return "\n".join(lines)

    def _print_status(self, epoch: int, val_cer: float, improved: bool):
        tag = "  ★ NEW BEST" if improved else ""
        print(
            f"  [Checkpoint] epoch={epoch}  val_cer={val_cer:.4f}"
            f"  best={self.best_cer:.4f}{tag}"
        )
        if epoch % 5 == 0:   # show disk usage every 5 epochs
            print(f"\n  Disk usage:\n{self.disk_usage()}\n")

    def _build_dict(
        self, epoch: int, val_cer: float, extra: Optional[Dict]
    ) -> Dict:
        ckpt: Dict[str, Any] = {
            "epoch":    epoch,
            "val_cer":  val_cer,
            "best_cer": self.best_cer,
            "model":    self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            ckpt["scheduler"] = self.scheduler.state_dict()
        if extra:
            ckpt.update(extra)
        return ckpt

```

### src/training_logger.py
```python
"""
src/training_logger.py

Comprehensive training logger that captures:
  - GPU usage (memory, utilization, temperature)
  - Training performance (samples/sec, epoch time)
  - System info (GPU model, CUDA version, Python version)
  - Per-epoch metrics (loss, CER, WER)
  - Full training summary

All logs are saved as both:
  - Human-readable .log file (for quick inspection)
  - JSON file (for programmatic analysis and paper)
"""

from __future__ import annotations
import json
import os
import platform
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import torch


class TrainingLogger:
    """
    Comprehensive logger for training runs.

    Usage:
        logger = TrainingLogger(log_dir="logs/ar", model_name="AR Transformer")
        logger.log_system_info(model)         # call once at start
        logger.log_epoch_start(epoch)          # call at epoch start
        logger.log_gpu_stats(epoch, step)      # call periodically
        logger.log_epoch_end(epoch, metrics)   # call at epoch end
        logger.log_training_complete(metrics)  # call at end
    """

    def __init__(self, log_dir: str, model_name: str = "model"):
        self.log_dir = log_dir
        self.model_name = model_name
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        self.json_file = os.path.join(log_dir, f"training_{timestamp}.json")

        self.training_start_time = time.time()
        self.epoch_start_time = None
        self.epoch_times = []
        self.epoch_metrics = []
        self.gpu_snapshots = []
        self.system_info = {}

        self._write_log(f"{'='*70}")
        self._write_log(f"  Training Log — {model_name}")
        self._write_log(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_log(f"{'='*70}\n")

    def _write_log(self, msg: str):
        """Append message to log file and print."""
        print(msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _save_json(self):
        """Save all collected data to JSON."""
        data = {
            "model_name": self.model_name,
            "system_info": self.system_info,
            "training_start": datetime.fromtimestamp(
                self.training_start_time
            ).isoformat(),
            "epoch_metrics": self.epoch_metrics,
            "epoch_times_seconds": self.epoch_times,
            "gpu_snapshots": self.gpu_snapshots,
        }
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    # ── System Info ──────────────────────────────────────────────

    def log_system_info(self, model=None, config: dict = None):
        """Log system info once at the start of training."""
        info = {
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = str(torch.backends.cudnn.version())
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_total_gb"] = round(props.total_memory / 1e9, 2)
            info["gpu_compute_capability"] = f"{props.major}.{props.minor}"

        if model is not None:
            info["model_parameters"] = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            info["model_parameters_formatted"] = f"{info['model_parameters']:,}"

        if config is not None:
            info["config"] = config

        self.system_info = info

        self._write_log("── System Info ─────────────────────────────────────")
        self._write_log(f"  Python     : {info['python_version']}")
        self._write_log(f"  PyTorch    : {info['pytorch_version']}")
        self._write_log(f"  Platform   : {info['platform']}")
        if info["cuda_available"]:
            self._write_log(f"  CUDA       : {info['cuda_version']}")
            self._write_log(f"  cuDNN      : {info['cudnn_version']}")
            self._write_log(f"  GPU        : {info['gpu_name']}")
            self._write_log(f"  GPU Memory : {info['gpu_memory_total_gb']} GB")
            self._write_log(f"  Compute Cap: {info['gpu_compute_capability']}")
        if model is not None:
            self._write_log(f"  Parameters : {info['model_parameters_formatted']}")
        self._write_log("")
        self._save_json()

    # ── GPU Stats ────────────────────────────────────────────────

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU memory and utilization stats."""
        if not torch.cuda.is_available():
            return {"gpu_available": False}

        stats = {
            "gpu_available": True,
            "gpu_memory_allocated_mb": round(
                torch.cuda.memory_allocated() / 1e6, 1
            ),
            "gpu_memory_reserved_mb": round(
                torch.cuda.memory_reserved() / 1e6, 1
            ),
            "gpu_memory_total_mb": round(
                torch.cuda.get_device_properties(0).total_memory / 1e6, 1
            ),
            "gpu_max_memory_allocated_mb": round(
                torch.cuda.max_memory_allocated() / 1e6, 1
            ),
        }
        total = stats["gpu_memory_total_mb"]
        used = stats["gpu_memory_allocated_mb"]
        stats["gpu_memory_utilization_pct"] = round(used / total * 100, 1)
        stats["gpu_memory_free_mb"] = round(total - used, 1)

        return stats

    def log_gpu_stats(self, epoch: int, step: int = 0):
        """Log GPU stats at a specific point in training."""
        stats = self.get_gpu_stats()
        if not stats["gpu_available"]:
            return

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "step": step,
            **stats,
        }
        self.gpu_snapshots.append(snapshot)

        self._write_log(
            f"  [GPU] Mem: {stats['gpu_memory_allocated_mb']:.0f}MB / "
            f"{stats['gpu_memory_total_mb']:.0f}MB "
            f"({stats['gpu_memory_utilization_pct']:.1f}%) | "
            f"Peak: {stats['gpu_max_memory_allocated_mb']:.0f}MB"
        )

    # ── Epoch Tracking ───────────────────────────────────────────

    def log_epoch_start(self, epoch: int):
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()
        self._write_log(f"\n{'─'*70}")
        self._write_log(
            f"  Epoch {epoch} started at "
            f"{datetime.now().strftime('%H:%M:%S')}"
        )
        self._write_log(f"{'─'*70}")

    def log_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_cer: float = None,
        val_wer: float = None,
        val_loss: float = None,
        extra_metrics: dict = None,
        samples_processed: int = 0,
    ):
        """Log metrics at the end of an epoch."""
        elapsed = time.time() - (self.epoch_start_time or self.training_start_time)
        self.epoch_times.append(elapsed)

        samples_per_sec = samples_processed / max(elapsed, 1e-6) if samples_processed else 0
        total_elapsed = time.time() - self.training_start_time
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)

        metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_cer": round(val_cer, 6) if val_cer is not None else None,
            "val_wer": round(val_wer, 6) if val_wer is not None else None,
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
            "epoch_time_seconds": round(elapsed, 1),
            "samples_per_second": round(samples_per_sec, 1),
            "total_time_seconds": round(total_elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }

        # Add GPU stats
        gpu_stats = self.get_gpu_stats()
        metrics["gpu_memory_allocated_mb"] = gpu_stats.get("gpu_memory_allocated_mb", 0)
        metrics["gpu_memory_peak_mb"] = gpu_stats.get("gpu_max_memory_allocated_mb", 0)

        if extra_metrics:
            metrics.update(extra_metrics)

        self.epoch_metrics.append(metrics)

        # Pretty print
        self._write_log(f"\n  ┌── Epoch {epoch} Summary ──────────────────────────")
        self._write_log(f"  │ Train Loss  : {train_loss:.4f}")
        if val_cer is not None:
            self._write_log(f"  │ Val CER     : {val_cer:.4f} ({val_cer*100:.2f}%)")
        if val_wer is not None:
            self._write_log(f"  │ Val WER     : {val_wer:.4f} ({val_wer*100:.2f}%)")
        self._write_log(f"  │ Epoch Time  : {elapsed:.0f}s ({timedelta(seconds=int(elapsed))})")
        self._write_log(f"  │ Speed       : {samples_per_sec:.1f} samples/sec")
        self._write_log(
            f"  │ GPU Memory  : {gpu_stats.get('gpu_memory_allocated_mb', 0):.0f}MB "
            f"(peak: {gpu_stats.get('gpu_max_memory_allocated_mb', 0):.0f}MB)"
        )
        self._write_log(f"  │ Total Time  : {timedelta(seconds=int(total_elapsed))}")

        # ETA
        remaining_epochs = extra_metrics.get("total_epochs", 50) - epoch if extra_metrics else 0
        if remaining_epochs > 0:
            eta_seconds = avg_epoch_time * remaining_epochs
            eta = timedelta(seconds=int(eta_seconds))
            self._write_log(f"  │ ETA         : {eta} ({remaining_epochs} epochs left)")

        self._write_log(f"  └─────────────────────────────────────────────────\n")
        self._save_json()

    # ── Training Complete ────────────────────────────────────────

    def log_training_complete(self, best_cer: float = None, best_epoch: int = None):
        """Log final training summary."""
        total_time = time.time() - self.training_start_time

        self._write_log(f"\n{'='*70}")
        self._write_log(f"  TRAINING COMPLETE — {self.model_name}")
        self._write_log(f"{'='*70}")
        self._write_log(f"  Total Time       : {timedelta(seconds=int(total_time))}")
        self._write_log(f"  Total Epochs     : {len(self.epoch_times)}")

        if self.epoch_times:
            self._write_log(f"  Avg Epoch Time   : {sum(self.epoch_times)/len(self.epoch_times):.0f}s")

        if best_cer is not None:
            self._write_log(f"  Best Val CER     : {best_cer:.4f} ({best_cer*100:.2f}%)")
        if best_epoch is not None:
            self._write_log(f"  Best Epoch       : {best_epoch}")

        gpu_stats = self.get_gpu_stats()
        if gpu_stats["gpu_available"]:
            self._write_log(f"  Peak GPU Memory  : {gpu_stats['gpu_max_memory_allocated_mb']:.0f}MB")

        # Find best and worst epochs
        if self.epoch_metrics:
            cers = [(m["epoch"], m["val_cer"]) for m in self.epoch_metrics if m.get("val_cer") is not None]
            if cers:
                best = min(cers, key=lambda x: x[1])
                worst = max(cers, key=lambda x: x[1])
                self._write_log(f"  Best Epoch CER   : Epoch {best[0]} → {best[1]:.4f}")
                self._write_log(f"  Worst Epoch CER  : Epoch {worst[0]} → {worst[1]:.4f}")

        self._write_log(f"\n  Log file  : {self.log_file}")
        self._write_log(f"  JSON file : {self.json_file}")
        self._write_log(f"{'='*70}\n")

        # Final JSON save
        self.system_info["total_training_time_seconds"] = round(total_time, 1)
        self.system_info["total_training_time_human"] = str(timedelta(seconds=int(total_time)))
        if best_cer is not None:
            self.system_info["best_val_cer"] = best_cer
        self._save_json()

```

### src/train_ctc.py
```python
"""
src/train_ctc.py

Training script for the CTC baseline model.

Usage:
    python -m src.train_ctc --config configs/ctc_config.yaml
    python -m src.train_ctc --config configs/ctc_config.yaml --resume checkpoints/ctc/current.pt
"""

from __future__ import annotations
import argparse
import os
import time
import unicodedata

import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from src.vocab import TeluguVocab
from src.dataset import build_dataloader
from src.models.ctc_model import CTCModel
from src.evaluate import compute_cer_wer
from src.checkpoint_manager import CheckpointManager
from src.training_logger import TrainingLogger


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_or_build_vocab(vocab_path, train_ann, val_ann) -> TeluguVocab:
    if os.path.exists(vocab_path):
        vocab = TeluguVocab.load(vocab_path)
    else:
        print("[train_ctc] Building vocab from training data ...")
        vocab = TeluguVocab.from_annotation_files([train_ann], build_matrix=True)
        vocab.validate_against_split(val_ann, split_name="val")
        vocab.save(vocab_path)
    return vocab


# ═══════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, loader, vocab, device, writer, epoch):
    model.eval()
    all_preds, all_gts = [], []
    total_loss   = 0.0
    n_batches    = 0
    pred_lengths = []

    for images, labels, label_lens, image_widths in loader:
        images       = images.to(device)
        labels       = labels.to(device)
        label_lens   = label_lens.to(device)
        image_widths = image_widths.to(device)

        loss = model.compute_loss(images, labels, label_lens, input_widths=image_widths)
        total_loss += loss.item()
        n_batches  += 1

        pred_ids = model.greedy_decode(images, input_widths=image_widths)
        pred_lengths.extend([len(p) for p in pred_ids])

        for i, (pred, llen) in enumerate(zip(pred_ids, label_lens.tolist())):
            gt_ids = labels[i, :llen].tolist()
            all_preds.append(unicodedata.normalize("NFC", vocab.decode(pred)))
            all_gts.append(unicodedata.normalize("NFC", vocab.decode(gt_ids)))

    cer, wer     = compute_cer_wer(all_preds, all_gts)
    avg_loss     = total_loss / max(n_batches, 1)
    avg_pred_len = sum(pred_lengths) / max(len(pred_lengths), 1)

    writer.add_scalar("val/loss",         avg_loss,     epoch)
    writer.add_scalar("val/CER",          cer,          epoch)
    writer.add_scalar("val/WER",          wer,          epoch)
    writer.add_scalar("val/avg_pred_len", avg_pred_len, epoch)

    print(f"  [Val epoch {epoch}] loss={avg_loss:.4f}  CER={cer:.4f}  "
          f"WER={wer:.4f}  avg_pred_len={avg_pred_len:.1f}")
    return cer


# ═══════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════

def train(config_path: str, resume_path: str = None):
    cfg  = load_config(config_path)
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        
    # Reproducibility
    torch.manual_seed(42)
    import random, numpy as np
    random.seed(42)
    np.random.seed(42)

    print(f"[train_ctc] Device: {device}")

    # ── Training Logger ──────────────────────────────────────────
    tlogger = TrainingLogger(log_dir=tcfg["log_dir"], model_name="CTC Baseline")

    # ── Vocab ────────────────────────────────────────────────────
    vocab = load_or_build_vocab(
        tcfg["vocab_path"],
        dcfg["train_annotation"],
        dcfg["val_annotation"],
    )
    vocab_size = len(vocab)
    print(f"[train_ctc] Vocab size: {vocab_size}")

    # ── Data ─────────────────────────────────────────────────────
    train_loader = build_dataloader(
        dcfg["train_annotation"], dcfg["train_image_root"], vocab,
        split         = "train",
        batch_size    = tcfg["batch_size"],
        num_workers   = dcfg["num_workers"],
        max_label_len = dcfg["max_label_len"],
        add_sos_eos   = False,
        use_elastic   = dcfg.get("use_elastic", False),
    )
    val_loader = build_dataloader(
        dcfg["val_annotation"], dcfg["val_image_root"], vocab,
        split         = "val",
        batch_size    = tcfg["batch_size"],
        num_workers   = dcfg["num_workers"],
        max_label_len = dcfg["max_label_len"],
        add_sos_eos   = False,
    )

    # ── Model ────────────────────────────────────────────────────
    model = CTCModel(
        vocab_size  = vocab_size,
        d_model     = mcfg["d_model"],
        lstm_hidden = mcfg["lstm_hidden"],
        lstm_layers = mcfg["lstm_layers"],
        dropout     = mcfg["dropout"],
        pretrained  = mcfg["pretrained"],
    ).to(device)
    print(f"[train_ctc] Parameters: {model.count_params():,}")
    tlogger.log_system_info(model=model, config=cfg)

    # ── Optimiser + Scheduler ────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr           = tcfg["lr"],
        weight_decay = tcfg["weight_decay"],
    )
    total_steps = len(train_loader) * tcfg["epochs"]
    scheduler   = OneCycleLR(
        optimizer,
        max_lr          = tcfg["lr"],
        total_steps     = total_steps,
        pct_start       = 0.1,
        anneal_strategy = "cos",
    )
    scaler = GradScaler("cuda", enabled=tcfg["mixed_precision"])

    # ── Checkpoint manager (rolling 2-slot) ───────────────────────
    ckpt_mgr = CheckpointManager(
        save_dir  = tcfg["save_dir"],
        model     = model,
        optimizer = optimizer,
        scheduler = scheduler,
    )

    start_epoch = 1
    if resume_path and os.path.exists(resume_path):
        ckpt = ckpt_mgr.load(resume_path, device=str(device))
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[train_ctc] Resuming from epoch {start_epoch}")

    # ── Logging ──────────────────────────────────────────────────
    writer      = SummaryWriter(tcfg["log_dir"])
    global_step = (start_epoch - 1) * len(train_loader)

    # ── Training ─────────────────────────────────────────────────
    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        t0 = time.time()
        tlogger.log_epoch_start(epoch)

        for batch_idx, (images, labels, label_lens, image_widths) in enumerate(train_loader):
            images       = images.to(device)
            labels       = labels.to(device)
            label_lens   = label_lens.to(device)
            image_widths = image_widths.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=tcfg["mixed_precision"]):
                loss = model.compute_loss(images, labels, label_lens, input_widths=image_widths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss  += loss.item()
            epoch_samples += images.size(0)
            global_step += 1

            if batch_idx % tcfg["log_interval"] == 0:
                lr_now = scheduler.get_last_lr()[0]
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr",   lr_now,      global_step)
                print(f"  Epoch {epoch}/{tcfg['epochs']}  "
                      f"Step {batch_idx}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  lr={lr_now:.2e}")

            # Log GPU stats every 200 steps
            if batch_idx % 200 == 0 and batch_idx > 0:
                tlogger.log_gpu_stats(epoch, batch_idx)

        avg_epoch_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}] avg_loss={avg_epoch_loss:.4f}  time={elapsed:.0f}s")
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)

        # ── Validate + save rolling checkpoint ───────────────────
        if epoch % tcfg["val_interval"] == 0:
            val_cer = validate(model, val_loader, vocab, device, writer, epoch)
            ckpt_mgr.save(epoch=epoch, val_cer=val_cer)

            # Log epoch summary
            tlogger.log_epoch_end(
                epoch=epoch,
                train_loss=avg_epoch_loss,
                val_cer=val_cer,
                samples_processed=epoch_samples,
                extra_metrics={"total_epochs": tcfg["epochs"]},
            )

    writer.close()
    tlogger.log_training_complete(
        best_cer=ckpt_mgr.best_cer,
    )
    print(f"\nDisk usage:\n{ckpt_mgr.disk_usage()}")


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ctc_config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Resume from: current.pt / previous.pt / best.pt")
    args = parser.parse_args()
    train(args.config, args.resume)

```

### src/train_ar.py
```python
"""
src/train_ar.py

Training script for the autoregressive Transformer decoder model.

Usage:
    python -m src.train_ar --config configs/ar_config.yaml
    python -m src.train_ar --config configs/ar_config.yaml --resume checkpoints/ar/current.pt
"""

from __future__ import annotations
import argparse
import math
import os
import time
import unicodedata

import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from src.vocab import TeluguVocab
from src.dataset import build_dataloader
from src.models.ar_model import ARModel
from src.evaluate import compute_cer_wer
from src.checkpoint_manager import CheckpointManager
from src.training_logger import TrainingLogger


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_or_build_vocab(vocab_path, train_ann, val_ann) -> TeluguVocab:
    if os.path.exists(vocab_path):
        vocab = TeluguVocab.load(vocab_path)
    else:
        print("[train_ar] Building vocab from training data ...")
        vocab = TeluguVocab.from_annotation_files([train_ann], build_matrix=True)
        vocab.validate_against_split(val_ann, split_name="val")
        vocab.save(vocab_path)
    return vocab


def warmup_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, loader, vocab, device, writer, epoch, constrain: bool, constrain_penalty: float = None, tb_tag: str = "val"):
    model.eval()
    all_preds, all_gts = [], []
    total_loss   = 0.0
    n_batches    = 0
    pred_lengths = []

    for images, labels, label_lens, image_widths in loader:
        images     = images.to(device)
        labels     = labels.to(device)
        label_lens = label_lens.to(device)
        image_widths = image_widths.to(device)

        # 1. Validation loss (teacher-forcing)
        loss, _, _ = model.compute_loss(images, labels, label_lens, input_widths=image_widths)
        total_loss += loss.item()
        n_batches  += 1

        pred_ids = model.greedy_decode(
            images,
            max_len   = 36,
            vocab     = vocab if constrain else None,
            constrain = constrain,
            constrain_penalty = constrain_penalty,
            input_widths = image_widths
        )
        pred_lengths.extend([len(p) for p in pred_ids])

        for i, (pred, lab, llen) in enumerate(zip(pred_ids, labels, label_lens.tolist())):
            gt_ids = lab[1:llen - 1].tolist()
            all_preds.append(unicodedata.normalize("NFC", vocab.decode(pred)))
            all_gts.append(unicodedata.normalize("NFC", vocab.decode(gt_ids)))

    cer, wer     = compute_cer_wer(all_preds, all_gts)
    avg_loss     = total_loss / max(n_batches, 1)
    avg_pred_len = sum(pred_lengths) / max(len(pred_lengths), 1)

    writer.add_scalar(f"{tb_tag}/loss",         avg_loss,     epoch)
    writer.add_scalar(f"{tb_tag}/CER",          cer,          epoch)
    writer.add_scalar(f"{tb_tag}/WER",          wer,          epoch)
    writer.add_scalar(f"{tb_tag}/avg_pred_len", avg_pred_len, epoch)

    tag_label = "(constrained)" if constrain else "(unconstrained)"
    print(f"  [Val epoch {epoch} {tag_label}] loss={avg_loss:.4f}  "
          f"CER={cer:.4f}  WER={wer:.4f}  avg_pred_len={avg_pred_len:.1f}")
    return cer


# ═══════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════

def train(config_path: str, resume_path: str = None):
    cfg  = load_config(config_path)
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        
    # Reproducibility
    torch.manual_seed(42)
    import random, numpy as np
    random.seed(42)
    np.random.seed(42)
    
    print(f"[train_ar] Device: {device}")

    # ── Training Logger ──────────────────────────────────────────
    tlogger = TrainingLogger(log_dir=tcfg["log_dir"], model_name="AR Transformer")

    # ── Vocab ────────────────────────────────────────────────────
    vocab = load_or_build_vocab(
        tcfg["vocab_path"],
        dcfg["train_annotation"],
        dcfg["val_annotation"],
    )
    vocab_size = len(vocab)
    print(f"[train_ar] Vocab size: {vocab_size}")

    # ── Data ─────────────────────────────────────────────────────
    train_loader = build_dataloader(
        dcfg["train_annotation"], dcfg["train_image_root"], vocab,
        split         = "train",
        batch_size    = tcfg["batch_size"],
        num_workers   = dcfg["num_workers"],
        max_label_len = dcfg["max_label_len"],
        add_sos_eos   = True,
        use_elastic   = dcfg.get("use_elastic", False),
    )
    val_loader = build_dataloader(
        dcfg["val_annotation"], dcfg["val_image_root"], vocab,
        split         = "val",
        batch_size    = tcfg["batch_size"],
        num_workers   = dcfg["num_workers"],
        max_label_len = dcfg["max_label_len"],
        add_sos_eos   = True,
    )

    # ── Model ────────────────────────────────────────────────────
    model = ARModel(
        vocab_size         = vocab_size,
        sos_id             = vocab.sos_id,
        eos_id             = vocab.eos_id,
        d_model            = mcfg["d_model"],
        nhead              = mcfg["nhead"],
        num_decoder_layers = mcfg["num_decoder_layers"],
        dim_feedforward    = mcfg["dim_feedforward"],
        dropout            = mcfg["dropout"],
        max_label_len      = mcfg["max_label_len"],
        label_smoothing    = mcfg["label_smoothing"],
        pretrained         = mcfg["pretrained"],
        num_encoder_layers = mcfg.get("num_encoder_layers", 2),
        high_res_temporal  = mcfg.get("high_res_temporal", False),
        ctc_weight         = mcfg.get("ctc_weight", 0.3),
    ).to(device)
    print(f"[train_ar] Parameters: {model.count_params():,}")
    tlogger.log_system_info(model=model, config=cfg)

    # ── Optimiser + Scheduler ────────────────────────────────────
    optimizer   = AdamW(
        model.parameters(),
        lr           = tcfg["lr"],
        weight_decay = tcfg["weight_decay"],
        betas        = (0.9, 0.98),
    )
    total_steps = len(train_loader) * tcfg["epochs"]
    scheduler   = warmup_cosine_schedule(optimizer, tcfg["warmup_steps"], total_steps)
    scaler = GradScaler("cuda", enabled=tcfg["mixed_precision"])

    # ── Checkpoint manager (rolling 2-slot) ───────────────────────
    ckpt_mgr = CheckpointManager(
        save_dir  = tcfg["save_dir"],
        model     = model,
        optimizer = optimizer,
        scheduler = scheduler,
    )

    start_epoch = 1
    if resume_path and os.path.exists(resume_path):
        ckpt = ckpt_mgr.load(resume_path, device=str(device))
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[train_ar] Resuming from epoch {start_epoch}")

    # ── Logging ──────────────────────────────────────────────────
    writer      = SummaryWriter(tcfg["log_dir"])
    constrain   = tcfg.get("constrain_decode", True)
    global_step = (start_epoch - 1) * len(train_loader)

    # ── Training ─────────────────────────────────────────────────
    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        t0 = time.time()
        tlogger.log_epoch_start(epoch)

        for batch_idx, (images, labels, label_lens, image_widths) in enumerate(train_loader):
            images     = images.to(device)
            labels     = labels.to(device)
            label_lens = label_lens.to(device)
            image_widths = image_widths.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=tcfg["mixed_precision"]):
                loss, ce_loss, ctc_loss = model.compute_loss(images, labels, label_lens, input_widths=image_widths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss  += loss.item()
            epoch_samples += images.size(0)
            global_step += 1

            if batch_idx % tcfg["log_interval"] == 0:
                lr_now = scheduler.get_last_lr()[0]
                writer.add_scalar("train/loss",     loss.item(),     global_step)
                writer.add_scalar("train/ce_loss",  ce_loss.item(),  global_step)
                writer.add_scalar("train/ctc_loss", ctc_loss.item(), global_step)
                writer.add_scalar("train/lr",       lr_now,          global_step)
                print(f"  Epoch {epoch}/{tcfg['epochs']}  "
                      f"Step {batch_idx}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  ce={ce_loss.item():.4f}  ctc={ctc_loss.item():.4f}  lr={lr_now:.2e}")

            # Log GPU stats every 200 steps
            if batch_idx % 200 == 0 and batch_idx > 0:
                tlogger.log_gpu_stats(epoch, batch_idx)

        avg_epoch_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}] avg_loss={avg_epoch_loss:.4f}  time={elapsed:.0f}s")
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)

        # ── Validate + save rolling checkpoint ───────────────────
        constrain = tcfg.get("constrain_decode", True)
        constrain_penalty = tcfg.get("constrain_penalty", None)
        # Log constrained metrics with a distinct TensorBoard tag
        if constrain:
            validate(model, val_loader, vocab, device, writer, epoch,
                     constrain=True, constrain_penalty=constrain_penalty,
                     tb_tag="val_constrained")

        # Unbiased checkpointing: always evaluate unconstrained CER for checkpoint selection
        val_cer = validate(model, val_loader, vocab, device, writer, epoch,
                           constrain=False, constrain_penalty=None,
                           tb_tag="val")
        ckpt_mgr.save(epoch=epoch, val_cer=val_cer)

        # Log epoch summary
        tlogger.log_epoch_end(
            epoch=epoch,
            train_loss=avg_epoch_loss,
            val_cer=val_cer,
            samples_processed=epoch_samples,
            extra_metrics={"total_epochs": tcfg["epochs"]},
        )

    writer.close()
    tlogger.log_training_complete(
        best_cer=ckpt_mgr.best_cer,
    )
    print(f"\nDisk usage:\n{ckpt_mgr.disk_usage()}")


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ar_config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Resume from: current.pt / previous.pt / best.pt")
    args = parser.parse_args()
    train(args.config, args.resume)

```

### src/evaluate.py
```python
"""
src/evaluate.py

Evaluation utilities for Telugu HTR.

Provides:
  - compute_cer_wer()          — CER and WER over a list of predictions
  - evaluate_model_ctc()       — full test-set eval for CTC model
  - evaluate_model_ar()        — full test-set eval for AR model (greedy + beam)
  - breakdown_by_virama()      — split results by compound-character words
  - print_error_examples()     — show worst-case failures
  - confusion_matrix_chars()   — character confusion analysis

Usage:
    python -m src.evaluate \
        --model_type ctc \
        --checkpoint checkpoints/ctc/best.pt \
        --config configs/ctc_config.yaml \
        --split test
"""

from __future__ import annotations
import argparse
import os
import unicodedata
import time
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

import editdistance
import torch
import yaml

from src.vocab import TeluguVocab, VIRAMA


# ═══════════════════════════════════════════════════════════════════
# Core metrics
# ═══════════════════════════════════════════════════════════════════

def compute_cer_wer(
    predictions: List[str],
    ground_truths: List[str],
) -> Tuple[float, float]:
    """
    Compute Character Error Rate (CER) and Word Error Rate (WER).

    CER = sum(edit_distance(pred, gt)) / sum(len(gt))
    WER = number of words where pred != gt / total words

    Returns (CER, WER) as floats in [0, 1].
    """
    assert len(predictions) == len(ground_truths), \
        f"Length mismatch: {len(predictions)} vs {len(ground_truths)}"

    total_chars  = 0
    total_edits  = 0
    word_errors  = 0

    for pred, gt in zip(predictions, ground_truths):
        # CER
        total_edits += editdistance.eval(list(pred), list(gt))
        total_chars += max(len(gt), 1)

        # WER
        if pred != gt:
            word_errors += 1

    cer = total_edits / max(total_chars, 1)
    wer = word_errors / max(len(predictions), 1)
    return cer, wer


def compute_bootstrap_ci(
    predictions: List[str],
    ground_truths: List[str],
    n_resamples: int = 1000,
) -> Tuple[float, float, float]:
    """
    Compute 95% Confidence Interval for CER using bootstrap resampling.
    Returns (mean_cer, lower_bound, upper_bound).
    """
    import numpy as np
    n = len(predictions)
    if n == 0:
        return 0.0, 0.0, 0.0
        
    preds = np.array(predictions)
    gts = np.array(ground_truths)
    
    bootstrap_cers = []
    for _ in range(n_resamples):
        indices = np.random.choice(n, n, replace=True)
        resample_preds = preds[indices].tolist()
        resample_gts = gts[indices].tolist()
        cer, _ = compute_cer_wer(resample_preds, resample_gts)
        bootstrap_cers.append(cer)
        
    bootstrap_cers = np.array(bootstrap_cers)
    mean_cer = np.mean(bootstrap_cers)
    lower = np.percentile(bootstrap_cers, 2.5)
    upper = np.percentile(bootstrap_cers, 97.5)
    return float(mean_cer), float(lower), float(upper)


# ═══════════════════════════════════════════════════════════════════
# Per-sample result dataclass
# ═══════════════════════════════════════════════════════════════════

class SampleResult:
    __slots__ = ("image_path", "gt", "pred", "cer", "correct")

    def __init__(self, image_path: str, gt: str, pred: str):
        self.image_path = image_path
        self.gt         = gt
        self.pred       = pred
        self.cer        = editdistance.eval(list(pred), list(gt)) / max(len(gt), 1)
        self.correct    = (pred == gt)


# ═══════════════════════════════════════════════════════════════════
# Virama / compound-character breakdown
# ═══════════════════════════════════════════════════════════════════

def breakdown_by_virama(
    predictions: List[str],
    ground_truths: List[str],
) -> Dict[str, dict]:
    """
    Split results into two groups:
      - "compound" : words whose GT contains Virama (్)
      - "simple"   : words without Virama

    Returns a dict with keys 'compound' and 'simple', each containing
    {'count', 'cer', 'wer'}.
    """
    groups: Dict[str, Tuple[List, List]] = {
        "compound": ([], []),
        "simple":   ([], []),
    }

    for pred, gt in zip(predictions, ground_truths):
        key = "compound" if VIRAMA in gt else "simple"
        groups[key][0].append(pred)
        groups[key][1].append(gt)

    results = {}
    for key, (preds, gts) in groups.items():
        if preds:
            cer, wer = compute_cer_wer(preds, gts)
            results[key] = {"count": len(preds), "cer": cer, "wer": wer}
        else:
            results[key] = {"count": 0, "cer": 0.0, "wer": 0.0}

    return results


# ═══════════════════════════════════════════════════════════════════
# Error analysis
# ═══════════════════════════════════════════════════════════════════

def print_error_examples(
    predictions: List[str],
    ground_truths: List[str],
    n: int = 20,
    sort_by_cer: bool = True,
):
    """Print the N highest-error examples."""
    errors = []
    for pred, gt in zip(predictions, ground_truths):
        if pred != gt:
            cer = editdistance.eval(list(pred), list(gt)) / max(len(gt), 1)
            errors.append((cer, gt, pred))

    if sort_by_cer:
        errors.sort(reverse=True)

    print(f"\n── Top-{min(n, len(errors))} error examples ──────────────────")
    for i, (cer, gt, pred) in enumerate(errors[:n]):
        print(f"  [{i+1:2d}] CER={cer:.3f}  GT='{gt}'  PRED='{pred}'")


def character_confusion_matrix(
    predictions: List[str],
    ground_truths: List[str],
    top_n: int = 20,
) -> Counter:
    """
    Approximate character confusion: align GT and pred at character level
    using the edit-distance traceback (simplified: just count substitutions
    from the naive alignment).

    Returns a Counter of (gt_char, pred_char) pairs, sorted by frequency.
    """
    confusion: Counter = Counter()

    for pred, gt in zip(predictions, ground_truths):
        # Simple prefix alignment (not traceback, but fast and good enough
        # for identifying common confusions)
        for g, p in zip(gt, pred):
            if g != p:
                confusion[(g, p)] += 1

    print(f"\n── Top-{top_n} character confusions ─────────────────────────")
    print(f"  {'GT':>6}  →  {'PRED':>6}  {'COUNT':>8}")
    for (g, p), count in confusion.most_common(top_n):
        print(f"  '{g}'  →  '{p}'  {count:>8}")

    return confusion


def avg_pred_length(predictions: List[str]) -> float:
    if not predictions:
        return 0.0
    return sum(len(p) for p in predictions) / len(predictions)


# ═══════════════════════════════════════════════════════════════════
# Full model evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model_ctc(
    model,
    loader,
    vocab:   TeluguVocab,
    device:  torch.device,
) -> Dict:
    """Run CTC model on a loader, return dict of metrics + lists."""
    model.eval()
    all_preds, all_gts = [], []
    inference_time = 0.0
    total_samples = 0

    for images, labels, label_lens, image_widths in loader:
        images       = images.to(device)
        labels       = labels.to(device)
        label_lens   = label_lens.to(device)
        image_widths = image_widths.to(device)

        t0 = time.time()
        pred_ids = model.greedy_decode(images, input_widths=image_widths)
        inference_time += time.time() - t0
        total_samples += images.size(0)

        for i, (pred, llen) in enumerate(zip(pred_ids, label_lens.tolist())):
            gt_ids = labels[i, :llen].tolist()
            pred_str = unicodedata.normalize("NFC", vocab.decode(pred))
            gt_str   = unicodedata.normalize("NFC", vocab.decode(gt_ids))
            all_preds.append(pred_str)
            all_gts.append(gt_str)

    cer, wer = compute_cer_wer(all_preds, all_gts)
    virama_bd = breakdown_by_virama(all_preds, all_gts)
    mean_cer, lower_ci, upper_ci = compute_bootstrap_ci(all_preds, all_gts)

    return {
        "cer": cer, "wer": wer,
        "cer_ci": (lower_ci, upper_ci),
        "inference_time_ms_per_sample": (inference_time / max(total_samples, 1)) * 1000,
        "predictions": all_preds,
        "ground_truths": all_gts,
        "virama_breakdown": virama_bd,
        "avg_pred_len": avg_pred_length(all_preds),
    }


@torch.no_grad()
def evaluate_model_ar(
    model,
    loader,
    vocab:         TeluguVocab,
    device:        torch.device,
    use_beam:      bool  = False,
    beam_size:     int   = 5,
    constrain:     bool  = True,
    constrain_penalty: float = None,
) -> Dict:
    """Run AR model on a loader, return dict of metrics + lists."""
    model.eval()
    all_preds, all_gts = [], []
    inference_time = 0.0
    total_samples = 0

    for images, labels, label_lens, image_widths in loader:
        images       = images.to(device)
        labels       = labels.to(device)
        label_lens   = label_lens.to(device)
        image_widths = image_widths.to(device)

        t0 = time.time()
        if use_beam:
            pred_ids = model.beam_decode(
                images, beam_size=beam_size,
                max_len=36, vocab=vocab if constrain else None,
                constrain=constrain,
                constrain_penalty=constrain_penalty,
                input_widths=image_widths,
                length_penalty=0.6
            )
        else:
            pred_ids = model.greedy_decode(
                images, max_len=36,
                vocab=vocab if constrain else None,
                constrain=constrain,
                constrain_penalty=constrain_penalty,
                input_widths=image_widths
            )
        inference_time += time.time() - t0
        total_samples += images.size(0)

        for i, (pred, lab, llen) in enumerate(zip(pred_ids, labels, label_lens.tolist())):
            # Skip SOS (index 0) and EOS when extracting GT
            gt_ids = lab[1:llen - 1].tolist()
            pred_str = unicodedata.normalize("NFC", vocab.decode(pred))
            gt_str   = unicodedata.normalize("NFC", vocab.decode(gt_ids))
            all_preds.append(pred_str)
            all_gts.append(gt_str)

    cer, wer = compute_cer_wer(all_preds, all_gts)
    virama_bd = breakdown_by_virama(all_preds, all_gts)
    mean_cer, lower_ci, upper_ci = compute_bootstrap_ci(all_preds, all_gts)

    return {
        "cer": cer, "wer": wer,
        "cer_ci": (lower_ci, upper_ci),
        "inference_time_ms_per_sample": (inference_time / max(total_samples, 1)) * 1000,
        "predictions": all_preds,
        "ground_truths": all_gts,
        "virama_breakdown": virama_bd,
        "avg_pred_len": avg_pred_length(all_preds),
    }


# ═══════════════════════════════════════════════════════════════════
# Ablation table printer
# ═══════════════════════════════════════════════════════════════════

def print_ablation_table(results: Dict[str, Dict]):
    """
    Print a formatted ablation table.

    results = {
        "CTC Baseline":           {"cer": 0.12, "wer": 0.45, ...},
        "AR (no constraint)":     {"cer": 0.09, ...},
        "AR + Telugu constraint": {"cer": 0.07, ...},
        ...
    }
    """
    print("\n" + "═" * 100)
    print(f"  {'Model':<30} {'CER':>8} {'WER':>8} "
          f"{'95% CI':>16} {'Comp CER':>10} {'Speed':>10}")
    print("─" * 100)
    for name, res in results.items():
        vb      = res.get("virama_breakdown", {})
        comp    = vb.get("compound", {})
        simple  = vb.get("simple", {})
        ci      = res.get("cer_ci", (0.0, 0.0))
        speed   = res.get("inference_time_ms_per_sample", 0.0)
        print(
            f"  {name:<30} "
            f"{res['cer'] * 100:>7.2f}% "
            f"{res['wer'] * 100:>7.2f}% "
            f"[{ci[0]*100:>5.2f}, {ci[1]*100:>5.2f}] "
            f"{comp.get('cer', 0.0) * 100:>9.2f}% "
            f"{speed:>7.1f} ms"
        )
    print("═" * 100 + "\n")


# ═══════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",  choices=["ctc", "ar"], required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--config",      required=True)
    parser.add_argument("--split",       default="test")
    parser.add_argument("--beam",        action="store_true")
    parser.add_argument(
        "--constrain-penalty",
        type=float,
        default=None,
        help="Soft penalty for invalid bigrams instead of -inf",
    )
    parser.add_argument("--beam_size",   type=int, default=5)
    parser.add_argument("--no_constrain", action="store_true")
    args = parser.parse_args()

    cfg   = yaml.safe_load(open(args.config))
    dcfg  = cfg["data"]
    mcfg  = cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.vocab import TeluguVocab
    vocab  = TeluguVocab.load(cfg["training"]["vocab_path"])

    ann_key  = f"{args.split}_annotation"
    root_key = f"{args.split}_image_root"
    from src.dataset import build_dataloader
    loader = build_dataloader(
        dcfg.get(ann_key,  f"data/raw/{args.split}/labels.txt"),
        dcfg.get(root_key, f"data/raw/{args.split}"),
        vocab,
        split       = args.split,
        batch_size  = 64,
        num_workers = dcfg.get("num_workers", 4),
        max_label_len = dcfg["max_label_len"],
        add_sos_eos = (args.model_type == "ar"),
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if args.model_type == "ctc":
        from src.models.ctc_model import CTCModel
        model = CTCModel(vocab_size=len(vocab), **{k: mcfg[k] for k in
            ["d_model", "lstm_hidden", "lstm_layers", "dropout", "pretrained"]})
        model.load_state_dict(ckpt["model"])
        model.to(device)
        res = evaluate_model_ctc(model, loader, vocab, device)
    else:
        from src.models.ar_model import ARModel
        model = ARModel(vocab_size=len(vocab), sos_id=vocab.sos_id, eos_id=vocab.eos_id,
                        num_encoder_layers=mcfg.get("num_encoder_layers", 2),
                        high_res_temporal=mcfg.get("high_res_temporal", False),
                        ctc_weight=mcfg.get("ctc_weight", 0.3),
                        **{k: mcfg[k] for k in
                           ["d_model", "nhead", "num_decoder_layers", "dim_feedforward",
                            "dropout", "max_label_len", "label_smoothing", "pretrained"]})
        model.load_state_dict(ckpt["model"])
        model.to(device)
        res = evaluate_model_ar(
            model, loader, vocab, device,
            use_beam  = args.beam,
            beam_size = args.beam_size,
            constrain = not args.no_constrain,
            constrain_penalty = args.constrain_penalty,
        )

    print(f"\n── Results on [{args.split}] ──────────────────────────────")
    print(f"  CER           : {res['cer']:.4f}")
    print(f"  WER           : {res['wer']:.4f}")
    if "cer_ci" in res:
        print(f"  95% CI (CER)  : [{res['cer_ci'][0]:.4f}, {res['cer_ci'][1]:.4f}]")
    if "inference_time_ms_per_sample" in res:
        print(f"  Speed         : {res['inference_time_ms_per_sample']:.1f} ms/sample")
    print(f"  Avg pred len  : {res['avg_pred_len']:.1f}")
    vb = res["virama_breakdown"]
    print(f"  Compound CER  : {vb['compound']['cer']:.4f}  (n={vb['compound']['count']})")
    print(f"  Simple CER    : {vb['simple']['cer']:.4f}   (n={vb['simple']['count']})")

    print_error_examples(res["predictions"], res["ground_truths"], n=20)
    character_confusion_matrix(res["predictions"], res["ground_truths"], top_n=20)

```

### src/models/__init__.py
```python

```

### src/models/cnn_encoder.py
```python
"""
src/models/cnn_encoder.py

Shared visual encoder used by both the CTC and AR models.

Architecture:
    Input [B, 1, 64, 512]
      │
      ▼
    Conv1x1(1→3)              expand grayscale for ResNet
      │
      ▼
    ResNet-18 (pretrained, backbone only — no avgpool, no fc)
    Maxpool stride:
        If high_res_temporal: stride=(2, 1)
        Else: stride=(2, 2)
    Layer3 and Layer4: stride=(1, 1) to preserve width resolution.
      → feature map [B, 512, 1, 64]   (H collapsed, W preserved)
      │
      ▼
    Squeeze height dim         [B, 512, 64]
      │
      ▼
    Conv1D(512→d_model)       channel reduction over sequence
      │
      ▼
    Positional encoding (sinusoidal)
      │
      ▼
    Encoder memory [B, 64, d_model]
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for 1-D sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                  # [1, L, D]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, D]"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Stride-patched ResNet-18 builder
# ---------------------------------------------------------------------------

def _patch_resnet18_strides(model: nn.Module, high_res_temporal: bool = False) -> nn.Module:
    """
    Modify ResNet-18 to preserve width resolution for CTC/AR.
    By default (high_res_temporal=False), stride downsamples width by 8.
    If high_res_temporal=True, maxpool stride becomes (2, 1), downsampling width by 4.
    """
    # Patch maxpool
    if high_res_temporal:
        model.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1)
    else:
        model.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)

    # Remove width-stride from layer3 and layer4: compress height but keep width
    for layer_name in ("layer3", "layer4"):
        layer = getattr(model, layer_name)
        first_block = layer[0]

        # Reconstruct conv1 with stride=(2,1) — setting .stride is a no-op on Conv2d!
        if hasattr(first_block, "conv1"):
            old = first_block.conv1
            new_conv = nn.Conv2d(
                old.in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=(2, 1),
                padding=old.padding, bias=(old.bias is not None),
            )
            new_conv.weight = old.weight
            if old.bias is not None:
                new_conv.bias = old.bias
            first_block.conv1 = new_conv

        # Reconstruct the downsample projection (skip connection) with stride=(2,1)
        if first_block.downsample is not None:
            old_ds = first_block.downsample[0]
            new_ds = nn.Conv2d(
                old_ds.in_channels, old_ds.out_channels,
                kernel_size=old_ds.kernel_size, stride=(2, 1),
                padding=old_ds.padding, bias=(old_ds.bias is not None),
            )
            new_ds.weight = old_ds.weight
            if old_ds.bias is not None:
                new_ds.bias = old_ds.bias
            first_block.downsample[0] = new_ds

    return model


# ---------------------------------------------------------------------------
# CNN Encoder
# ---------------------------------------------------------------------------

class ResNetEncoder(nn.Module):
    """
    ResNet-18 visual encoder producing a sequence of feature vectors.

    Parameters
    ----------
    d_model           : output feature dimension (default 256)
    pretrained        : load ImageNet weights (default True)
    dropout           : dropout on positional encoding
    high_res_temporal : use lower stride in maxpool to preserve more temporal resolution
    """

    def __init__(
        self,
        d_model: int     = 256,
        pretrained: bool = True,
        dropout: float   = 0.1,
        high_res_temporal: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.high_res_temporal = high_res_temporal
        self.width_downsample = 4 if high_res_temporal else 8

        # ---- ResNet-18 backbone with patched strides ----
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet  = tvm.resnet18(weights=weights)
        resnet  = _patch_resnet18_strides(resnet, high_res_temporal=high_res_temporal)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        # After patch: output is [B, 512, 1, 64] for input [B, 3, 64, 512]

        # ---- Channel & Height reduction: 512 → d_model via 2D conv ----
        # Takes [B, 512, 2, W'] and outputs [B, d_model, 1, W'] (for H=64)
        self.proj = nn.Sequential(
            nn.Conv2d(512, d_model, kernel_size=(2, 1), bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

        # ---- Positional encoding ----
        self.pos_enc = PositionalEncoding(d_model, max_len=512, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, 1, H, W]  grayscale image tensor in [-1, 1]

        Returns
        -------
        memory : [B, S, d_model]
                 S = W/8 = 64  for W=512, using patched strides.
                 (height is fully collapsed to 1 then squeezed)
        """
        # 1-ch → 3-ch (replicate grayscale across RGB channels for pretrained ResNet)
        x = x.repeat(1, 3, 1, 1)                  # [B, 3, H, W]

        # ResNet backbone
        feat = self.backbone(x)                   # [B, 512, 2, W'] (for H=64)

        # Spatial compression & channel reduction
        feat = self.proj(feat)                    # [B, d_model, H', W']

        # Collapse any remaining height dimension robustly
        feat = feat.mean(dim=2)                   # [B, d_model, W']

        feat = feat.permute(0, 2, 1)              # [B, W', d_model]

        # Positional encoding
        memory = self.pos_enc(feat)               # [B, S, d_model]
        return memory

    def get_output_len(self, input_width: int) -> int:
        """Compute sequence length S for a given input width W (useful for CTC)."""
        dummy = torch.zeros(1, 1, 64, input_width)
        with torch.no_grad():
            out = self.forward(dummy)
        return out.size(1)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ResNetEncoder shape test ===")
    enc = ResNetEncoder(d_model=256, pretrained=False)
    dummy = torch.randn(2, 1, 64, 512)
    out   = enc(dummy)
    print(f"Input  : {list(dummy.shape)}")   # [2, 1, 64, 512]
    print(f"Output : {list(out.shape)}")     # [2, 64, 256]  ← S=64, not 32
    seq_len = enc.get_output_len(512)
    print(f"Sequence length for W=512: {seq_len}")

```

### src/models/ctc_model.py
```python
"""
src/models/ctc_model.py

CTC baseline model.

Pipeline:
    Image [B, 1, 64, 512]
      → ResNetEncoder → memory [B, 64, 256]
      → 2-layer BiLSTM → [B, 64, 512]
      → Linear → logits [B, 64, vocab_size]
      → CTCLoss  (training)  /  CTC greedy decode  (inference)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from src.models.cnn_encoder import ResNetEncoder


class CTCModel(nn.Module):
    """
    CNN + BiLSTM + CTC handwritten text recognition model.

    Parameters
    ----------
    vocab_size   : number of output tokens (including blank).
                   CTC blank index is assumed to be 0 (PAD token).
    d_model      : encoder feature dimension.
    lstm_hidden  : hidden size of each LSTM direction.
    lstm_layers  : number of BiLSTM layers.
    dropout      : dropout applied inside LSTM.
    pretrained   : use ImageNet weights for ResNet-18.
    """

    def __init__(
        self,
        vocab_size:   int,
        d_model:      int   = 256,
        lstm_hidden:  int   = 256,
        lstm_layers:  int   = 2,
        dropout:      float = 0.2,
        pretrained:   bool  = True,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.blank_id    = 0     # PAD token doubles as CTC blank

        # ── Visual encoder ───────────────────────────────────────
        self.encoder = ResNetEncoder(
            d_model    = d_model,
            pretrained = pretrained,
            dropout    = dropout,
        )

        # ── BiLSTM sequence model ────────────────────────────────
        self.lstm = nn.LSTM(
            input_size    = d_model,
            hidden_size   = lstm_hidden,
            num_layers    = lstm_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if lstm_layers > 1 else 0.0,
        )

        # ── CTC projection ───────────────────────────────────────
        self.ctc_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, vocab_size),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.ctc_head[-1].weight)
        nn.init.zeros_(self.ctc_head[-1].bias)

    # ── Forward ─────────────────────────────────────────────────

    def forward(self, images: torch.Tensor, input_widths: torch.Tensor = None):
        """
        Parameters
        ----------
        images : [B, 1, H, W]
        input_widths : [B] optionally pass scaled widths to compute true sequence lengths

        Returns
        -------
        log_probs    : [T, B, vocab_size]  log-softmax CTC output
        output_lens  : [B]  sequence length T (same for all in batch unless masked)
        """
        # Visual features: [B, T, d_model]
        memory = self.encoder(images)              # [B, T, d_model]

        # BiLSTM
        lstm_out, _ = self.lstm(memory)            # [B, T, 2*lstm_hidden]

        # CTC logits → log-probs
        logits    = self.ctc_head(lstm_out)        # [B, T, vocab_size]
        log_probs = logits.log_softmax(dim=-1)     # [B, T, vocab_size]

        # CTCLoss expects [T, B, C]
        log_probs = log_probs.permute(1, 0, 2)    # [T, B, vocab_size]

        T = log_probs.size(0)
        if input_widths is not None:
            output_lens = (input_widths // self.encoder.width_downsample).clamp(max=T).to(images.device)
        else:
            output_lens = torch.full(
                (images.size(0),), T, dtype=torch.long, device=images.device
            )

        return log_probs, output_lens

    # ── Loss ─────────────────────────────────────────────────────

    def compute_loss(
        self,
        images:       torch.Tensor,   # [B, 1, H, W]
        labels:       torch.Tensor,   # [B, T_label]  padded label ids
        label_lens:   torch.Tensor,   # [B]           actual label lengths
        input_widths: torch.Tensor = None,  # [B]     image widths
    ) -> torch.Tensor:
        """Compute CTC loss for a batch."""
        log_probs, output_lens = self.forward(images, input_widths=input_widths)

        # CTCLoss needs labels as a 1-D concatenated tensor
        # and label_lengths as a 1-D tensor
        # CTCLoss accepts 2-D labels since PyTorch 1.9
        loss = nn.CTCLoss(blank=self.blank_id, reduction="mean", zero_infinity=True)(
            log_probs,    # [T, B, C]
            labels,       # [B, T_label]  — 2-D form accepted
            output_lens,  # [B]
            label_lens,   # [B]
        )
        return loss

    # ── Greedy decode ─────────────────────────────────────────────

    @torch.no_grad()
    def greedy_decode(self, images: torch.Tensor, input_widths: torch.Tensor = None) -> list[list[int]]:
        """
        CTC greedy decoding (argmax + collapse).

        Parameters
        ----------
        images : [B, 1, H, W]
        input_widths : [B] optional true image widths before padding

        Returns
        -------
        list of token id lists, one per batch item.
        """
        log_probs, output_lens = self.forward(images, input_widths=input_widths)   # [T, B, V], [B]
        pred_ids = log_probs.argmax(dim=-1)   # [T, B]
        pred_ids = pred_ids.permute(1, 0)     # [B, T]

        results = []
        blank = self.blank_id
        for idx, seq in enumerate(pred_ids):
            # Only look at valid (non-padded) time steps
            valid_len = output_lens[idx].item()
            collapsed = []
            prev = blank
            for tok in seq[:valid_len].tolist():
                if tok != blank and tok != prev:
                    collapsed.append(tok)
                prev = tok
            results.append(collapsed)
        return results

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Sanity check ──────────────────────────────────────────────────
if __name__ == "__main__":
    model = CTCModel(vocab_size=120, pretrained=False)
    print(f"Parameters: {model.count_params():,}")

    imgs   = torch.randn(4, 1, 64, 512)
    labels = torch.randint(1, 100, (4, 12))
    llens  = torch.tensor([10, 12, 8, 11])

    loss = model.compute_loss(imgs, labels, llens)
    print(f"CTC loss: {loss.item():.4f}")

    preds = model.greedy_decode(imgs)
    print(f"Greedy decode lengths: {[len(p) for p in preds]}")

```

### src/models/ar_model.py
```python
"""
src/models/ar_model.py

Autoregressive Transformer decoder model.

Pipeline:
    Image [B, 1, 64, 512]
      → ResNetEncoder  → encoder memory [B, 64, 256]
      → Transformer Decoder (cross-attention over memory)
         Input: shifted label tokens (teacher forcing during training)
      → Linear → logits [B, T, vocab_size]
      → Cross-Entropy loss  (training)
      → Greedy / beam decode  (inference)

The decoder is causal: each position can only attend to previous positions
(standard auto-regressive constraint via causal mask).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cnn_encoder import ResNetEncoder, PositionalEncoding


class ARModel(nn.Module):
    """
    CNN Encoder + Autoregressive Transformer Decoder for Telugu HTR.

    Parameters
    ----------
    vocab_size        : size of the character vocabulary.
    sos_id / eos_id   : special token ids.
    d_model           : feature dimension (encoder and decoder share this).
    nhead             : number of attention heads.
    num_decoder_layers: number of Transformer decoder layers.
    dim_feedforward   : inner dimension of the decoder FFN.
    dropout           : dropout throughout.
    max_label_len     : maximum label sequence length (for positional enc).
    label_smoothing   : label smoothing for cross-entropy loss.
    pretrained        : use ImageNet weights for ResNet-18.
    """

    def __init__(
        self,
        vocab_size:         int,
        sos_id:             int   = 1,
        eos_id:             int   = 2,
        d_model:            int   = 256,
        nhead:              int   = 8,
        num_encoder_layers: int   = 2,
        num_decoder_layers: int   = 4,
        dim_feedforward:    int   = 1024,
        dropout:            float = 0.1,
        max_label_len:      int   = 64,
        label_smoothing:    float = 0.1,
        pretrained:         bool  = True,
        high_res_temporal:  bool  = False,
        ctc_weight:         float = 0.3,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.sos_id      = sos_id
        self.eos_id      = eos_id
        self.d_model     = d_model
        self.max_label_len = max_label_len

        # ── Visual encoder ───────────────────────────────────────
        self.cnn_encoder = ResNetEncoder(
            d_model    = d_model,
            pretrained = pretrained,
            dropout    = dropout,
            high_res_temporal = high_res_temporal,
        )

        if num_encoder_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        else:
            self.transformer_encoder = None

        # ── Token embedding + positional encoding (decoder side) ─
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.embed_scale = math.sqrt(d_model)  # Vaswani et al. 2017, Eq. 3
        self.dec_pos_enc = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        # ── Transformer decoder ──────────────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,   # Pre-LN: more stable training
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers = num_decoder_layers,
        )

        # ── Output projection ────────────────────────────────────
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: output projection shares weights with token embedding
        self.output_proj.weight = self.token_embed.weight

        # ── Auxiliary CTC head for joint training ────────────────
        self.ctc_head = nn.Linear(d_model, vocab_size)
        self.ctc_weight = ctc_weight  # weighting factor for CTC loss (0 = pure CE)

        # ── Loss Function ─────────────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss(
            ignore_index    = 0,    # PAD
            label_smoothing = label_smoothing,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        # output_proj weights are tied to token_embed, so no separate init needed
        # output_proj has no bias (bias=False)

    # ── Causal mask ──────────────────────────────────────────────

    def _causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask for autoregressive decoding. [sz, sz]"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask    # True = blocked

    # ── Padding mask ─────────────────────────────────────────────

    def _pad_mask(self, seq: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        """Key padding mask: True where token == pad_id. [B, T]"""
        return seq == pad_id

    def _encoder_padding_mask(self, input_widths: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Create memory_key_padding_mask. True = padded position to ignore. [B, S]"""
        encoder_lens = (input_widths // self.cnn_encoder.width_downsample).clamp(max=seq_len)
        positions = torch.arange(seq_len, device=input_widths.device).unsqueeze(0)  # [1, S]
        mask = positions >= encoder_lens.unsqueeze(1)  # [B, S]
        return mask

    # ── Forward (teacher forcing) ────────────────────────────────

    def forward(
        self,
        images:  torch.Tensor,   # [B, 1, H, W]
        tgt_ids: torch.Tensor,   # [B, T]  shifted input  (SOS + label[:-1])
        input_widths: torch.Tensor = None, # [B] optional image widths
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass.

        Returns
        -------
        logits : [B, T, vocab_size]
        """
        B, T = tgt_ids.shape
        device = images.device

        # Encoder memory
        memory = self.cnn_encoder(images)              # [B, S, d_model]

        memory_pad_mask = None
        if input_widths is not None:
            memory_pad_mask = self._encoder_padding_mask(input_widths, memory.size(1))

        if self.transformer_encoder is not None:
            memory = self.transformer_encoder(
                src=memory, 
                src_key_padding_mask=memory_pad_mask
            )

        # Decoder input embeddings + positional encoding
        tgt_emb = self.token_embed(tgt_ids) * self.embed_scale  # [B, T, d_model]
        tgt_emb = self.dec_pos_enc(tgt_emb)

        # Causal mask
        causal_mask = self._causal_mask(T, device)

        # Key padding mask for decoder input
        tgt_key_pad = self._pad_mask(tgt_ids)      # [B, T]

        # Transformer decoder
        dec_out = self.decoder(
            tgt             = tgt_emb,
            memory          = memory,
            tgt_mask        = causal_mask,
            tgt_key_padding_mask = tgt_key_pad,
            memory_key_padding_mask = memory_pad_mask,
        )                                          # [B, T, d_model]

        logits = self.output_proj(dec_out)         # [B, T, vocab_size]
        return logits

    # ── Loss ─────────────────────────────────────────────────────

    def compute_loss(
        self,
        images:     torch.Tensor,   # [B, 1, H, W]
        labels:     torch.Tensor,   # [B, T+2]  SOS + label + EOS (padded)
        label_lens: torch.Tensor,   # [B]  lengths including SOS and EOS
        input_widths: torch.Tensor = None, # [B]
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with teacher forcing.

        labels should be: [SOS, c1, c2, ..., cn, EOS, PAD, PAD, ...]
        Decoder input  : labels[:, :-1]  → [SOS, c1, ..., cn]
        Target         : labels[:, 1:]   → [c1, ..., cn, EOS]
        """
        decoder_input = labels[:, :-1]    # [B, T]
        target        = labels[:, 1:]     # [B, T]

        # 1. Run Encoder
        memory = self.cnn_encoder(images)
        memory_pad_mask = None
        if input_widths is not None:
            memory_pad_mask = self._encoder_padding_mask(input_widths, memory.size(1))
        if self.transformer_encoder is not None:
            memory = self.transformer_encoder(src=memory, src_key_padding_mask=memory_pad_mask)

        # 2. Run Decoder for AR CE Loss
        tgt_emb = self.token_embed(decoder_input) * self.embed_scale
        tgt_emb = self.dec_pos_enc(tgt_emb)
        causal_mask = self._causal_mask(decoder_input.size(1), images.device)
        tgt_key_pad = self._pad_mask(decoder_input)
        dec_out = self.decoder(
            tgt=tgt_emb, memory=memory, tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_pad, memory_key_padding_mask=memory_pad_mask
        )
        logits = self.output_proj(dec_out)
        ce_loss = self.criterion(logits.reshape(-1, self.vocab_size), target.reshape(-1))

        # 3. Auxiliary CTC Loss on Encoder Memory (skip if weight == 0)
        if self.ctc_weight > 0:
            ctc_logits = self.ctc_head(memory)
            log_probs = ctc_logits.log_softmax(dim=-1).permute(1, 0, 2)  # [T, B, C]
            
            batch_size = labels.size(0)
            ctc_targets = []
            ctc_label_lens = []
            for i in range(batch_size):
                llen = label_lens[i].item()
                # extract actual tokens without SOS (pos 0) and EOS (pos llen-1)
                valid_tokens = labels[i, 1:llen-1]
                ctc_targets.append(valid_tokens)
                ctc_label_lens.append(len(valid_tokens))
                
            max_len = max(ctc_label_lens) if ctc_label_lens else 0
            ctc_target_tensor = torch.zeros((batch_size, max_len), dtype=torch.long, device=labels.device)
            for i in range(batch_size):
                ctc_target_tensor[i, :ctc_label_lens[i]] = ctc_targets[i]
                
            T_ctc = log_probs.size(0)
            if input_widths is not None:
                output_lens = (input_widths // self.cnn_encoder.width_downsample).clamp(max=T_ctc).to(images.device)
            else:
                output_lens = torch.full((batch_size,), T_ctc, dtype=torch.long, device=images.device)
                
            ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
            ctc_loss = ctc_loss_fn(
                log_probs, ctc_target_tensor, output_lens, 
                torch.tensor(ctc_label_lens, dtype=torch.long, device=images.device)
            )

            # Combine losses
            total_loss = (1.0 - self.ctc_weight) * ce_loss + self.ctc_weight * ctc_loss
            return total_loss, ce_loss.detach(), ctc_loss.detach()
        else:
            # Pure CE — no auxiliary CTC
            return ce_loss, ce_loss.detach(), torch.tensor(0.0, device=images.device)

    # ── Greedy decode (inference) ─────────────────────────────────

    @torch.no_grad()
    def greedy_decode(
        self,
        images:    torch.Tensor,              # [B, 1, H, W]
        max_len:   int           = 32,
        vocab     = None,                     # TeluguVocab — for constrained decoding
        constrain: bool          = False,     # apply validity matrix
        constrain_penalty: float = None,      # soft penalty for invalid transitions
        input_widths: torch.Tensor = None,    # [B]
    ) -> list[list[int]]:
        """
        Autoregressive greedy decoding.

        If constrain=True and vocab is provided, logits for illegal
        next tokens are set to -inf before argmax.

        Returns
        -------
        list of token id lists (without SOS, stopped at EOS).
        """
        B      = images.size(0)
        device = images.device

        memory = self.cnn_encoder(images)             # [B, S, d_model]
        
        memory_pad_mask = None
        if input_widths is not None:
            memory_pad_mask = self._encoder_padding_mask(input_widths, memory.size(1))

        if self.transformer_encoder is not None:
            memory = self.transformer_encoder(
                src=memory, 
                src_key_padding_mask=memory_pad_mask
            )

        # Start with SOS token
        generated = torch.full((B, 1), self.sos_id, dtype=torch.long, device=device)
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_len):
            tgt_emb   = self.token_embed(generated) * self.embed_scale  # [B, t, d_model]
            tgt_emb   = self.dec_pos_enc(tgt_emb)
            causal_mk = self._causal_mask(generated.size(1), device)

            dec_out = self.decoder(
                tgt    = tgt_emb,
                memory = memory,
                tgt_mask = causal_mk,
                memory_key_padding_mask = memory_pad_mask,
            )                                                # [B, t, d_model]

            logits = self.output_proj(dec_out[:, -1, :])    # [B, V]

            # ── Telugu constraint ──────────────────────────────
            if constrain and vocab is not None:
                prev_tokens = generated[:, -1]               # [B]
                for b in range(B):
                    if finished[b]:
                        continue
                    prev_id = prev_tokens[b].item()
                    valid_mask = vocab.get_valid_next_tensor(prev_id, device=device)
                    if constrain_penalty is not None:
                        logits[b] = torch.where(valid_mask, logits[b], logits[b] - constrain_penalty)
                    else:
                        logits[b][~valid_mask] = float("-inf")

            next_tok = logits.argmax(dim=-1, keepdim=True)  # [B, 1]

            # Mark finished sequences
            finished = finished | (next_tok.squeeze(1) == self.eos_id)

            generated = torch.cat([generated, next_tok], dim=1)

            if finished.all():
                break

        # Strip SOS, stop at EOS
        results = []
        for row in generated:
            ids = row.tolist()[1:]   # remove SOS
            out = []
            for tok in ids:
                if tok == self.eos_id:
                    break
                out.append(tok)
            results.append(out)

        return results

    # ── Beam search decode ────────────────────────────────────────

    @torch.no_grad()
    def beam_decode(
        self,
        images:    torch.Tensor,
        beam_size: int  = 5,
        max_len:   int  = 32,
        vocab      = None,
        constrain: bool = False,
        constrain_penalty: float = None,
        input_widths: torch.Tensor = None,
        length_penalty: float = 0.6,
    ) -> list[list[int]]:
        """
        Beam search decoding — processes one image at a time for simplicity.

        Returns list of best token id sequences (without SOS/EOS).
        """
        B      = images.size(0)
        device = images.device
        results = []

        for b in range(B):
            img    = images[b:b+1]                           # [1, 1, H, W]
            memory = self.cnn_encoder(img)                       # [1, S, d_model]
            
            memory_pad_mask = None
            if input_widths is not None:
                memory_pad_mask = self._encoder_padding_mask(input_widths[b:b+1], memory.size(1))
            
            if self.transformer_encoder is not None:
                memory = self.transformer_encoder(src=memory, src_key_padding_mask=memory_pad_mask)

            if memory_pad_mask is not None:
                memory_pad_mask = memory_pad_mask.expand(beam_size, -1)

            memory = memory.expand(beam_size, -1, -1)       # [K, S, d_model]

            # beams: list of (score, token_ids)
            beams  = [(0.0, [self.sos_id])]
            done   = []

            for step in range(max_len):
                all_candidates = []

                # Expand memory to current beam count
                cur_k  = len(beams)
                mem_k  = memory[:cur_k]

                # Build current token sequences
                seqs = torch.tensor(
                    [b_tok for _, b_tok in beams],
                    dtype=torch.long, device=device
                )                                            # [cur_k, t]

                tgt_emb   = self.token_embed(seqs) * self.embed_scale
                tgt_emb   = self.dec_pos_enc(tgt_emb)
                causal_mk = self._causal_mask(seqs.size(1), device)

                dec_out = self.decoder(
                    tgt    = tgt_emb,
                    memory = mem_k,
                    tgt_mask = causal_mk,
                    memory_key_padding_mask = memory_pad_mask[:cur_k] if memory_pad_mask is not None else None,
                )                                            # [cur_k, t, d_model]

                logits = self.output_proj(dec_out[:, -1, :])  # [cur_k, V]

                # Apply Telugu constraints per beam
                if constrain and vocab is not None:
                    for k_i, (_, tok_seq) in enumerate(beams):
                        prev_id = tok_seq[-1]
                        valid_mask = vocab.get_valid_next_tensor(prev_id, device=device)
                        if constrain_penalty is not None:
                            logits[k_i] = torch.where(valid_mask, logits[k_i], logits[k_i] - constrain_penalty)
                        else:
                            logits[k_i][~valid_mask] = float("-inf")

                log_probs = F.log_softmax(logits, dim=-1)    # [cur_k, V]

                for k_i, (score, tok_seq) in enumerate(beams):
                    top_probs, top_ids = log_probs[k_i].topk(beam_size)
                    for prob, tok in zip(top_probs.tolist(), top_ids.tolist()):
                        if tok == self.eos_id:
                            done.append((score + prob, tok_seq[1:]))  # strip SOS
                        else:
                            all_candidates.append((score + prob, tok_seq + [tok]))

                if not all_candidates:
                    break

                # Keep top-K
                all_candidates.sort(key=lambda x: x[0] / max(len(x[1]), 1) ** length_penalty, reverse=True)
                beams = all_candidates[:beam_size]

            if not done:
                done = [(sc, seq[1:]) for sc, seq in beams]

            best = max(done, key=lambda x: x[0] / max(len(x[1]), 1) ** length_penalty)[1]
            results.append(best)

        return results

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Sanity check ──────────────────────────────────────────────────
if __name__ == "__main__":
    model = ARModel(vocab_size=120, pretrained=False)
    print(f"Parameters: {model.count_params():,}")

    imgs   = torch.randn(2, 1, 64, 512)
    # labels: SOS + 10 chars + EOS, padded to 14
    labels = torch.zeros(2, 14, dtype=torch.long)
    labels[:, 0] = 1   # SOS
    labels[:, 1:11] = torch.randint(4, 100, (2, 10))
    labels[:, 11] = 2  # EOS
    llens  = torch.tensor([12, 12])

    loss, ce_loss, ctc_loss = model.compute_loss(imgs, labels, llens)
    print(f"AR Total loss: {loss.item():.4f}, CE loss: {ce_loss.item():.4f}, CTC loss: {ctc_loss.item():.4f}")

    preds = model.greedy_decode(imgs, max_len=15)
    print(f"Greedy decode lengths: {[len(p) for p in preds]}")

```

### src/decoding/__init__.py
```python

```

### src/decoding/telugu_mask.py
```python
"""
src/decoding/telugu_mask.py

Utilities for applying the data-driven Telugu script constraint mask
during autoregressive decoding.

The core object is the TeluguMask, a thin wrapper around the validity
matrix stored in TeluguVocab. It provides:
  - apply_to_logits()  — zero-out illegal next tokens before argmax / beam
  - batch_apply()      — vectorised apply for whole batches
  - stats_report()     — how often the mask actually fires during a decode run
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class MaskStats:
    """Accumulated statistics about how often the mask fires."""
    total_steps:   int = 0
    masked_steps:  int = 0      # steps where ≥1 token was blocked
    tokens_masked: int = 0      # total number of tokens blocked across all steps

    def update(self, blocked_count: int):
        self.total_steps   += 1
        if blocked_count > 0:
            self.masked_steps  += 1
            self.tokens_masked += blocked_count

    def report(self) -> str:
        if self.total_steps == 0:
            return "MaskStats: no steps recorded."
        fire_rate = 100.0 * self.masked_steps / self.total_steps
        return (
            f"MaskStats | steps={self.total_steps} | "
            f"mask_fired={self.masked_steps} ({fire_rate:.1f}%) | "
            f"tokens_blocked={self.tokens_masked}"
        )


class TeluguMask:
    """
    Applies the transition validity matrix from a TeluguVocab to decoder
    logits during autoregressive inference.

    Usage
    ─────
        mask_module = TeluguMask(vocab)

        # Inside decode loop:
        logits = model.decode_step(memory, prev_tokens)
        logits = mask_module.apply_to_logits(logits, prev_token_ids)
        next_tokens = logits.argmax(-1)
    """

    NEG_INF = float("-inf")

    def __init__(self, vocab, collect_stats: bool = True):
        """
        Parameters
        ----------
        vocab        : TeluguVocab instance (must have _valid_next built).
        collect_stats: if True, accumulate firing statistics for analysis.
        """
        self.vocab = vocab
        self.V     = len(vocab)
        self.stats = MaskStats() if collect_stats else None

        # Pre-cache the validity matrix as a GPU-friendly boolean tensor
        # Shape: [V, V]  valid_tensor[i, j] = True if j is valid after i
        valid_list = vocab._valid_next
        self._valid_tensor: Optional[torch.Tensor] = torch.tensor(
            valid_list, dtype=torch.bool
        )   # will be moved to device on first use

    def _ensure_device(self, device: torch.device):
        if self._valid_tensor.device != device:
            self._valid_tensor = self._valid_tensor.to(device)

    # ── Single-step apply ─────────────────────────────────────────

    def apply_to_logits(
        self,
        logits:          torch.Tensor,   # [B, V]
        prev_token_ids:  torch.Tensor,   # [B]   long
    ) -> torch.Tensor:
        """
        Mask illegal next-token logits to -inf.

        Parameters
        ----------
        logits         : raw decoder logits, shape [B, V].
        prev_token_ids : the previously predicted token for each item, [B].

        Returns
        -------
        Masked logits [B, V].  The original tensor is modified in-place
        for efficiency and also returned.
        """
        device = logits.device
        self._ensure_device(device)

        B = logits.size(0)
        for b in range(B):
            prev = prev_token_ids[b].item()
            valid_row = self._valid_tensor[prev]         # [V]  bool
            blocked   = ~valid_row                       # [V]  True = block
            n_blocked = blocked.sum().item()

            logits[b][blocked] = self.NEG_INF

            if self.stats is not None:
                self.stats.update(n_blocked)

        return logits

    # ── Vectorised batch apply ────────────────────────────────────

    def apply_to_logits_vectorised(
        self,
        logits:         torch.Tensor,   # [B, V]
        prev_token_ids: torch.Tensor,   # [B]   long
    ) -> torch.Tensor:
        """
        Fully vectorised version using advanced indexing.
        Faster for large batches / large vocabularies.
        """
        device = logits.device
        self._ensure_device(device)

        # Gather validity rows for all prev tokens in batch: [B, V]
        prev   = prev_token_ids.clamp(0, self.V - 1)    # safety
        valid  = self._valid_tensor[prev]                # [B, V]
        logits = logits.masked_fill(~valid, self.NEG_INF)

        if self.stats is not None:
            n_blocked = (~valid).sum().item()
            self.stats.update(n_blocked)

        return logits

    # ── Beam search version ───────────────────────────────────────

    def apply_to_beam_logits(
        self,
        logits:         torch.Tensor,   # [K, V]  K = beam_size
        prev_token_ids: List[int],
    ) -> torch.Tensor:
        """
        Apply mask to K beams, each with its own previous token.
        """
        device = logits.device
        self._ensure_device(device)

        for k, prev in enumerate(prev_token_ids):
            prev = max(0, min(prev, self.V - 1))
            valid_row = self._valid_tensor[prev]
            logits[k][~valid_row] = self.NEG_INF

        return logits

    # ── Statistics ───────────────────────────────────────────────

    def reset_stats(self):
        if self.stats is not None:
            self.stats = MaskStats()

    def print_stats(self):
        if self.stats is not None:
            print(f"[TeluguMask] {self.stats.report()}")
        else:
            print("[TeluguMask] Stats collection disabled.")

```

## 6. Evaluation Scripts

### scripts/run_full_evaluation.py
```python
"""
scripts/run_full_evaluation.py

Master script to generate ALL results and figures for the IEEE paper.

Usage (run on your GPU server):
    python scripts/run_full_evaluation.py

This script will:
  1. Evaluate AR v2 model: greedy (unconstrained), greedy (constrained), beam search
  2. Evaluate CTC baseline model (if checkpoint exists)
  3. Generate training curves from JSON log
  4. Generate confusion matrix heatmap
  5. Generate error examples figure
  6. Generate ablation comparison bar chart
  7. Print a formatted results table for the paper

All figures are saved to: results/paper_figures/
"""

import os
import sys
import json
import argparse
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

RESULTS_DIR = "results/paper_figures"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# Part 1: Generate Training Curves from JSON log
# ═══════════════════════════════════════════════════════════════════

def generate_training_curves():
    """Plot training loss and validation CER over epochs for AR v2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Try to find the JSON training log
    log_dir = "logs/ar_v2"
    json_files = [f for f in os.listdir(log_dir) if f.endswith(".json")] if os.path.isdir(log_dir) else []

    if not json_files:
        print("[SKIP] No JSON log found in logs/ar_v2/. Trying to parse text log...")
        generate_training_curves_from_text()
        return

    json_path = os.path.join(log_dir, sorted(json_files)[-1])  # latest
    print(f"[PLOT] Reading training log: {json_path}")

    with open(json_path) as f:
        log_data = json.load(f)

    epochs = []
    train_losses = []
    val_cers = []

    if isinstance(log_data, list):
        for entry in log_data:
            epochs.append(entry.get("epoch", len(epochs) + 1))
            train_losses.append(entry.get("train_loss", entry.get("avg_loss", 0)))
            val_cers.append(entry.get("val_cer", 0))
    elif isinstance(log_data, dict) and "epochs" in log_data:
        for entry in log_data["epochs"]:
            epochs.append(entry.get("epoch", len(epochs) + 1))
            train_losses.append(entry.get("train_loss", entry.get("avg_loss", 0)))
            val_cers.append(entry.get("val_cer", 0))
    else:
        print(f"[WARN] Unexpected JSON structure. Keys: {list(log_data.keys()) if isinstance(log_data, dict) else 'list'}")
        generate_training_curves_from_text()
        return

    _plot_curves(epochs, train_losses, val_cers)


def generate_training_curves_from_text():
    """Fallback: parse text log file for epoch summaries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import re

    log_dir = "logs/ar_v2"
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")] if os.path.isdir(log_dir) else []
    if not log_files:
        # Try the tee'd log
        if os.path.exists("logs/ar_v2_training.log"):
            log_path = "logs/ar_v2_training.log"
        else:
            print("[SKIP] No training log found. Skipping training curves.")
            return
    else:
        log_path = os.path.join(log_dir, sorted(log_files)[-1])

    print(f"[PLOT] Parsing text log: {log_path}")

    epochs = []
    train_losses = []
    val_cers = []

    with open(log_path) as f:
        for line in f:
            # Match: [Epoch N] avg_loss=X.XXXX  time=Xs
            m_loss = re.match(r'\[Epoch (\d+)\] avg_loss=([0-9.]+)', line.strip())
            if m_loss:
                ep = int(m_loss.group(1))
                loss = float(m_loss.group(2))
                epochs.append(ep)
                train_losses.append(loss)

            # Match: [Val epoch N (unconstrained)] ... CER=X.XXXX
            m_cer = re.search(r'\[Val epoch \d+ \(unconstrained\)\].*CER=([0-9.]+)', line.strip())
            if m_cer:
                val_cers.append(float(m_cer.group(1)))

    # Align lengths
    min_len = min(len(epochs), len(train_losses), len(val_cers))
    epochs = epochs[:min_len]
    train_losses = train_losses[:min_len]
    val_cers = val_cers[:min_len]

    if min_len == 0:
        print("[SKIP] Could not parse any epoch data from log.")
        return

    _plot_curves(epochs, train_losses, val_cers)


def _plot_curves(epochs, train_losses, val_cers):
    """Create the dual-axis training curves plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_loss = "#2196F3"
    color_cer = "#FF5722"

    # Training loss
    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("Training Loss", color=color_loss, fontsize=13)
    ax1.plot(epochs, train_losses, color=color_loss, linewidth=2, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_ylim(bottom=0)

    # Validation CER
    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation CER (%)", color=color_cer, fontsize=13)
    val_cers_pct = [c * 100 for c in val_cers]
    ax2.plot(epochs, val_cers_pct, color=color_cer, linewidth=2, linestyle="--", label="Val CER")
    ax2.tick_params(axis="y", labelcolor=color_cer)
    ax2.set_ylim(bottom=0)

    # Add best CER annotation
    best_idx = np.argmin(val_cers_pct)
    best_cer = val_cers_pct[best_idx]
    best_epoch = epochs[best_idx]
    ax2.annotate(
        f"Best: {best_cer:.2f}% (Epoch {best_epoch})",
        xy=(best_epoch, best_cer),
        xytext=(best_epoch + 5, best_cer + 3),
        fontsize=11, fontweight="bold", color=color_cer,
        arrowprops=dict(arrowstyle="->", color=color_cer, lw=1.5),
    )

    # Add CTC baseline reference line
    ax2.axhline(y=3.91, color="green", linestyle=":", linewidth=1.5, alpha=0.7)
    ax2.text(2, 4.2, "CTC Baseline (3.91%)", color="green", fontsize=10, alpha=0.8)

    fig.suptitle("AR Transformer v2 — Training Progress", fontsize=15, fontweight="bold")
    fig.tight_layout()

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=11)

    path = os.path.join(RESULTS_DIR, "training_curves_ar_v2.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")

    # Also save PDF for IEEE
    path_pdf = os.path.join(RESULTS_DIR, "training_curves_ar_v2.pdf")
    fig2, ax1 = plt.subplots(figsize=(10, 6))
    color_loss = "#2196F3"
    color_cer = "#FF5722"
    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("Training Loss", color=color_loss, fontsize=13)
    ax1.plot(epochs, train_losses, color=color_loss, linewidth=2, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_ylim(bottom=0)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation CER (%)", color=color_cer, fontsize=13)
    val_cers_pct = [c * 100 for c in [v for v in ([v for v in val_cers])]]
    ax2.plot(epochs, val_cers_pct, color=color_cer, linewidth=2, linestyle="--", label="Val CER")
    ax2.tick_params(axis="y", labelcolor=color_cer)
    ax2.set_ylim(bottom=0)
    ax2.axhline(y=3.91, color="green", linestyle=":", linewidth=1.5, alpha=0.7)
    ax2.text(2, 4.2, "CTC Baseline (3.91%)", color="green", fontsize=10, alpha=0.8)
    best_idx = np.argmin(val_cers_pct)
    best_cer = val_cers_pct[best_idx]
    best_epoch = epochs[best_idx]
    ax2.annotate(
        f"Best: {best_cer:.2f}% (Epoch {best_epoch})",
        xy=(best_epoch, best_cer),
        xytext=(best_epoch + 5, best_cer + 3),
        fontsize=11, fontweight="bold", color=color_cer,
        arrowprops=dict(arrowstyle="->", color=color_cer, lw=1.5),
    )
    fig2.suptitle("AR Transformer v2 — Training Progress", fontsize=15, fontweight="bold")
    fig2.tight_layout()
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=11)
    fig2.savefig(path_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path_pdf}")


# ═══════════════════════════════════════════════════════════════════
# Part 2: Run Model Evaluations
# ═══════════════════════════════════════════════════════════════════

def run_evaluations():
    """Run all model evaluations and return results dict."""
    import torch
    import yaml
    from src.vocab import TeluguVocab
    from src.dataset import build_dataloader
    from src.evaluate import (
        evaluate_model_ar, evaluate_model_ctc,
        print_error_examples, character_confusion_matrix,
        print_ablation_table,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Device: {device}")

    all_results = {}

    # ─── AR v2 Model ─────────────────────────────────────────────
    ar_v2_ckpt = "checkpoints/ar_v2/best.pt"
    ar_v2_cfg_path = "configs/ar_v2_config.yaml"

    if os.path.exists(ar_v2_ckpt) and os.path.exists(ar_v2_cfg_path):
        print("\n" + "=" * 70)
        print("  Evaluating AR Transformer v2")
        print("=" * 70)

        cfg = yaml.safe_load(open(ar_v2_cfg_path))
        mcfg = cfg["model"]
        dcfg = cfg["data"]
        vocab = TeluguVocab.load(cfg["training"]["vocab_path"])

        from src.models.ar_model import ARModel
        model = ARModel(
            vocab_size=len(vocab),
            sos_id=vocab.sos_id,
            eos_id=vocab.eos_id,
            num_encoder_layers=mcfg.get("num_encoder_layers", 2),
            high_res_temporal=mcfg.get("high_res_temporal", False),
            ctc_weight=mcfg.get("ctc_weight", 0.3),
            d_model=mcfg["d_model"],
            nhead=mcfg["nhead"],
            num_decoder_layers=mcfg["num_decoder_layers"],
            dim_feedforward=mcfg["dim_feedforward"],
            dropout=mcfg["dropout"],
            max_label_len=mcfg["max_label_len"],
            label_smoothing=mcfg["label_smoothing"],
            pretrained=False,  # Don't download pretrained; we're loading weights
        )
        ckpt = torch.load(ar_v2_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(device)

        loader = build_dataloader(
            dcfg.get("test_annotation", "data/raw/test/labels.txt"),
            dcfg.get("test_image_root", "data/raw/test"),
            vocab, split="test", batch_size=64,
            num_workers=dcfg.get("num_workers", 4),
            max_label_len=dcfg["max_label_len"],
            add_sos_eos=True,
        )

        # ── Variant 1: Greedy, Unconstrained ──
        print("\n[1/4] AR v2 — Greedy (Unconstrained)...")
        res_greedy_unc = evaluate_model_ar(
            model, loader, vocab, device,
            use_beam=False, constrain=False,
        )
        all_results["AR v2 (greedy, unconstrained)"] = res_greedy_unc
        print(f"  CER: {res_greedy_unc['cer']*100:.2f}%  WER: {res_greedy_unc['wer']*100:.2f}%")

        # ── Variant 2: Greedy, Constrained ──
        print("\n[2/4] AR v2 — Greedy (Constrained)...")
        res_greedy_con = evaluate_model_ar(
            model, loader, vocab, device,
            use_beam=False, constrain=True, constrain_penalty=10.0,
        )
        all_results["AR v2 (greedy, constrained)"] = res_greedy_con
        print(f"  CER: {res_greedy_con['cer']*100:.2f}%  WER: {res_greedy_con['wer']*100:.2f}%")

        # ── Variant 3: Beam Search, Unconstrained ──
        print("\n[3/4] AR v2 — Beam Search (Unconstrained, beam=5)...")
        res_beam_unc = evaluate_model_ar(
            model, loader, vocab, device,
            use_beam=True, beam_size=5, constrain=False,
        )
        all_results["AR v2 (beam=5, unconstrained)"] = res_beam_unc
        print(f"  CER: {res_beam_unc['cer']*100:.2f}%  WER: {res_beam_unc['wer']*100:.2f}%")

        # ── Variant 4: Beam Search, Constrained ──
        print("\n[4/4] AR v2 — Beam Search (Constrained, beam=5)...")
        res_beam_con = evaluate_model_ar(
            model, loader, vocab, device,
            use_beam=True, beam_size=5, constrain=True, constrain_penalty=10.0,
        )
        all_results["AR v2 (beam=5, constrained)"] = res_beam_con
        print(f"  CER: {res_beam_con['cer']*100:.2f}%  WER: {res_beam_con['wer']*100:.2f}%")

        # ── Error Analysis (use best variant) ──
        best_key = min(all_results, key=lambda k: all_results[k]["cer"])
        best_res = all_results[best_key]
        print(f"\n[ERROR ANALYSIS] Using best variant: {best_key}")
        print_error_examples(best_res["predictions"], best_res["ground_truths"], n=20)
        confusion = character_confusion_matrix(best_res["predictions"], best_res["ground_truths"], top_n=20)

        # Generate confusion matrix heatmap
        generate_confusion_heatmap(confusion, title="AR Transformer v2")

    else:
        print(f"[SKIP] AR v2 checkpoint not found at {ar_v2_ckpt}")

    # ─── CTC Baseline Model ─────────────────────────────────────
    ctc_ckpt = "checkpoints/ctc/best.pt"
    ctc_cfg_path = "configs/ctc_config.yaml"

    if os.path.exists(ctc_ckpt) and os.path.exists(ctc_cfg_path):
        print("\n" + "=" * 70)
        print("  Evaluating CTC Baseline")
        print("=" * 70)

        cfg = yaml.safe_load(open(ctc_cfg_path))
        mcfg = cfg["model"]
        dcfg = cfg["data"]
        vocab = TeluguVocab.load(cfg["training"]["vocab_path"])

        from src.models.ctc_model import CTCModel
        model = CTCModel(
            vocab_size=len(vocab),
            d_model=mcfg["d_model"],
            lstm_hidden=mcfg["lstm_hidden"],
            lstm_layers=mcfg["lstm_layers"],
            dropout=mcfg["dropout"],
            pretrained=False,
        )
        ckpt = torch.load(ctc_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(device)

        loader = build_dataloader(
            dcfg.get("test_annotation", "data/raw/test/labels.txt"),
            dcfg.get("test_image_root", "data/raw/test"),
            vocab, split="test", batch_size=64,
            num_workers=dcfg.get("num_workers", 4),
            max_label_len=dcfg["max_label_len"],
            add_sos_eos=False,
        )

        print("\n[CTC] Greedy decoding...")
        res_ctc = evaluate_model_ctc(model, loader, vocab, device)
        all_results["CTC Baseline"] = res_ctc
        print(f"  CER: {res_ctc['cer']*100:.2f}%  WER: {res_ctc['wer']*100:.2f}%")

    else:
        print(f"[SKIP] CTC checkpoint not found at {ctc_ckpt}")

    # ─── Print Full Ablation Table ───────────────────────────────
    if all_results:
        print_ablation_table(all_results)

        # Generate ablation bar chart
        generate_ablation_chart(all_results)

        # Save results JSON
        save_results = {}
        for name, res in all_results.items():
            save_results[name] = {
                "cer": res["cer"],
                "wer": res["wer"],
                "cer_ci": res.get("cer_ci", (0, 0)),
                "speed_ms": res.get("inference_time_ms_per_sample", 0),
                "avg_pred_len": res.get("avg_pred_len", 0),
                "virama_breakdown": {
                    k: {kk: vv for kk, vv in v.items()} 
                    for k, v in res.get("virama_breakdown", {}).items()
                },
            }

        json_path = os.path.join(RESULTS_DIR, "all_results.json")
        with open(json_path, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\n[SAVED] {json_path}")

    return all_results


# ═══════════════════════════════════════════════════════════════════
# Part 3: Generate Confusion Matrix Heatmap
# ═══════════════════════════════════════════════════════════════════

def generate_confusion_heatmap(confusion_counter, title="Model", top_n=15):
    """Generate a heatmap of character confusions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Try to use a Telugu-capable font
    try:
        telugu_fonts = [f for f in font_manager.findSystemFonts() if "noto" in f.lower() and "telugu" in f.lower()]
        if telugu_fonts:
            prop = font_manager.FontProperties(fname=telugu_fonts[0])
        else:
            prop = None
    except:
        prop = None

    most_common = confusion_counter.most_common(top_n)
    if not most_common:
        print("[SKIP] No confusions to plot.")
        return

    # Get unique chars
    gt_chars = list(dict.fromkeys([g for (g, p), _ in most_common]))
    pred_chars = list(dict.fromkeys([p for (g, p), _ in most_common]))

    # Build matrix
    all_chars = list(dict.fromkeys(gt_chars + pred_chars))
    n = len(all_chars)
    char_to_idx = {c: i for i, c in enumerate(all_chars)}
    matrix = np.zeros((n, n))

    for (g, p), count in most_common:
        if g in char_to_idx and p in char_to_idx:
            matrix[char_to_idx[g], char_to_idx[p]] = count

    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    # Labels
    fontprops = {"fontproperties": prop} if prop else {}
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_chars, fontsize=12, **fontprops)
    ax.set_yticklabels(all_chars, fontsize=12, **fontprops)
    ax.set_xlabel("Predicted Character", fontsize=13)
    ax.set_ylabel("Ground Truth Character", fontsize=13)
    ax.set_title(f"Character Confusion Matrix — {title}", fontsize=14, fontweight="bold")

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = int(matrix[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=9, color="white" if val > matrix.max() * 0.6 else "black")

    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "confusion_matrix_ar_v2.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")

    path_pdf = os.path.join(RESULTS_DIR, "confusion_matrix_ar_v2.pdf")
    fig.savefig(path_pdf, dpi=300, bbox_inches="tight") if False else None  # PDF needs re-render
    print(f"[SAVED] {path}")


# ═══════════════════════════════════════════════════════════════════
# Part 4: Generate Ablation Bar Chart
# ═══════════════════════════════════════════════════════════════════

def generate_ablation_chart(results):
    """Generate a bar chart comparing CER across model variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    cers = [results[n]["cer"] * 100 for n in names]
    wers = [results[n]["wer"] * 100 for n in names]

    # Shorten names for display
    short_names = []
    for n in names:
        n = n.replace("AR v2 ", "").replace("(", "").replace(")", "")
        short_names.append(n)

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, cers, width, label="CER (%)", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, wers, width, label="WER (%)", color="#FF9800", alpha=0.85)

    ax.set_xlabel("Model Variant", fontsize=13)
    ax.set_ylabel("Error Rate (%)", fontsize=13)
    ax.set_title("Ablation Study — Telugu HTR Model Comparison", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15, ha="right", fontsize=10)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.2f}%",
                ha="center", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}%",
                ha="center", fontsize=9)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "ablation_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    path_pdf = os.path.join(RESULTS_DIR, "ablation_comparison.pdf")
    fig.savefig(path_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")
    print(f"[SAVED] {path_pdf}")


# ═══════════════════════════════════════════════════════════════════
# Part 5: CER Progression Table (for paper)
# ═══════════════════════════════════════════════════════════════════

def print_cer_progression():
    """Print the CER progression across epochs for the paper table."""
    milestones = [
        (1,  18.06),
        (2,   9.25),
        (5,   6.39),
        (10,  3.97),
        (20,  3.50),
        (30,  3.43),
        (40,  2.87),
        (50,  2.83),
        (60,  2.59),
        (72,  2.40),  # Best
        (80,  2.42),
    ]

    print("\n" + "=" * 50)
    print("  CER Progression — AR Transformer v2")
    print("=" * 50)
    print(f"  {'Epoch':<10} {'Val CER':<12} {'Status'}")
    print("-" * 50)
    for epoch, cer in milestones:
        status = "★ BEST" if cer == 2.40 else ""
        print(f"  {epoch:<10} {cer:.2f}%{'':<7} {status}")
    print("=" * 50)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all IEEE paper results and figures")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip model evaluation, only generate plots from existing data")
    parser.add_argument("--plots-only", action="store_true",
                        help="Only generate training curves (no GPU needed)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Telugu HTR — Full Evaluation & Figure Generation")
    print("=" * 70)

    # Always generate training curves (no GPU needed)
    print("\n[STEP 1] Generating training curves...")
    generate_training_curves()

    # Print CER progression table
    print_cer_progression()

    if args.plots_only:
        print("\n[DONE] Plots generated. Use --skip-eval=false to run full evaluation.")
        sys.exit(0)

    if not args.skip_eval:
        print("\n[STEP 2] Running model evaluations...")
        results = run_evaluations()
    else:
        print("\n[SKIP] Model evaluation skipped (--skip-eval)")

    print("\n" + "=" * 70)
    print("  ALL DONE! Figures saved to: results/paper_figures/")
    print("=" * 70)
    print("\nFiles generated:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        size = os.path.getsize(os.path.join(RESULTS_DIR, f))
        print(f"  {f:40s} {size/1024:.1f} KB")

```

## 7. IEEE Paper (LaTeX)

### paper/main.tex
```latex
\documentclass[conference]{IEEEtran}

% ═══════════════════════════════════════════════════════════════════
% Packages
% ═══════════════════════════════════════════════════════════════════
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{hyperref}
\usepackage{tabularx}
\usepackage{array}
\usepackage{balance}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}
\sloppy

% ═══════════════════════════════════════════════════════════════════
% Title and Authors
% ═══════════════════════════════════════════════════════════════════
\title{Telugu Handwritten Word Recognition using Autoregressive Transformer Decoder with Joint CTC-Attention Training}

\author{
\IEEEauthorblockN{Sakilam Bhargav}
\IEEEauthorblockA{
\textit{Dept. of CSE (Data Science)} \\
\textit{B.V. Raju Institute of Technology} \\
Narsapur, India \\
23211a67a7@bvrit.ac.in}
\and
\IEEEauthorblockN{Dr. R. Venkata Ramana Chary}
\IEEEauthorblockA{
\textit{Dept. of CSE (Data Science)} \\
\textit{B.V. Raju Institute of Technology} \\
Narsapur, India \\
ramanachary.rv@bvrit.ac.in}
\and
\IEEEauthorblockN{Thatha Nikitha}
\IEEEauthorblockA{
\textit{Dept. of CSE (Data Science)} \\
\textit{B.V. Raju Institute of Technology} \\
Narsapur, India \\
23211a67b6@bvrit.ac.in}
\and
\IEEEauthorblockN{Sabavat Vinod Nayak}
\IEEEauthorblockA{
\textit{Dept. of CSE (Data Science)} \\
\textit{B.V. Raju Institute of Technology} \\
Narsapur, India \\
23211a67a6@bvrit.ac.in}
}

\maketitle

% ═══════════════════════════════════════════════════════════════════
% Abstract
% ═══════════════════════════════════════════════════════════════════
\begin{abstract}
Despite steady advances in optical character recognition, handwritten text in Telugu---a Dravidian language with over 80~million speakers---continues to resist accurate machine reading. The script's syllabic structure, where consonants fuse with vowel diacritics and stack through virama-based conjuncts, produces hundreds of visually distinct compound glyphs that share few visual cues with their constituent parts. We compare two recognition architectures on the IIIT-HW-Telugu benchmark: a CNN-BiLSTM-CTC baseline and an autoregressive Transformer decoder trained with a joint CTC-attention loss. Both share a stride-patched ResNet-18 visual encoder designed to preserve horizontal spatial resolution for sequential decoding, and we additionally incorporate a data-driven Telugu character transition matrix to constrain the decoder's output space at inference time. On 17,910 held-out word images, our best Transformer variant reaches a Character Error Rate~(CER) of 3.66\%, a 6.4\% relative reduction compared with the CTC baseline~(3.91\%). During development, we found that two architectural details---scaling token embeddings by $\sqrt{d_{\text{model}}}$ and tying the input and output embedding weights---were essential for Transformer convergence on Telugu. A qualitative error analysis exposes a practical tradeoff between the two decoders: CTC tends to make many small, distributed character substitutions, whereas the Transformer is mostly more accurate but can occasionally undergo autoregressive collapse, generating a fluent-looking word that is entirely wrong. Our code and trained models are publicly available.
\end{abstract}

\begin{IEEEkeywords}
handwritten text recognition, Telugu, CTC, Transformer, autoregressive decoding, Indic scripts, linguistic constraints
\end{IEEEkeywords}

% ═══════════════════════════════════════════════════════════════════
% I. INTRODUCTION
% ═══════════════════════════════════════════════════════════════════
\section{Introduction}

For Latin-script languages, handwritten text recognition~(HTR) is largely a practical success---modern encoder-decoder pipelines routinely achieve single-digit character error rates on benchmarks like IAM and RIMES~\cite{vaswani2017attention, li2023trocr}. Indic scripts, however, tell a different story. Telugu, the fourth most spoken scheduled language in India with over 82~million native speakers and one of six languages carrying classical status~\cite{ethnologue2024}, has received comparatively little attention from the HTR community despite its large speaker base.

What makes Telugu particularly hard for HTR is its syllabic orthography. The script uses 56~base consonants (\textit{hallulu}), 16~vowels (\textit{acchulu}), and vowel diacritics (\textit{matras}) that attach to and reshape consonant glyphs. The virama character (\textit{halant}, U+0C4D) stacks consonants into conjuncts (\textit{vattu} forms), and the resulting compound glyphs can look nothing like the individual consonants they are composed from---a property that sets Telugu apart from Latin-based scripts where letter shapes stay recognizable in most combinations.

Two main decoding strategies dominate current HTR work. CTC~\cite{graves2006ctc} sidesteps the need for character-level alignment labels: a recurrent encoder emits a probability distribution at every horizontal position, and a dynamic-programming step sums over all valid ways those distributions could produce the target string. This framewise approach is fast at inference and naturally suits left-to-right scripts, but each character prediction is made independently of its neighbours. Autoregressive~(AR) Transformer decoders~\cite{vaswani2017attention} take a different route, generating one character at a time and conditioning each step on every character produced so far through cross-attention to the visual features. This lets them capture long-range character dependencies---for example, knowing that a particular vowel sign is unlikely after a given consonant cluster---but they are heavier to train and susceptible to exposure bias, where the decoder's own errors feed back into subsequent predictions.

TrOCR~\cite{li2023trocr} and PARSeq~\cite{bautista2022parseq} have convincingly demonstrated that, given millions of pre-training images, Transformer decoders outperform CTC. But this advantage may not transfer to low-resource scripts: Gongidi and Jawahar~\cite{gongidi2021iiit} and Deshpande~et~al.~\cite{deshpande2021} both found CTC architectures to be surprisingly competitive for Indic languages when labelled data is scarce. This tension raises a practical question that we set out to answer: \textit{Given a moderately-sized Telugu word-image dataset ($\approx$80k training samples), does an autoregressive Transformer actually beat a well-tuned CTC baseline, and if so, by how much?}

Concretely, this paper makes the following contributions:

\begin{enumerate}
    \item A shared visual encoder built on ResNet-18 with modified stride patterns that keeps the horizontal resolution intact for both CTC and Transformer decoders.
    
    \item An autoregressive Transformer decoder trained with a joint CTC-attention loss, together with an analysis of two architectural details---embedding scaling and weight tying---without which the Transformer fails to converge on Telugu data.
    
    \item A data-driven Telugu character transition matrix, mined from training-set bigram counts, that we test as a soft constraint during decoding.
    
    \item A head-to-head comparison on the IIIT-HW-Telugu benchmark~\cite{gongidi2021iiit}, where our best model reaches 3.66\% CER---to our knowledge the lowest lexicon-free error rate reported on this dataset, beating the previous best of 4.58\% by Dutta~et~al.~\cite{dutta2018}.
\end{enumerate}

% ═══════════════════════════════════════════════════════════════════
% II. RELATED WORK
% ═══════════════════════════════════════════════════════════════════
\section{Related Work}

\subsection{CTC-Based Handwriting Recognition}

CTC~\cite{graves2006ctc} remains the workhorse loss for HTR because it needs only image--transcript pairs, not character bounding boxes. Shi~et~al.~\cite{shi2017crnn} popularized the CRNN pipeline---convolutional features fed to a BiLSTM and decoded with CTC---and it quickly became the default baseline in the field. Puigcerver~\cite{puigcerver2017} later showed that width-preserving convolutions strengthen the encoder, while Scheidl~et~al.~\cite{scheidl2018word} explored language-model-aware beam search at the CTC output. These incremental refinements have kept CTC competitive even as attention-based alternatives have matured.

\subsection{Attention and Transformer-Based HTR}

Transformers entered the HTR space through scene-text recognition. Li~et~al.'s TrOCR~\cite{li2023trocr} pairs a pre-trained ViT encoder with a GPT-2-style decoder and achieves leading results on several benchmarks, though its success hinges on large-scale pre-training that simply does not exist for most Indic scripts. Bautista and Atienza's PARSeq~\cite{bautista2022parseq} takes a different angle, training a single model under every possible character-order permutation so that it can decode both left-to-right and in parallel. Neither system has been evaluated on Telugu, leaving open the question of whether their gains carry over to syllabic writing systems.

\subsection{Indic Script Recognition}

Published work on Telugu HTR is limited. Dutta~et~al.~\cite{dutta2018} were among the first to train CNN-RNN-CTC pipelines on the IIIT-HW-Telugu dataset, reporting around 15\% CER; with IAM pre-training they brought this down to 4.58\%. Gongidi and Jawahar~\cite{gongidi2021iiit} curated the broader IIIT-INDIC-HW-WORDS benchmark and reached 6.41\% CER with a CRNN model, dropping to 1.52\% CER when a lexicon was used during decoding---an impressive number, but one that requires a closed vocabulary. Deshpande~et~al.~\cite{deshpande2021} tried attention-based decoders for Indic characters and reported approximately 10.2\% CER. More recently, PLATTER~\cite{platter2025} tackles page-level Indic HTR, though its evaluation covers Hindi and Bangla rather than Telugu.

\subsection{Joint CTC-Attention Training}

The idea of adding an auxiliary CTC loss to an attention decoder was first explored in automatic speech recognition by Watanabe~et~al.~\cite{watanabe2017hybrid}. Their insight was straightforward: the CTC branch forces the encoder to produce features that are already roughly aligned with the target sequence, which makes it easier for the attention mechanism to learn a clean left-to-right alignment instead of collapsing. Michael~et~al.~\cite{michael2019evaluating} later applied this hybrid strategy to HTR and observed faster convergence and fewer attention-collapse episodes compared with pure attention training. We adopt the same joint-loss formulation in our Transformer decoder.

% ═══════════════════════════════════════════════════════════════════
% III. PROPOSED METHOD
% ═══════════════════════════════════════════════════════════════════
\section{Proposed Method}

The recognition pipeline has four parts, illustrated in Fig.~\ref{fig:architecture}: a CNN-based feature encoder shared by both decoders, a CTC baseline decoder, an autoregressive Transformer decoder with an auxiliary CTC head, and an optional Telugu character-transition mask applied at inference.

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.92\textwidth]{figures/system_architecture.jpg}
    \caption{System overview. The stride-patched ResNet-18 encoder converts a word image into a 64-step feature sequence by keeping layers~3--4 strides at $(2,1)$ instead of $(2,2)$. Two decoder paths share this representation: \textbf{Path~A} feeds it through a 2-layer BiLSTM for CTC decoding; \textbf{Path~B} refines it with a 3-layer Transformer encoder, branches off an auxiliary CTC head ($\lambda{=}0.3$), and decodes autoregressively with a 6-layer Transformer decoder. An optional Telugu grammar mask penalises unlikely character transitions at inference.}
    \label{fig:architecture}
\end{figure*}

\subsection{CNN Feature Encoder}

Our visual encoder starts from a standard ResNet-18~\cite{he2016resnet} pre-trained on ImageNet, but we change how it handles spatial dimensions. An unmodified ResNet-18 reduces both height and width by a factor of 32, which would leave a 512-pixel-wide input image with only 16~horizontal positions---far too few for CTC or attention to decode a typical 8--10~character Telugu word. We fix this by changing the stride of the first convolution in layers~3 and~4 from $(2,2)$ to $(2,1)$:
\begin{equation}
    \text{layer3}[0].\text{stride}: (2,2) \rightarrow (2,1)
\end{equation}
\begin{equation}
    \text{layer4}[0].\text{stride}: (2,2) \rightarrow (2,1)
\end{equation}

The height still shrinks (we do not need vertical resolution at this stage), but the width is preserved, resulting in feature maps of shape $[B, 512, 4, W/8]$. Since the input is grayscale, we replicate it to three channels to match the pre-trained weights. A $1 \times 1$ convolution then projects the 512~channels down to $d_{\text{model}}$, and the height dimension is collapsed by adaptive average pooling, yielding a sequence of $T = W/8 = 64$ feature vectors. Sinusoidal positional encodings~\cite{vaswani2017attention} are added so the decoder knows where each feature vector sits along the word.

\subsection{CTC Baseline Architecture}

For our CTC baseline, we stack a 2-layer bidirectional LSTM (hidden size~256, giving 512-dimensional concatenated states) on top of the shared encoder features. A linear head projects these states to logits over the 91-token vocabulary. We train with CTC loss~\cite{graves2006ctc}, reusing the PAD token (index~0) as the CTC blank. BiLSTM weights are initialized with the orthogonal scheme following standard practice. At test time we decode greedily, collapsing repeated tokens and blanks in a single pass.

\subsection{Autoregressive Transformer Decoder}

Our Transformer decoder is based on the standard architecture of Vaswani~et~al.~\cite{vaswani2017attention}, with a few practical changes. Each of the $N$ decoder layers contains masked self-attention, cross-attention over the encoder memory, and a feed-forward block with GELU activations. We apply layer normalisation \textit{before} each sub-layer (Pre-LN), which Xiong~et~al.~\cite{xiong2020layer} showed stabilises gradient flow in deep Transformers. Optionally, a lightweight 3-layer Transformer encoder refines the CNN features before they reach the decoder's cross-attention heads.

Two architectural details proved critical for convergence on our Telugu dataset:

\textbf{Embedding Scaling.} We found empirically that scaling token embeddings by $\sqrt{d_{\text{model}}}$---as originally suggested by Vaswani~et~al.~\cite{vaswani2017attention}---is not optional for our Telugu vocabulary:
\begin{equation}
    \mathbf{e}_t = \text{Embed}(y_t) \cdot \sqrt{d_{\text{model}}} + \text{PE}(t)
\end{equation}
Without this factor, the sinusoidal positional signal drowns out the token identity, and the model ends up attending based on position rather than character content. This failure mode is especially damaging for variable-length Telugu words, where the same character can appear at many different positions.

\textbf{Weight Tying.} Following Press and Wolf~\cite{press2017output}, we share the weight matrix between the input embedding layer and the output softmax projection:
\begin{equation}
    P(y_t | y_{<t}, \mathbf{h}) = \text{softmax}(\mathbf{W}_{\text{embed}} \cdot \mathbf{d}_t)
\end{equation}
Besides cutting the parameter count, this forces each token's embedding to work well both for representing input context and for predicting the next character---an implicit regulariser that we found helpful on our 91-token vocabulary.

\textbf{Joint CTC-CE Training.} We train with a weighted sum of cross-entropy and an auxiliary CTC loss computed on the encoder output:
\begin{equation}
    \mathcal{L} = (1 - \lambda) \cdot \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{CTC}}
    \label{eq:joint_loss}
\end{equation}
with $\lambda = 0.3$. The cross-entropy term includes label smoothing ($\epsilon = 0.05$, after Szegedy~et~al.~\cite{szegedy2016rethinking}). The CTC branch nudges the encoder toward producing features that are already roughly time-aligned with the target characters, which in turn lets the attention decoder converge faster and more reliably.

\subsection{Telugu Linguistic Constraint Mask}

Telugu has phonotactic constraints---not every character can follow every other character. To exploit this, we scan the training transcripts and record which character bigrams actually occur, storing the result in a binary matrix $\mathbf{M} \in \{0, 1\}^{|V| \times |V|}$:
\begin{equation}
    \mathbf{M}[i][j] = \begin{cases} 1 & \text{if bigram } (c_i, c_j) \text{ observed in training} \\ 0 & \text{otherwise} \end{cases}
\end{equation}

At each decoding step $t$, we look up the row corresponding to the most recently generated token $c_{t-1}$ and subtract a large penalty $\delta = 10.0$ from the logits of any character not seen after $c_{t-1}$ in training:
\begin{equation}
    \ell'_j = \ell_j - \delta \cdot (1 - \mathbf{M}[c_{t-1}][j])
\end{equation}
We deliberately use a soft penalty rather than hard masking so that the decoder can still emit rare-but-valid pairs that were simply not observed in the 80k training words.

% ═══════════════════════════════════════════════════════════════════
% IV. EXPERIMENTS
% ═══════════════════════════════════════════════════════════════════
\section{Experiments}

\subsection{Dataset}

All experiments use the IIIT-HW-Telugu dataset~\cite{dutta2018, gongidi2021iiit}, a multi-writer collection of word-level handwritten Telugu images. Table~\ref{tab:dataset} summarises the split sizes.

\begin{table}[htbp]
\centering
\caption{IIIT-HW-Telugu Dataset Statistics}
\label{tab:dataset}
\begin{tabular}{lccc}
\toprule
\textbf{Split} & \textbf{Images} & \textbf{Vocab Size} & \textbf{Avg. Length} \\
\midrule
Train & 80,693 & 87 chars & 8.7 chars \\
Validation & 20,048 & -- & -- \\
Test & 17,910 & -- & -- \\
\midrule
Total & 118,651 & 91 tokens\textsuperscript{*} & -- \\
\bottomrule
\end{tabular}
\vspace{2pt}
\raggedright\small\textsuperscript{*}Including 4 special tokens: PAD, SOS, EOS, UNK. All labels are NFC-normalized.
\end{table}

Every image is resized to $64 \times 512$ pixels, padding the shorter side to preserve the aspect ratio, and pixel values are normalised to $[-1, 1]$ (mean~0.5, std~0.5). During training we apply several on-the-fly augmentations to simulate the variability of real handwriting: random rotation up to $\pm 5\degree$ ($p{=}0.5$), random perspective warping ($p{=}0.3$), brightness and contrast jitter, morphological dilation or erosion to mimic pen-thickness changes ($p{=}0.3$), and elastic distortion through Gaussian-smoothed displacement fields ($\alpha{=}10$, $\sigma{=}3$, $p{=}0.3$). Because all augmentations run on the CPU during data loading, they add virtually no GPU overhead. Fig.~\ref{fig:pipeline} shows the full preprocessing pipeline.

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/data_pipeline.jpg}
    \caption{Preprocessing and augmentation pipeline. Each image passes through fixed steps (grayscale, resize, pad, normalise) and then, during training only, through a chain of random augmentations. Augmentations are applied independently with the probabilities shown, so every epoch sees slightly different versions of every sample.}
    \label{fig:pipeline}
\end{figure}

\subsection{Implementation Details}

\begin{table}[htbp]
\centering
\caption{Training Hyperparameters}
\label{tab:hyperparams}
\begin{tabular}{lcc}
\toprule
\textbf{Parameter} & \textbf{CTC} & \textbf{AR Transformer} \\
\midrule
$d_{\text{model}}$ & 256 & 384 \\
Encoder layers & -- & 3 (Transformer) \\
Decoder layers & 2 (BiLSTM) & 6 (Transformer) \\
Attention heads & -- & 8 \\
$d_{\text{ff}}$ & -- & 1,536 \\
Batch size & 64 & 64 \\
Learning rate & $3 \times 10^{-4}$ & $3 \times 10^{-4}$ \\
LR schedule & Cosine annealing & Warmup + cosine \\
Warmup steps & -- & 4,000 \\
Optimizer & AdamW & AdamW \\
Weight decay & $10^{-4}$ & $10^{-4}$ \\
Dropout & 0.1 & 0.15 \\
Label smoothing & -- & 0.05 \\
CTC weight ($\lambda$) & -- & 0.3 \\
Grad clip & 5.0 & 5.0 \\
Epochs & 50 & 80 \\
Mixed precision & FP16 & FP16 \\
\bottomrule
\end{tabular}
\end{table}

We ran every experiment on a single NVIDIA GPU with 24~GB of VRAM, using PyTorch~2.7.1 and CUDA~11.8. Wall-clock training time was roughly 3.5~hours for the CTC model (50~epochs) and 5.4~hours for the AR Transformer (80~epochs). Both models train in FP16 mixed precision via PyTorch's built-in GradScaler.

\subsection{Evaluation Protocol}

We measure accuracy with two standard metrics. Character Error Rate~(CER) is the total edit distance across all test samples divided by the total number of ground-truth characters:
\begin{equation}
    \text{CER} = \frac{\sum_{i=1}^{N} \text{ED}(\hat{y}_i, y_i)}{\sum_{i=1}^{N} |y_i|}
\end{equation}
where $\text{ED}(\cdot, \cdot)$ denotes Levenshtein distance. Word Error Rate~(WER) is simply the fraction of words where the prediction contains any error at all. To judge whether observed differences are meaningful, we use non-parametric bootstrap resampling~\cite{efron1993bootstrap} with 1,000 resamples and report 95\% percentile confidence intervals.

% ═══════════════════════════════════════════════════════════════════
% V. RESULTS AND ANALYSIS
% ═══════════════════════════════════════════════════════════════════
\section{Results and Analysis}
\label{sec:results}

\subsection{Main Results}

Table~\ref{tab:ablation} lists the CER, WER, and inference speed of every model variant we trained, evaluated on the 17,910 held-out test images.

\begin{table*}[t]
\centering
\caption{Test-Set Results on IIIT-HW-Telugu ($N = 17{,}910$)}
\label{tab:ablation}
\begin{tabular}{llccccc}
\toprule
\textbf{\#} & \textbf{Model} & \textbf{CER (\%)} & \textbf{WER (\%)} & \textbf{95\% CI (CER)} & \textbf{Comp. CER (\%)} & \textbf{Speed (ms)} \\
\midrule
1 & CTC Baseline (BiLSTM) & 3.91 & 24.80 & [3.79, 4.04] & 3.70 & \textbf{0.3} \\
2 & AR Transformer (greedy, unconstrained) & 3.67 & 23.33 & [3.55, 3.79] & 3.32 & 1.4 \\
3 & AR Transformer (greedy, constrained) & 3.74 & 23.32 & [3.60, 3.88] & 3.33 & 1.8 \\
4 & AR Transformer (beam-5, unconstrained) & \textbf{3.66} & 23.34 & [3.54, 3.79] & \textbf{3.31} & 77.0 \\
5 & AR Transformer (beam-5, constrained) & 3.67 & \textbf{23.28} & [3.55, 3.79] & 3.30 & 82.5 \\
\bottomrule
\end{tabular}
\end{table*}

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/training_curves.png}
    \caption{AR Transformer training curves over 80~epochs. \textbf{Top:} The joint loss ($0.7\mathcal{L}_{\text{CE}} + 0.3\mathcal{L}_{\text{CTC}}$) drops steadily on the training set; validation loss flattens around epoch~55. \textbf{Bottom:} Validation CER bottoms out at 2.40\% at epoch~72 ($\star$); beyond that point, mild overfitting sets in.}
    \label{fig:training}
\end{figure}

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/ablation_comparison.png}
    \caption{Test-set CER for every model configuration. All four Transformer variants (rows~2--5) beat the CTC baseline (row~1); beam search adds only a small further gain. Error bars are 95\% bootstrap confidence intervals.}
    \label{fig:ablation}
\end{figure}

We highlight three patterns worth noting (see also Figs.~\ref{fig:training} and~\ref{fig:ablation}):

\noindent\textbf{(1) The Transformer beats CTC.} Our best Transformer configuration (Row~4, beam-5 without constraints) reaches 3.66\% CER---a 6.4\% relative drop from the CTC baseline's 3.91\%. The two 95\% confidence intervals barely overlap ([3.54,~3.79] vs.~[3.79,~4.04]), which gives us reasonable confidence that the gap is real, not noise. We should note that our earliest Transformer runs performed \textit{worse} than CTC; the improvement only appeared after we added embedding scaling and weight tying (Section~III-C).

\noindent\textbf{(2) Beam search barely helps.} Greedy decoding already gives 3.67\% CER; widening the beam to~5 shaves off just 0.01 percentage points, and the confidence intervals overlap entirely. This tells us that the encoder features, not the search strategy, are the bottleneck---the decoder is already quite sure of its top-1 prediction.

\noindent\textbf{(3) The grammar mask helps WER but hurts CER.} Applying the Telugu transition matrix actually pushes greedy CER up slightly (3.67\%~$\to$~3.74\%), most likely because it penalises rare-but-valid character pairs that were absent from the training bigrams. At the word level, though, constrained beam search achieves the lowest WER of all configurations (23.28\%), suggesting that the mask does suppress some whole-word errors even if it introduces a few extra character-level mistakes.

\subsection{Comparison with Published Results}

Table~\ref{tab:comparison} puts our numbers alongside previously reported Telugu HTR results.

\begin{table}[htbp]
\centering
\caption{Comparison with Published Telugu HTR Results}
\label{tab:comparison}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{CER (\%)} & \textbf{WER (\%)} & \textbf{Lexicon} \\
\midrule
Dutta et al.~\cite{dutta2018} CNN-BLSTM & 9.15 & 37.92 & No \\
Dutta et al.~\cite{dutta2018} + IAM pretrain & 4.58 & 23.98 & No \\
Gongidi \& Jawahar~\cite{gongidi2021iiit} CRNN & $\sim$7.8 & $\sim$35.0 & No \\
Gongidi \& Jawahar~\cite{gongidi2021iiit} + Lexicon & 1.54 & 2.84 & \textbf{Yes} \\
\midrule
Ours: CTC Baseline & 3.91 & 24.80 & No \\
\textbf{Ours: AR Transformer} & \textbf{3.66} & \textbf{23.28} & No \\
\bottomrule
\end{tabular}
\end{table}

Among lexicon-free systems, our AR Transformer sets a new best on IIIT-HW-Telugu: 3.66\% CER is a \textbf{20\% relative reduction} from the 4.58\% that Dutta~et~al.~\cite{dutta2018} achieved with IAM pre-training, and a \textbf{53\% relative reduction} from the raw CRNN baseline of Gongidi and Jawahar~\cite{gongidi2021iiit}. The only system with lower CER (1.54\%) relies on a closed lexicon, meaning it can only predict words it has seen before---a restriction that limits practical use.

\subsection{Compound vs.\ Simple Character Analysis}

To understand where the models struggle, we split the test set into \textit{compound} words (those containing at least one virama, $n \approx 12{,}000$) and \textit{simple} words (no virama at all, $n \approx 5{,}900$).

\begin{table}[htbp]
\centering
\caption{CER by Word Type}
\label{tab:compound}
\begin{tabular}{lcc}
\toprule
\textbf{Word Type} & \textbf{CTC CER (\%)} & \textbf{AR CER (\%)} \\
\midrule
Compound (with virama) & 3.70 & \textbf{3.31} \\
Simple (no virama) & 4.29 & 4.20 \\
\bottomrule
\end{tabular}
\end{table}

Somewhat surprisingly, both models do \textit{better} on compound words than on simple ones. We think the reason is that conjunct glyphs, once the model has seen enough examples, are actually quite distinctive---their fused shapes are unambiguous. Simple words, by contrast, tend to be short and contain common characters that look similar to one another, leaving fewer visual cues for the decoder to latch onto.

\subsection{Qualitative Error Analysis}

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/confusion_matrix_ar_v2.png}
    \caption{Character-level confusion matrix for the AR Transformer (test set). Darker cells indicate more frequent confusions. The dominant errors are between consonant pairs that share stroke structure (e.g., \textit{dda}$\leftrightarrow$\textit{da}) and between vowel signs that differ by a single stroke.}
    \label{fig:confusion}
\end{figure}

Looking at individual error cases tells us something that aggregate CER cannot: the two architectures fail in qualitatively different ways. Fig.~\ref{fig:confusion} shows the Transformer's character-level confusion matrix; Fig.~\ref{fig:examples} compares representative predictions from both models.

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/prediction_examples.jpg}
    \caption{Side-by-side predictions from CTC and the AR Transformer. The top rows are correct from both; the bottom rows show typical failures. CTC swaps visually similar consonants, while the Transformer occasionally hallucinates a fluent but completely wrong word.}
    \label{fig:examples}
\end{figure}

\textbf{CTC errors} are almost always \textit{local substitutions}---one or two characters in a word get swapped for visually similar alternatives. The most common culprits are consonant pairs that share stroke geometry (\textit{dda}~$\leftrightarrow$~\textit{da}, 478 cases in the test set) and vowel-sign pairs that differ by a single stroke (\textit{e-matra}~$\leftrightarrow$~\textit{i-matra}, 169 cases). Because each CTC output depends only on the encoder, these mistakes are scattered evenly across samples and rarely corrupt more than a couple of characters per word.

\textbf{Transformer errors} look very different. Most of the time the decoder nails the word exactly (which is why its overall CER is lower), but when it does fail, the failure can be dramatic. We noticed multiple cases of what we call \textit{autoregressive collapse}: the decoder produces a perfectly fluent Telugu word that has nothing to do with the input image. In one striking example, the same common word was predicted for four entirely unrelated images. The mechanism is straightforward---once the first character goes wrong, self-attention over the decoder's own outputs reinforces the mistake, and the model confidently generates a coherent but incorrect sequence. CTC, which has no feedback loop from its own predictions, is structurally immune to this kind of failure.

\subsection{Validation-Test Gap}

The Transformer's best validation CER was 2.40\% (at epoch~72), noticeably lower than the 3.67\% it scored on the held-out test set. A gap of this size deserves explanation, and we see three likely causes:

\textbf{(1) Writer mismatch.} The test set includes handwriting styles that are not well represented in the validation set. Because the Transformer has more parameters than the CTC model, it is more prone to memorising writer-specific quirks during training.

\textbf{(2) Checkpoint selection.} We pick the checkpoint with the lowest validation CER out of 80~epochs, which inevitably favours a model that happens to do well on the particular validation samples---a mild form of overfitting to the validation distribution.

\textbf{(3) Scale.} Li~et~al.~\cite{li2023trocr} and Bautista and Atienza~\cite{bautista2022parseq} both trained their Transformer decoders on millions of images, often with additional pre-training. Our 80k training images may simply be too few for the Transformer to generalise as reliably as it could with more data.

% ═══════════════════════════════════════════════════════════════════
% VI. DISCUSSION
% ═══════════════════════════════════════════════════════════════════
\section{Discussion}

The CTC-versus-Transformer comparison is not as clean-cut as the overall CER numbers might suggest, and the details matter for practitioners choosing between the two.

CTC bakes in a strong assumption---characters appear left-to-right, and the encoder output is already roughly aligned with them. This acts as free regularization and makes CTC data-efficient: it reaches 3.91\% CER in 50~epochs, whereas the Transformer needs 80. The Transformer has no such built-in prior; it must discover the left-to-right alignment on its own through cross-attention, which takes more examples and more compute.

The autoregressive collapse we documented in Section~\ref{sec:results} has real-world implications. If an application demands predictable failure modes---say, digitizing historical Telugu manuscripts or legal records where a single completely wrong word is worse than several small typos---then CTC is the safer choice. For applications that simply need the lowest average error rate, the Transformer wins.

Our Telugu grammar mask did not help CER in a statistically meaningful way. In hindsight, this is not surprising: the transition matrix is built from training-set bigrams alone, so any valid character pair that happens to be absent from the 80k training words gets penalized at test time. A more principled approach would derive the mask from formal Telugu phonological rules or from a much larger unlabelled text corpus.

We see three natural extensions of this work. First, pre-training the Transformer decoder on synthetic Telugu text (rendered from digital fonts) could give it the language-modelling knowledge it currently lacks without requiring additional handwritten data. Second, non-autoregressive decoders along the lines of PARSeq~\cite{bautista2022parseq} could combine the Transformer's contextual awareness with CTC-like parallel decoding, sidestepping the exposure-bias problem. Third, a simple ensemble of CTC and Transformer predictions could exploit their complementary error patterns---CTC's distributed substitutions and the Transformer's occasional whole-word collapses rarely overlap on the same sample.

% ═══════════════════════════════════════════════════════════════════
% VII. CONCLUSION
% ═══════════════════════════════════════════════════════════════════
\section{Conclusion}

This paper compared CTC and autoregressive Transformer decoders for Telugu word-level handwriting recognition, keeping the visual encoder---a stride-patched ResNet-18---identical across both so that the comparison is fair. Our Transformer, trained with a joint CTC-attention loss, reaches 3.66\% CER on the IIIT-HW-Telugu test set, which to our knowledge is the best lexicon-free result reported on this benchmark---a 20\% relative improvement over the previous best of 4.58\% by Dutta~et~al.~\cite{dutta2018}. Along the way, we found that two seemingly minor architectural details (embedding scaling by $\sqrt{d_{\text{model}}}$ and weight tying) were essential for convergence on this script, and we documented a tradeoff that matters in practice: CTC makes many small, predictable character errors, while the Transformer is mostly more accurate but occasionally produces a fluent-looking word that is entirely incorrect. We release our code and trained models to support further work on Indic script recognition.

% ═══════════════════════════════════════════════════════════════════
% REPRODUCIBILITY
% ═══════════════════════════════════════════════════════════════════
\section*{Reproducibility}

We release all source code, configuration files, and training scripts at \url{https://github.com/[your-username]/telugu-htr}. The stack is Python~3.12.13 with PyTorch~2.7.1 (CUDA~11.8, cuDNN~9.1.0) on Ubuntu~22.04. A single 24~GB NVIDIA GPU is sufficient: the CTC model trains in about 3.5~hours (50~epochs) and the AR Transformer in about 5.4~hours (80~epochs). The IIIT-HW-Telugu dataset can be downloaded from \url{https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data}.

% ═══════════════════════════════════════════════════════════════════
% REFERENCES
% ═══════════════════════════════════════════════════════════════════
\bibliographystyle{IEEEtran}

\begin{thebibliography}{20}

\bibitem{vaswani2017attention}
A.~Vaswani, N.~Shazeer, N.~Parmar, J.~Uszkoreit, L.~Jones, A.~N.~Gomez, {\L}.~Kaiser, and I.~Polosukhin, ``Attention is all you need,'' in \textit{Proc. Advances in Neural Information Processing Systems (NeurIPS)}, vol.~30, 2017, pp.~5998--6008.

\bibitem{graves2006ctc}
A.~Graves, S.~Fern{\'a}ndez, F.~Gomez, and J.~Schmidhuber, ``Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks,'' in \textit{Proc. Int. Conf. Machine Learning (ICML)}, 2006, pp.~369--376.

\bibitem{he2016resnet}
K.~He, X.~Zhang, S.~Ren, and J.~Sun, ``Deep residual learning for image recognition,'' in \textit{Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)}, 2016, pp.~770--778.

\bibitem{li2023trocr}
M.~Li, T.~Lv, J.~Chen, L.~Cui, Y.~Lu, D.~Florencio, C.~Zhang, Z.~Li, and F.~Wei, ``TrOCR: Transformer-based optical character recognition with pre-trained models,'' in \textit{Proc. AAAI Conf. Artificial Intelligence}, vol.~37, 2023, pp.~13094--13102.

\bibitem{bautista2022parseq}
D.~Bautista and R.~Atienza, ``Scene text recognition with permuted autoregressive sequence models,'' in \textit{Proc. European Conf. Computer Vision (ECCV)}, 2022, pp.~178--196.

\bibitem{shi2017crnn}
B.~Shi, X.~Bai, and C.~Yao, ``An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition,'' \textit{IEEE Trans. Pattern Analysis and Machine Intelligence}, vol.~39, no.~11, pp.~2298--2304, 2017.

\bibitem{puigcerver2017}
J.~Puigcerver, ``Are multidimensional recurrent layers really necessary for handwritten text recognition?,'' in \textit{Proc. Int. Conf. Document Analysis and Recognition (ICDAR)}, 2017, pp.~67--72.

\bibitem{scheidl2018word}
H.~Scheidl, S.~Fiel, and R.~Sablatnig, ``Word beam search: A connectionist temporal classification decoding algorithm,'' in \textit{Proc. Int. Conf. Frontiers in Handwriting Recognition (ICFHR)}, 2018, pp.~253--258.

\bibitem{dutta2018}
K.~Dutta, P.~Krishnan, M.~Mathew, and C.~V.~Jawahar, ``Improving CNN-RNN hybrid networks for handwriting recognition,'' in \textit{Proc. Int. Conf. Frontiers in Handwriting Recognition (ICFHR)}, 2018, pp.~80--85.

\bibitem{gongidi2021iiit}
S.~Gongidi and C.~V.~Jawahar, ``IIIT-INDIC-HW-WORDS: A dataset for Indic handwritten text recognition,'' in \textit{Proc. Int. Conf. Document Analysis and Recognition (ICDAR)}, 2021, pp.~444--459.

\bibitem{deshpande2021}
S.~Deshpande, R.~Jayadevan, and R.~Kolhe, ``Handwritten Devanagari and Telugu character recognition using deep learning,'' in \textit{Proc. IEEE Int. Conf. Computing, Communication and Networking Technologies}, 2021.

\bibitem{platter2025}
A.~Namboodiri et~al., ``PLATTER: A page-level handwritten text recognition framework for Indic scripts,'' in \textit{Proc. Int. Conf. Document Analysis and Recognition (ICDAR)}, 2025.

\bibitem{watanabe2017hybrid}
S.~Watanabe, T.~Hori, S.~Kim, J.~R.~Hershey, and T.~Hayashi, ``Hybrid CTC/attention architecture for end-to-end speech recognition,'' \textit{IEEE J. Selected Topics in Signal Processing}, vol.~11, no.~8, pp.~1240--1253, 2017.

\bibitem{michael2019evaluating}
J.~Michael, R.~Labahn, T.~Gr{\"u}ning, and J.~Z{\"o}llner, ``Evaluating sequence-to-sequence models for handwritten text recognition,'' in \textit{Proc. Int. Conf. Document Analysis and Recognition (ICDAR)}, 2019, pp.~1286--1293.

\bibitem{xiong2020layer}
R.~Xiong et~al., ``On layer normalization in the Transformer architecture,'' in \textit{Proc. Int. Conf. Machine Learning (ICML)}, 2020, pp.~10524--10533.

\bibitem{press2017output}
O.~Press and L.~Wolf, ``Using the output embedding to improve language models,'' in \textit{Proc. European Chapter of the ACL}, 2017, pp.~157--163.

\bibitem{szegedy2016rethinking}
C.~Szegedy, V.~Vanhoucke, S.~Ioffe, J.~Shlens, and Z.~Wojna, ``Rethinking the inception architecture for computer vision,'' in \textit{Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)}, 2016, pp.~2818--2826.

\bibitem{ethnologue2024}
D.~M.~Eberhard, G.~F.~Simons, and C.~D.~Fennig, \textit{Ethnologue: Languages of the World}, 27th~ed. Dallas, TX: SIL International, 2024.

\bibitem{efron1993bootstrap}
B.~Efron and R.~J.~Tibshirani, \textit{An Introduction to the Bootstrap}. New York: Chapman \& Hall, 1993.

\end{thebibliography}

\end{document}

```

## 8. Requirements

### requirements.txt
```txt
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.24.0
editdistance>=0.6.3
pyyaml>=6.0
tensorboard>=2.13.0
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
pandas>=2.0.0
scipy>=1.10.0
opencv-python>=4.7.0

```

## 9. README

### README.md
```markdown
# Telugu Handwritten Text Recognition (HTR)

A research-grade system for recognising handwritten Telugu words using a
**CNN + Transformer autoregressive decoder** with a **data-driven Telugu
script constraint** that is built directly from the training labels —
never from hard-coded linguistic rules.

---

## Research Contribution

| Approach | Prior Work? | This Project |
|---|---|---|
| CNN + BiLSTM + CTC | Yes | Baseline benchmark |
| CNN + Transformer Encoder + Decoder (AR) | Yes | SOTA architecture baseline |
| + Data-derived Telugu constraints | Partially | Rigorous empirical evaluation |

**Core objective**: Empirically evaluate whether a data-derived, Telugu-script-aware autoregressive decoder outperforms standard CTC and plain autoregressive baselines on the IIIT-HW-Telugu benchmark, particularly on compound/ligature-heavy words (words containing Virama ్).

## Project Structure

```
major/
├── src/
│   ├── vocab.py                # Telugu vocab + data-driven transition matrix
│   ├── transforms.py           # Image preprocessing + augmentation
│   ├── dataset.py              # IIIT-HW-Telugu dataset loader
│   ├── checkpoint_manager.py   # Rolling 2-slot checkpoint system
│   ├── train_ctc.py            # CTC baseline training
│   ├── train_ar.py             # Autoregressive model training
│   ├── evaluate.py             # CER / WER + virama breakdown + error analysis
│   ├── models/
│   │   ├── cnn_encoder.py      # ResNet-18 encoder (stride-patched for HTR, supports S=64/128)
│   │   ├── ctc_model.py        # CNN + BiLSTM + CTC
│   │   └── ar_model.py         # CNN + Transformer Encoder + Decoder (greedy + beam)
│   └── decoding/
│       └── telugu_mask.py      # Data-driven Telugu constraint mask
├── configs/
│   ├── ctc_config.yaml         # CTC training hyperparameters
│   └── ar_config.yaml          # AR training hyperparameters
├── data/
│   └── raw/
│       ├── train/              # Train images + labels.txt
│       ├── val/                # Val images + labels.txt
│       └── test/               # Test images + labels.txt
├── checkpoints/                # Saved model weights (rolling 2-slot)
├── logs/                       # TensorBoard logs
├── notebooks/                  # Exploration + results notebooks
├── requirements.txt
└── README.md
```

---

## Dataset

**IIIT-HW-Telugu** — [CVIT IIIT Hyderabad](http://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data)

| Split | Images |
|---|---|
| Train | 80,693 |
| Val | 20,048 |
| Test | 17,910 |
| **Total** | **118,651** |

**Annotation file format** (`labels.txt`, one line per sample):
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
characters. This project patches the `maxpool` layer and `layer3/4` to 
preserve width resolution, giving a configurable sequence length (default S = 64, 
ablation S = 128).

```
Input  [B, 1, 64, 512]
  → Channel Replication (1→3)  preserve ImageNet pretraining
  → ResNet-18 backbone         stride-patched to preserve width
  → feature map [B, 512, 1, 64 or 128]
  → squeeze height, Conv1D(512→256)
  → Positional encoding
  → Encoder memory [B, 64/128, 256]
```

### Autoregressive Model (AR)
The AR model follows a state-of-the-art TrOCR-style architecture:
1. **Transformer Encoder (3 layers):** Adds global visual context to the CNN features before decoding.
2. **Transformer Decoder (6 layers, $d_{\text{model}}$=384):** Predicts character by character, attending to the encoder memory.

### CTC Baseline

```
Encoder memory [B, 64, 256]
  → 2-layer BiLSTM (hidden=256, bidirectional)
  → Linear(512 → vocab_size)
  → CTC loss / greedy decode
```

### Telugu-Aware Constraint Mask

Built **purely from observed label bigrams** in the training set —
no hard-coded linguistic rules.

Algorithm:
1. Scan every training label → extract every `(prev_char, next_char)` pair
2. `valid_next[prev][next] = True` only if that pair was seen in training data
3. During AR decoding: apply a soft penalty to logits of blocked next tokens before argmax
4. Run `validate_against_split(val_ann)` before training → violation rate must be ~0%

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| CNN backbone | ResNet-18 pretrained | Stronger features, stable training |
| Sequence model | CNN + Transformer hybrid | Better than pure ViT on limited HTR data |
| Stride fix | layer3/layer4 → (1,1) | S=64/128 instead of S=32; safe CTC margin |
| Script rules | Data-driven from labels | Never blocks a real training transition |
| Evaluation Metric | NFC Normalized CER | Ensures visually identical representations are scored fairly |
| Baselines | CTC + plain AR | Cleanly isolates novelty |
| Checkpoints | Unbiased Unconstrained CER | Ensures objective model selection |

---

## Checkpoint System

Exactly **3 files on disk at all times**, no matter how many epochs run:

```
checkpoints/<run>/
    best.pt        ← best val CER ever
    current.pt     ← most recent completed epoch
    previous.pt    ← epoch before that
```

Algorithm each epoch end:
1. `current.pt` → renamed to `previous.pt`
2. New state saved as `current.pt`
3. If val CER improved → `current.pt` copied to `best.pt`

Resume from any point:
```bash
python -m src.train_ctc --config configs/ctc_config.yaml --resume checkpoints/ctc/current.pt
python -m src.train_ar  --config configs/ar_config.yaml  --resume checkpoints/ar/current.pt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step-by-Step: How to Run

### Step 1 — Build vocabulary from training data

```bash
python -m src.vocab \
    data/raw/train/labels.txt \
    checkpoints/vocab.pkl \
    data/raw/val/labels.txt
```

This will:
- Scan all training labels → collect observed Telugu characters
- Build character vocabulary (expect ~80–100 tokens)
- Build **data-driven transition validity matrix** from observed bigrams
- Print a per-category transition audit table
- Validate the matrix against the val split → must show ~0% violation rate
- Save vocab to `checkpoints/vocab.pkl`

**Read the audit output before training.** If violation rate > 0.1%, the
constraint matrix will hurt AR decoding on that fraction of samples.

---

### Step 2 — Train CTC baseline (~1 hour on RTX 3090 Ti)

```bash
python -m src.train_ctc --config configs/ctc_config.yaml
```

Monitor live:
```bash
tensorboard --logdir logs/ctc
```

Logs per epoch: `train/loss`, `train/lr`, `val/loss`, `val/CER`,
`val/WER`, `val/avg_pred_len`.

---

### Step 3 — Train autoregressive model (~3 hours on RTX 3090 Ti)

```bash
python -m src.train_ar --config configs/ar_config.yaml
```

Resume after any interruption:
```bash
python -m src.train_ar --config configs/ar_config.yaml --resume checkpoints/ar/current.pt
```

---

### Step 4 — Run the 4-row ablation on test set

```bash
# Run A — CTC baseline
python -m src.evaluate \
    --model_type ctc \
    --checkpoint checkpoints/ctc/best.pt \
    --config configs/ctc_config.yaml \
    --split test

# Run B — AR decoder, no Telugu constraint
python -m src.evaluate \
    --model_type ar \
    --checkpoint checkpoints/ar/best.pt \
    --config configs/ar_config.yaml \
    --split test --no_constrain

# Run C — AR decoder + Telugu constraint (greedy)
python -m src.evaluate \
    --model_type ar \
    --checkpoint checkpoints/ar/best.pt \
    --config configs/ar_config.yaml \
    --split test

# Run D — AR decoder + Telugu constraint + beam search
python -m src.evaluate \
    --model_type ar \
    --checkpoint checkpoints/ar/best.pt \
    --config configs/ar_config.yaml \
    --split test --beam --beam_size 5
```

---

## Ablation Table (expected format)

| Model | Telugu Constraint | Beam | CER | WER | Compound CER | Simple CER |
|---|---|---|---|---|---|---|
| CTC Baseline | — | greedy | — | — | — | — |
| AR Decoder | ✗ | greedy | — | — | — | — |
| AR + Constraint | ✓ | greedy | — | — | — | — |
| AR + Constraint | ✓ | beam=5 | — | — | — | — |

Primary metric: **CER** (Character Error Rate).
Secondary: **WER**, compound CER (Virama words), simple CER.

---

## Hyperparameters

| Parameter | CTC | AR |
|---|---|---|
| Image H × W | 64 × 512 | 64 × 512 |
| Batch size | 64 | 64 |
| Optimizer | AdamW | AdamW |
| Learning rate | 1e-3 | 3e-4 |
| LR schedule | OneCycleLR | Warmup + Cosine |
| Warmup steps | — | 4000 |
| Weight decay | 1e-4 | 1e-4 |
| Label smoothing | — | 0.05 |
| Gradient clip | 5.0 | 5.0 |
| Mixed precision | fp16 | fp16 |
| Max epochs | 50 | 80 |
| Encoder layers | — | 3 |
| Decoder layers | — | 6 |
| Attention heads | — | 8 |
| $d_{\text{model}}$ | 256 | 384 |
| $d_{\text{ff}}$ | — | 1536 |
| Dropout | 0.2 | 0.15 |

---

## Hardware

| Component | Spec |
|---|---|
| GPU | RTX 3090 Ti (24 GB VRAM) |
| System RAM | 4 GB |

> **Note on system RAM**: `num_workers` is set to 2 (not 4) in both
> configs to avoid RAM pressure with 4 GB system memory.
> Drop to `num_workers: 0` if you still see OOM errors during data loading.

Estimated training times:

| Run | Duration |
|---|---|
| CTC (50 epochs) | ~3.5 hours |
| AR v2 (80 epochs) | ~5.4 hours |
| Full ablation eval | ~30–40 min |
| **Total** | **~4.5 hours** |

---

## File Reference

| File | Purpose |
|---|---|
| `src/vocab.py` | Telugu Unicode vocab, data-driven transition matrix, audit + validation |
| `src/transforms.py` | Grayscale, resize H=64, pad W=512, augmentation |
| `src/dataset.py` | Dataset class, `build_dataloader()` factory |
| `src/checkpoint_manager.py` | Rolling 2-slot: best / current / previous |
| `src/models/cnn_encoder.py` | ResNet-18, stride patch, positional encoding |
| `src/models/ctc_model.py` | CTC model, greedy decode |
| `src/models/ar_model.py` | AR model, greedy decode, beam search |
| `src/decoding/telugu_mask.py` | Constraint mask, vectorised apply, stats |
| `src/train_ctc.py` | CTC training loop |
| `src/train_ar.py` | AR training loop |
| `src/evaluate.py` | CER, WER, virama breakdown, confusion matrix, ablation table |
| `configs/ctc_config.yaml` | CTC hyperparameters |
| `configs/ar_config.yaml` | AR hyperparameters |

```
