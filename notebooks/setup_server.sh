#!/bin/bash
# ==============================================================
# Telugu HTR Project — Remote Server Setup Script
# ==============================================================
# Run this on your college SSH server:
#   bash setup_server.sh
#
# This creates the complete folder structure for training.
# After running, copy your source code into the project folder.
# ==============================================================

PROJECT_NAME="telugu_htr"
BASE_DIR="$HOME/$PROJECT_NAME"

echo "════════════════════════════════════════════════════════"
echo "  Telugu HTR — Server Setup"
echo "  Creating project at: $BASE_DIR"
echo "════════════════════════════════════════════════════════"

# ── Create main project structure ────────────────────────────
mkdir -p "$BASE_DIR"

# Source code
mkdir -p "$BASE_DIR/src/models"
mkdir -p "$BASE_DIR/src/decoding"

# Config files
mkdir -p "$BASE_DIR/configs"

# Scripts
mkdir -p "$BASE_DIR/scripts"

# Data directories
mkdir -p "$BASE_DIR/data/raw/train"
mkdir -p "$BASE_DIR/data/raw/val"
mkdir -p "$BASE_DIR/data/raw/test"
mkdir -p "$BASE_DIR/data/benchmark/iiit_telugu/images"

# Checkpoint directories (one per model variant)
mkdir -p "$BASE_DIR/checkpoints/ctc"
mkdir -p "$BASE_DIR/checkpoints/ar"
mkdir -p "$BASE_DIR/checkpoints/ar_no_ctc"

# TensorBoard log directories
mkdir -p "$BASE_DIR/logs/ctc"
mkdir -p "$BASE_DIR/logs/ar"
mkdir -p "$BASE_DIR/logs/ar_no_ctc"

# Results output
mkdir -p "$BASE_DIR/results"

echo ""
echo "✅ Folder structure created:"
echo ""
find "$BASE_DIR" -type d | head -30 | sed "s|$HOME/||" | while read dir; do
    depth=$(echo "$dir" | tr -cd '/' | wc -c)
    indent=$(printf '%*s' $((depth * 2)) '')
    basename=$(basename "$dir")
    echo "  ${indent}📁 ${basename}/"
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Next steps:"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  1. Copy your code to the server:"
echo "     scp -r src/ configs/ scripts/ requirements.txt user@server:~/$PROJECT_NAME/"
echo ""
echo "  2. Place your dataset:"
echo "     - Train images → ~/$PROJECT_NAME/data/raw/train/"
echo "     - Train labels → ~/$PROJECT_NAME/data/raw/train/labels.txt"
echo "     - Val images   → ~/$PROJECT_NAME/data/raw/val/"
echo "     - Val labels   → ~/$PROJECT_NAME/data/raw/val/labels.txt"
echo "     - Test images  → ~/$PROJECT_NAME/data/raw/test/"
echo "     - Test labels  → ~/$PROJECT_NAME/data/raw/test/labels.txt"
echo ""
echo "  3. Install dependencies:"
echo "     cd ~/$PROJECT_NAME && pip install -r requirements.txt"
echo ""
echo "  4. Check GPU:"
echo "     python -c \"import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')\""
echo ""
echo "  5. Start training:"
echo "     cd ~/$PROJECT_NAME"
echo "     python -m src.train_ctc --config configs/ctc_config.yaml"
echo "     python -m src.train_ar --config configs/ar_config.yaml"
echo ""
echo "════════════════════════════════════════════════════════"
