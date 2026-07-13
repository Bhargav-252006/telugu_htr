"""
scripts/prepare_iiit_benchmark.py

Download and prepare the IIIT-INDIC-HW-WORDS Telugu split for benchmarking.

The IIIT-INDIC-HW-WORDS dataset is available at:
  https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data

This script converts the IIIT format to the format expected by our dataloader:
    <image_filename> <ground_truth_label>

Usage:
    1. Download the IIIT-INDIC-HW-WORDS dataset manually from CVIT website
    2. Extract the Telugu split into data/benchmark/iiit_raw/
    3. Run this script:
       python scripts/prepare_iiit_benchmark.py \
           --input_dir data/benchmark/iiit_raw \
           --output_dir data/benchmark/iiit_telugu

Structure of output:
    data/benchmark/iiit_telugu/
        images/         ← all word images
        labels.txt      ← <filename> <label> per line
"""

from __future__ import annotations
import argparse
import os
import shutil
import unicodedata
import glob
from pathlib import Path


def convert_iiit_format(input_dir: str, output_dir: str):
    """Convert IIIT-INDIC-HW-WORDS format to our labels.txt format."""
    img_out = os.path.join(output_dir, "images")
    os.makedirs(img_out, exist_ok=True)

    samples = []
    image_count = 0

    # IIIT format varies — try common structures
    # Pattern 1: input_dir/<word_id>/<image>.png with groundtruth.txt
    # Pattern 2: input_dir/images/ + input_dir/labels.txt
    # Pattern 3: input_dir/*.png with separate annotation file

    gt_file = None
    for candidate in ["groundtruth.txt", "labels.txt", "annotation.txt",
                       "gt.txt", "Telugu.txt"]:
        path = os.path.join(input_dir, candidate)
        if os.path.exists(path):
            gt_file = path
            break

    if gt_file:
        print(f"[prepare] Found annotation file: {gt_file}")
        with open(gt_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Try common formats
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    # Try tab-separated
                    parts = line.split("\t", maxsplit=1)
                if len(parts) != 2:
                    continue

                img_ref, label = parts
                label = unicodedata.normalize("NFC", label.strip())

                # Find the actual image file
                img_path = None
                for candidate_path in [
                    os.path.join(input_dir, img_ref),
                    os.path.join(input_dir, "images", img_ref),
                    os.path.join(input_dir, "Telugu", img_ref),
                ]:
                    if os.path.exists(candidate_path):
                        img_path = candidate_path
                        break

                if img_path is None:
                    continue

                # Copy to output
                out_name = f"bench_{image_count:05d}.png"
                shutil.copy2(img_path, os.path.join(img_out, out_name))
                samples.append(f"{out_name} {label}")
                image_count += 1
    else:
        # Try directory-based structure
        print("[prepare] No annotation file found, scanning directories...")
        for img_path in sorted(glob.glob(os.path.join(input_dir, "**", "*.png"),
                                          recursive=True)):
            # Try to find a corresponding .txt file with the label
            txt_path = img_path.replace(".png", ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, encoding="utf-8") as f:
                    label = unicodedata.normalize("NFC", f.read().strip())
                if label:
                    out_name = f"bench_{image_count:05d}.png"
                    shutil.copy2(img_path, os.path.join(img_out, out_name))
                    samples.append(f"{out_name} {label}")
                    image_count += 1

    if not samples:
        print("[ERROR] No samples found. Please check the input directory structure.")
        print("Expected: either a groundtruth.txt with '<filename> <label>' format,")
        print("or per-image .txt files alongside .png files.")
        return

    # Write labels file
    labels_path = os.path.join(output_dir, "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(samples) + "\n")

    print(f"\n[prepare] Successfully prepared {image_count} benchmark samples")
    print(f"  Images  → {img_out}")
    print(f"  Labels  → {labels_path}")
    print(f"\nRun evaluation with:")
    print(f"  python scripts/evaluate_benchmark.py \\")
    print(f"      --config configs/ar_config.yaml \\")
    print(f"      --checkpoint checkpoints/ar/best.pt \\")
    print(f"      --model_type ar \\")
    print(f"      --benchmark_annotation {labels_path} \\")
    print(f"      --benchmark_image_root {img_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare IIIT-INDIC-HW-WORDS Telugu split for benchmarking"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Path to extracted IIIT dataset Telugu split")
    parser.add_argument("--output_dir", default="data/benchmark/iiit_telugu",
                        help="Output directory for prepared benchmark")
    args = parser.parse_args()
    convert_iiit_format(args.input_dir, args.output_dir)
