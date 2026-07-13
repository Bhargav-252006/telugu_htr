#!/usr/bin/env python3
"""
Generate a publication-quality architecture diagram for the Telugu HTR IEEE paper.
Uses matplotlib with patches and annotations — no external dependencies needed.

Usage:
    python scripts/generate_architecture_diagram.py

Output:
    paper/figures/architecture_diagram.pdf
    paper/figures/architecture_diagram.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# ── Configuration ────────────────────────────────────────────────
COLORS = {
    "input":       "#E3F2FD",   # very light blue
    "encoder":     "#BBDEFB",   # light blue
    "ctc_path":    "#C8E6C9",   # light green
    "ar_path":     "#FFE0B2",   # light orange
    "loss":        "#FFCDD2",   # light red
    "mask":        "#F3E5F5",   # light purple
    "output":      "#F5F5F5",   # light gray
    "border":      "#333333",
    "arrow":       "#555555",
    "text":        "#1A1A1A",
}

FONT = {"family": "sans-serif", "size": 8}
matplotlib.rc("font", **FONT)


def draw_box(ax, x, y, w, h, label, color, fontsize=7, bold=False,
             linestyle="-", linewidth=1.2, sublabel=None):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor=COLORS["border"],
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2 + (0.02 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=COLORS["text"], zorder=3)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.04, sublabel,
                ha="center", va="center", fontsize=5.5,
                fontstyle="italic", color="#666666", zorder=3)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color=None, style="-|>", lw=1.2,
               linestyle="-", connectionstyle="arc3,rad=0"):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    if color is None:
        color = COLORS["arrow"]
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=color,
        linewidth=lw,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        mutation_scale=12,
        zorder=1,
    )
    ax.add_patch(arrow)
    return arrow


def main():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    ax.set_xlim(-0.05, 2.55)
    ax.set_ylim(-0.15, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Title ──────────────────────────────────────────────────────
    ax.text(1.25, 1.20, "Telugu HTR System Architecture",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=COLORS["text"])

    # ════════════════════════════════════════════════════════════════
    # Column 1: Input
    # ════════════════════════════════════════════════════════════════
    draw_box(ax, 0.0, 0.42, 0.22, 0.18, "Input Image", COLORS["input"],
             fontsize=7, bold=True, sublabel="B×1×64×512")

    # ════════════════════════════════════════════════════════════════
    # Column 2: Shared CNN Encoder
    # ════════════════════════════════════════════════════════════════
    # Outer box
    enc_x, enc_y, enc_w, enc_h = 0.30, 0.22, 0.40, 0.58
    outer = FancyBboxPatch(
        (enc_x, enc_y), enc_w, enc_h,
        boxstyle="round,pad=0.02",
        facecolor=COLORS["encoder"],
        edgecolor=COLORS["border"],
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(outer)
    ax.text(enc_x + enc_w / 2, enc_y + enc_h - 0.03,
            "ResNet-18 CNN Encoder", ha="center", va="center",
            fontsize=8, fontweight="bold", color=COLORS["text"], zorder=3)

    # Sub-blocks inside encoder
    sb_w, sb_h = 0.34, 0.08
    sb_x = enc_x + 0.03
    
    draw_box(ax, sb_x, 0.62, sb_w, sb_h, "Conv1 + MaxPool",
             "#E3F2FD", fontsize=6)
    draw_box(ax, sb_x, 0.52, sb_w, sb_h, "ResNet Blocks 1–4",
             "#E3F2FD", fontsize=6)
    draw_box(ax, sb_x, 0.42, sb_w, sb_h, "1×1 Conv → d=256",
             "#E3F2FD", fontsize=6)
    draw_box(ax, sb_x, 0.32, sb_w, sb_h, "Positional Encoding",
             "#E3F2FD", fontsize=6)

    # Internal arrows
    for y_start in [0.62, 0.52, 0.42]:
        draw_arrow(ax, sb_x + sb_w / 2, y_start,
                   sb_x + sb_w / 2, y_start - 0.02,
                   lw=0.8)

    # Output label
    ax.text(enc_x + enc_w / 2, enc_y - 0.03, "(B×64×256)",
            ha="center", va="center", fontsize=6, fontstyle="italic",
            color="#666666")

    # Arrow: Input → Encoder
    draw_arrow(ax, 0.22, 0.51, 0.30, 0.51)

    # ════════════════════════════════════════════════════════════════
    # Column 3: Two decoder pathways
    # ════════════════════════════════════════════════════════════════

    # ── PATH A: CTC Baseline (top) ─────────────────────────────────
    path_a_y = 0.85
    ax.text(0.95, 1.05, "Path A: CTC Baseline",
            ha="center", va="center", fontsize=7, fontweight="bold",
            color="#2E7D32", zorder=3)

    draw_box(ax, 0.78, path_a_y, 0.22, 0.14, "BiLSTM", COLORS["ctc_path"],
             fontsize=7, bold=True, sublabel="2 layers, h=256")

    draw_box(ax, 1.06, path_a_y, 0.20, 0.14, "Linear Proj", COLORS["ctc_path"],
             fontsize=7, sublabel="→ |V|=91")

    draw_box(ax, 1.32, path_a_y, 0.22, 0.14, "CTC Decode", COLORS["ctc_path"],
             fontsize=7, bold=True, sublabel="greedy")

    # Arrows for path A
    draw_arrow(ax, 1.00, path_a_y + 0.07, 1.06, path_a_y + 0.07)
    draw_arrow(ax, 1.26, path_a_y + 0.07, 1.32, path_a_y + 0.07)

    # ── PATH B: AR Transformer (bottom) ─────────────────────────────
    ax.text(1.20, 0.68, "Path B: Autoregressive Transformer",
            ha="center", va="center", fontsize=7, fontweight="bold",
            color="#E65100", zorder=3)

    # Transformer Encoder
    draw_box(ax, 0.78, 0.48, 0.24, 0.14, "Transformer Enc", COLORS["ar_path"],
             fontsize=7, bold=True, sublabel="2L, 4 heads")

    # Auxiliary CTC Head (dashed)
    draw_box(ax, 0.82, 0.74, 0.18, 0.08, "Aux CTC Head", COLORS["loss"],
             fontsize=6, linestyle="--", linewidth=0.8)
    # Arrow from Transformer Enc to Aux CTC
    draw_arrow(ax, 0.90, 0.62, 0.90, 0.74,
               linestyle="--", lw=0.8, color="#E57373")
    ax.text(0.93, 0.69, "λ=0.3", fontsize=5, color="#C62828", fontstyle="italic")

    # Transformer Decoder
    draw_box(ax, 1.10, 0.48, 0.24, 0.14, "Transformer Dec", COLORS["ar_path"],
             fontsize=7, bold=True, sublabel="4L, 4 heads")
    
    # Arrow Enc → Dec
    draw_arrow(ax, 1.02, 0.55, 1.10, 0.55)
    ax.text(1.06, 0.58, "cross-\nattn", fontsize=4.5, ha="center",
            color="#E65100", fontstyle="italic")

    # Telugu Grammar Mask
    draw_box(ax, 1.40, 0.48, 0.22, 0.14, "Telugu Grammar", COLORS["mask"],
             fontsize=7, bold=True, sublabel="Bigram Mask",
             linestyle="--", linewidth=1.0)
    draw_arrow(ax, 1.34, 0.55, 1.40, 0.55)

    # Beam Search
    draw_box(ax, 1.68, 0.48, 0.22, 0.14, "Beam Search", COLORS["output"],
             fontsize=7, bold=True, sublabel="k=5")
    draw_arrow(ax, 1.62, 0.55, 1.68, 0.55)

    # ── Shared Encoder → Both Paths ─────────────────────────────────
    # Arrow from encoder to path A
    draw_arrow(ax, 0.50, 0.80, 0.50, 0.92,
               lw=1.0, color=COLORS["arrow"])
    draw_arrow(ax, 0.50, 0.92, 0.78, 0.92,
               lw=1.0, color=COLORS["arrow"])

    # Arrow from encoder to path B
    draw_arrow(ax, 0.70, 0.55, 0.78, 0.55,
               lw=1.0, color=COLORS["arrow"])

    # ── Outputs ──────────────────────────────────────────────────────
    # CTC output
    draw_box(ax, 1.60, path_a_y, 0.26, 0.14, "Output Text", COLORS["output"],
             fontsize=7, bold=True, sublabel="CER: 3.91%")
    draw_arrow(ax, 1.54, path_a_y + 0.07, 1.60, path_a_y + 0.07)

    # AR output
    draw_box(ax, 1.96, 0.48, 0.26, 0.14, "Output Text", COLORS["output"],
             fontsize=7, bold=True, sublabel="CER: 4.85%")
    draw_arrow(ax, 1.90, 0.55, 1.96, 0.55)

    # ── Joint Loss Box ──────────────────────────────────────────────
    draw_box(ax, 0.78, 0.18, 0.56, 0.12,
             "Joint Loss:  L = (1−λ)·L_CE + λ·L_CTC   (λ=0.3)",
             COLORS["loss"], fontsize=6.5, bold=True, linestyle="-")

    # Dashed arrows from decoder and aux CTC to loss
    draw_arrow(ax, 1.22, 0.48, 1.10, 0.30,
               linestyle="--", lw=0.8, color="#E57373")
    draw_arrow(ax, 0.90, 0.74, 0.95, 0.30,
               linestyle="--", lw=0.8, color="#E57373")

    # ── Target tokens input to decoder ───────────────────────────────
    draw_box(ax, 1.12, 0.32, 0.20, 0.10, "Target Tokens", "#FFF9C4",
             fontsize=6, sublabel="(teacher forcing)")
    draw_arrow(ax, 1.22, 0.42, 1.22, 0.48,
               lw=0.8, color="#F57F17")

    # ── Legend ───────────────────────────────────────────────────────
    legend_x = 2.05
    legend_y = 0.10
    legend_items = [
        ("Shared Encoder", COLORS["encoder"]),
        ("CTC Path", COLORS["ctc_path"]),
        ("AR Path", COLORS["ar_path"]),
        ("Loss", COLORS["loss"]),
        ("Constraint", COLORS["mask"]),
    ]
    for i, (label, color) in enumerate(legend_items):
        y = legend_y + i * 0.06
        rect = FancyBboxPatch(
            (legend_x, y), 0.08, 0.04,
            boxstyle="round,pad=0.005",
            facecolor=color,
            edgecolor=COLORS["border"],
            linewidth=0.6,
        )
        ax.add_patch(rect)
        ax.text(legend_x + 0.10, y + 0.02, label,
                fontsize=5.5, va="center", color=COLORS["text"])

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs("paper/figures", exist_ok=True)
    out_pdf = "paper/figures/architecture_diagram.pdf"
    out_png = "paper/figures/architecture_diagram.png"

    plt.tight_layout(pad=0.1)
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"[OK] Architecture diagram saved: {out_pdf}")
    print(f"[OK] Architecture diagram saved: {out_png}")


if __name__ == "__main__":
    main()
