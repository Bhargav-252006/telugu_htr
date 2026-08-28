"""
Generate ablation comparison bar chart from the actual all_results.json.
Can be run locally (only needs matplotlib + the JSON results file).

Usage:
    python scripts/plot_ablation_chart.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

results_path = os.path.join(os.path.dirname(__file__), "..", "results", "paper_figures", "all_results.json")
out_path = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "ablation_comparison.png")

with open(results_path) as f:
    data = json.load(f)

names = list(data.keys())
cers = [data[n]["cer"] * 100 for n in names]
ci_lo = [data[n]["cer_ci"][0] * 100 for n in names]
ci_hi = [data[n]["cer_ci"][1] * 100 for n in names]
errors = [[c - lo for c, lo in zip(cers, ci_lo)],
          [hi - c for c, hi in zip(cers, ci_hi)]]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(names))
colors = ["#4CAF50"] * (len(names) - 1) + ["#2196F3"]
bars = ax.barh(x, cers, xerr=errors, capsize=4, color=colors,
               edgecolor="white", linewidth=0.5)
ax.set_yticks(x)
ax.set_yticklabels([n.replace("AR v2 ", "AR ") for n in names], fontsize=10)
ax.set_xlabel("CER (%)")
ax.set_title("Test-Set CER by Model Configuration (with 95% CI)")
ax.invert_yaxis()

for bar, cer in zip(bars, cers):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{cer:.2f}%", va="center", fontsize=9)

fig.tight_layout()
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved {out_path}")
