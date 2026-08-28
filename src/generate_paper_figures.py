import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_ablation(results_file, output_path):
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    models = []
    cers = []
    wers = []
    
    for k, v in data.items():
        if k == '_metadata' or 'no CTC' in k:
            continue
        models.append(k)
        cers.append(v['cer'] * 100)
        wers.append(v['wer'] * 100)
        
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, cers, width, label='CER (%)', color='skyblue')
    rects2 = ax.bar(x + width/2, wers, width, label='WER (%)', color='salmon')
    
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Test Set Error Rates by Model Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

def plot_training_curves(log_json, output_path):
    with open(log_json, 'r') as f:
        data = json.load(f)
    
    if "epochs" not in data:
        print("No epoch data found in log")
        return
        
    epochs = []
    train_loss = []
    val_loss = []
    val_cer = []
    
    for ep in data["epochs"]:
        epochs.append(ep["epoch"])
        train_loss.append(ep["train_loss"])
        val_loss.append(ep["val_loss"])
        val_cer.append(ep["val_cer"] * 100)
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs, val_loss, label='Val Loss', color='orange', linewidth=2)
    ax1.set_ylabel('Loss (CTC+CE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.plot(epochs, val_cer, label='Val CER (%)', color='green', linewidth=2)
    
    # mark best epoch
    best_epoch_idx = np.argmin(val_cer)
    ax2.plot(epochs[best_epoch_idx], val_cer[best_epoch_idx], 'r*', markersize=15, label=f'Best ({val_cer[best_epoch_idx]:.2f}%)')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('CER (%)')
    ax2.set_title('Validation Character Error Rate')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == '__main__':
    os.makedirs('../paper/figures', exist_ok=True)
    plot_ablation('results/ablation_results.json', '../paper/figures/ablation_comparison_actual.png')
    
    # Hardcoding the path to the best AR training log
    log_path = '../logs/ar_v2/training_20260731_134445.json'
    if os.path.exists(log_path):
        plot_training_curves(log_path, '../paper/figures/training_curves_actual.png')
    
    print("Run this script to generate actual data plots from your project's JSON results.")
