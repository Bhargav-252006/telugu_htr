"""
Export per-sample predictions for both CTC and AR v2 models.
Run this ON THE SERVER where the dataset and checkpoints are available.

Usage:
    python scripts/export_predictions.py

Outputs:
    results/predictions_ctc.csv
    results/predictions_ar_v2_beam5.csv
"""
import os, sys, csv, unicodedata
import torch, yaml, editdistance
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vocab import TeluguVocab
from src.dataset import build_dataloader

force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
if device.type == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU memory: {allocated:.1f} GB used / {total:.1f} GB total", flush=True)
vocab = TeluguVocab.load("checkpoints/vocab.pkl")
os.makedirs("results", exist_ok=True)

def export_ctc():
    cfg = yaml.safe_load(open("configs/ctc_config.yaml"))
    loader = build_dataloader(
        "data/raw/test/labels.txt", ".", vocab,
        split="test", batch_size=32, num_workers=0,
        max_label_len=cfg["data"]["max_label_len"], add_sos_eos=False
    )
    from src.models.ctc_model import CTCModel
    mcfg = cfg["model"]
    model = CTCModel(vocab_size=len(vocab), d_model=mcfg["d_model"],
                     lstm_hidden=mcfg["lstm_hidden"], lstm_layers=mcfg["lstm_layers"],
                     dropout=mcfg["dropout"], pretrained=mcfg["pretrained"])
    ckpt = torch.load("checkpoints/ctc/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    rows = []
    with torch.no_grad():
        for images, labels, label_lens, image_widths in tqdm(loader, desc="CTC"):
            images = images.to(device)
            pred_ids = model.greedy_decode(images, input_widths=image_widths.to(device))
            for i, (pred, llen) in enumerate(zip(pred_ids, label_lens.tolist())):
                gt_ids = labels[i, :llen].tolist()
                pred_str = unicodedata.normalize("NFC", vocab.decode(pred))
                gt_str = unicodedata.normalize("NFC", vocab.decode(gt_ids))
                ed = editdistance.eval(list(pred_str), list(gt_str))
                rows.append({"ground_truth": gt_str, "prediction": pred_str,
                             "edit_distance": ed, "gt_length": len(gt_str)})

    with open("results/predictions_ctc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ground_truth","prediction","edit_distance","gt_length"])
        w.writeheader()
        w.writerows(rows)
    print(f"CTC: {len(rows)} samples -> results/predictions_ctc.csv")

def export_ar_v2():
    cfg = yaml.safe_load(open("configs/ar_v2_config.yaml"))
    loader = build_dataloader(
        "data/raw/test/labels.txt", ".", vocab,
        split="test", batch_size=32, num_workers=0,
        max_label_len=cfg["data"]["max_label_len"], add_sos_eos=True
    )
    from src.models.ar_model import ARModel
    mcfg = cfg["model"]
    model = ARModel(vocab_size=len(vocab), sos_id=vocab.sos_id, eos_id=vocab.eos_id,
                    d_model=mcfg["d_model"], nhead=mcfg["nhead"],
                    num_encoder_layers=mcfg.get("num_encoder_layers", 3),
                    num_decoder_layers=mcfg["num_decoder_layers"],
                    dim_feedforward=mcfg["dim_feedforward"],
                    dropout=mcfg["dropout"], max_label_len=mcfg["max_label_len"],
                    label_smoothing=mcfg["label_smoothing"],
                    pretrained=mcfg["pretrained"],
                    ctc_weight=mcfg.get("ctc_weight", 0.3),
                    high_res_temporal=mcfg.get("high_res_temporal", False))
    ckpt = torch.load("checkpoints/ar_v2/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    rows = []
    with torch.no_grad():
        for images, labels, label_lens, image_widths in tqdm(loader, desc="AR v2"):
            images = images.to(device)
            pred_ids = model.beam_decode(images, beam_size=5, max_len=36,
                                         vocab=None, constrain=False,
                                         input_widths=image_widths.to(device),
                                         length_penalty=0.6)
            for i, (pred, lab, llen) in enumerate(zip(pred_ids, labels, label_lens.tolist())):
                gt_ids = lab[1:llen-1].tolist()
                pred_str = unicodedata.normalize("NFC", vocab.decode(pred))
                gt_str = unicodedata.normalize("NFC", vocab.decode(gt_ids))
                ed = editdistance.eval(list(pred_str), list(gt_str))
                rows.append({"ground_truth": gt_str, "prediction": pred_str,
                             "edit_distance": ed, "gt_length": len(gt_str)})

    with open("results/predictions_ar_v2_beam5.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ground_truth","prediction","edit_distance","gt_length"])
        w.writeheader()
        w.writerows(rows)
    print(f"AR v2: {len(rows)} samples -> results/predictions_ar_v2_beam5.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ctc", "ar", "both"], default="both",
                        help="Which model to export: ctc, ar, or both (default: both)")
    args = parser.parse_args()

    if args.model in ("ctc", "both"):
        print("Exporting CTC predictions...")
        export_ctc()
    if args.model in ("ar", "both"):
        print("\nExporting AR v2 predictions...")
        export_ar_v2()
    print("\nDone!")
