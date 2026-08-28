"""
Generate prediction_examples figure for the paper.
Uses the CORRECT ValTransform preprocessing from src/transforms.py.
Loads ~12 test images and uses greedy decode — runs in ~1-2 min on CPU.

Run ON THE SERVER:
    FORCE_CPU=1 python scripts/generate_prediction_examples.py
"""
import os, sys, random, unicodedata
import torch, yaml
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vocab import TeluguVocab
from src.transforms import ValTransform

# ── Device ──
force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ── Vocab & Transform ──
vocab = TeluguVocab.load("checkpoints/vocab.pkl")
val_transform = ValTransform()  # correct: grayscale → aspect-ratio resize → pad → normalize

# ── Telugu font ──
telugu_font = None
for fp in [
    os.path.expanduser("~/.fonts/NotoSansTelugu-Regular.ttf"),
    "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf",
]:
    if os.path.exists(fp):
        telugu_font = fp
        break
if telugu_font is None:
    for f in fm.fontManager.ttflist:
        if "telugu" in f.name.lower():
            telugu_font = f.fname
            break

prop = fm.FontProperties(fname=telugu_font) if telugu_font else fm.FontProperties()
if not telugu_font:
    print("WARNING: No Telugu font found! Run: mkdir -p ~/.fonts && wget -O ~/.fonts/NotoSansTelugu-Regular.ttf 'https://github.com/google/fonts/raw/main/ofl/notosanstelugu/NotoSansTelugu%5Bwdth%2Cwght%5D.ttf' && fc-cache -fv")

# ── Load test labels ──
labels_dict = {}
with open("data/raw/test/labels.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            labels_dict[parts[0]] = unicodedata.normalize("NFC", parts[1])
print(f"Loaded {len(labels_dict)} test labels", flush=True)

# ── Select ~16 diverse images ──
random.seed(42)
all_items = [(p, gt) for p, gt in labels_dict.items() if os.path.exists(p)]
random.shuffle(all_items)

selected = []
short, medium, long_w = [], [], []
for path, gt in all_items:
    if len(gt) <= 4:
        short.append((path, gt))
    elif len(gt) <= 8:
        medium.append((path, gt))
    else:
        long_w.append((path, gt))

selected = short[:5] + medium[:6] + long_w[:5]
random.shuffle(selected)
print(f"Selected {len(selected)} test images", flush=True)

# ── Load models (pretrained=False since we load checkpoints) ──
print("Loading CTC model...", flush=True)
cfg_ctc = yaml.safe_load(open("configs/ctc_config.yaml"))
from src.models.ctc_model import CTCModel
mc = cfg_ctc["model"]
ctc_model = CTCModel(vocab_size=len(vocab), d_model=mc["d_model"],
                     lstm_hidden=mc["lstm_hidden"], lstm_layers=mc["lstm_layers"],
                     dropout=mc["dropout"], pretrained=False)
ckpt = torch.load("checkpoints/ctc/best.pt", map_location=device, weights_only=False)
ctc_model.load_state_dict(ckpt["model"])
ctc_model.to(device).eval()

print("Loading AR v2 model...", flush=True)
cfg_ar = yaml.safe_load(open("configs/ar_v2_config.yaml"))
from src.models.ar_model import ARModel
ma = cfg_ar["model"]
ar_model = ARModel(vocab_size=len(vocab), sos_id=vocab.sos_id, eos_id=vocab.eos_id,
                   d_model=ma["d_model"], nhead=ma["nhead"],
                   num_encoder_layers=ma.get("num_encoder_layers", 3),
                   num_decoder_layers=ma["num_decoder_layers"],
                   dim_feedforward=ma["dim_feedforward"],
                   dropout=ma["dropout"], max_label_len=ma["max_label_len"],
                   label_smoothing=ma["label_smoothing"], pretrained=False,
                   ctc_weight=ma.get("ctc_weight", 0.3),
                   high_res_temporal=ma.get("high_res_temporal", False))
ckpt = torch.load("checkpoints/ar_v2/best.pt", map_location=device, weights_only=False)
ar_model.load_state_dict(ckpt["model"])
ar_model.to(device).eval()

# ── Run predictions ──
print("Running predictions...", flush=True)
results = []
with torch.no_grad():
    for i, (img_path, gt) in enumerate(selected):
        img_pil = Image.open(img_path).convert("RGB")
        img_orig = Image.open(img_path).convert("L")  # for display

        # Use the CORRECT ValTransform
        img_tensor, img_width = val_transform(img_pil)
        img_tensor = img_tensor.unsqueeze(0).to(device)       # [1, 1, 64, 512]
        img_width_t = torch.tensor([img_width], dtype=torch.long).to(device)

        # CTC
        ctc_ids = ctc_model.greedy_decode(img_tensor, input_widths=img_width_t)
        ctc_pred = unicodedata.normalize("NFC", vocab.decode(ctc_ids[0]))

        # AR greedy
        ar_ids = ar_model.greedy_decode(img_tensor)
        ar_pred = unicodedata.normalize("NFC", vocab.decode(ar_ids[0]))

        ctc_ok = (ctc_pred == gt)
        ar_ok = (ar_pred == gt)
        results.append({"img": img_orig, "gt": gt, "ctc": ctc_pred, "ar": ar_pred,
                         "ctc_ok": ctc_ok, "ar_ok": ar_ok})
        print(f"  [{i+1}/{len(selected)}] GT='{gt}' | CTC='{ctc_pred}' {'OK' if ctc_ok else 'ERR'} | AR='{ar_pred}' {'OK' if ar_ok else 'ERR'}", flush=True)

# ── Pick best examples for the figure ──
both_ok = [r for r in results if r["ctc_ok"] and r["ar_ok"]]
ctc_err_ar_ok = [r for r in results if not r["ctc_ok"] and r["ar_ok"]]
ar_err_ctc_ok = [r for r in results if r["ctc_ok"] and not r["ar_ok"]]
both_err = [r for r in results if not r["ctc_ok"] and not r["ar_ok"]]

display = both_ok[:3] + ctc_err_ar_ok[:2] + ar_err_ctc_ok[:1] + both_err[:2]
if len(display) < 6:
    display += [r for r in results if r not in display][:6 - len(display)]

print(f"\nDisplaying {len(display)} examples in figure", flush=True)

# ── Plot as a table-like figure ──
n = len(display)
fig, axes = plt.subplots(n, 4, figsize=(14, 1.8 * n),
                         gridspec_kw={"width_ratios": [2, 1.5, 1.5, 1.5]})

# Header
headers = ["Input Image", "Ground Truth", "CTC Prediction", "AR Prediction"]
for j, h in enumerate(headers):
    axes[0, j].set_title(h, fontsize=12, fontweight="bold", pad=8)

for i, r in enumerate(display):
    # Col 0: Handwriting image
    axes[i, 0].imshow(r["img"], cmap="gray", aspect="auto")
    axes[i, 0].axis("off")

    # Col 1: Ground truth
    axes[i, 1].text(0.5, 0.5, r["gt"], transform=axes[i, 1].transAxes,
                    fontproperties=prop, fontsize=16, ha="center", va="center")
    axes[i, 1].axis("off")

    # Col 2: CTC prediction
    color = "#2E7D32" if r["ctc_ok"] else "#C62828"
    axes[i, 2].text(0.5, 0.5, r["ctc"], transform=axes[i, 2].transAxes,
                    fontproperties=prop, fontsize=16, ha="center", va="center", color=color)
    axes[i, 2].axis("off")

    # Col 3: AR prediction
    color = "#2E7D32" if r["ar_ok"] else "#C62828"
    axes[i, 3].text(0.5, 0.5, r["ar"], transform=axes[i, 3].transAxes,
                    fontproperties=prop, fontsize=16, ha="center", va="center", color=color)
    axes[i, 3].axis("off")

fig.tight_layout()

os.makedirs("results/paper_figures", exist_ok=True)
os.makedirs("paper/figures", exist_ok=True)
for path in ["results/paper_figures/prediction_examples.png", "paper/figures/prediction_examples.png"]:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

print("\nDone!")
