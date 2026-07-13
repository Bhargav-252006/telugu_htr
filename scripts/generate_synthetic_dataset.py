"""
scripts/generate_synthetic_dataset.py

Generates a small synthetic Telugu handwriting dataset (images and labels)
so that the entire project pipeline (sanity checks, vocabulary construction,
and training scripts) can be run and tested immediately.
"""

import os
import random
from PIL import Image, ImageDraw

def generate_synthetic_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "raw")
    
    # Real or representative Telugu words
    telugu_words = [
        "కాలం", "పూజ", "తెలుగు", "భారతదేశం", "అమ్మ",
        "నాన్న", "విద్యా", "ప్రగతి", "సంస్కృతి", "విశ్వవిద్యాలయం",
        "సూర్యుడు", "చంద్రుడు", "నక్షత్రం", "సముద్రం", "పర్వతం",
        "పుస్తకం", "కలం", "కాగితం", "గణితం", "విజ్ఞానం",
        "గ్రామం", "శక్తి", "సంతోషం", "హృదయం", "పరిశోధన",
        "కార్యక్రమం", "అధికారి", "నాయకుడు", "సమాజం", "ప్రపంచం"
    ]
    
    splits = [
        ("train", 128),  # 128 samples to allow training a tiny batch
        ("val", 32),
        ("test", 32)
    ]
    
    print("Generating synthetic HTR dataset...")
    for split_name, count in splits:
        split_dir = os.path.join(data_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        labels_file = os.path.join(split_dir, "labels.txt")
        samples = []
        
        for idx in range(count):
            word = random.choice(telugu_words)
            img_name = f"word_{idx:05d}.png"
            img_path = os.path.join(split_dir, img_name)
            
            # Create a synthetic handwriting image (gray background, random lines/curves)
            img = Image.new("RGB", (256, 64), color=(245, 245, 245))
            draw = ImageDraw.Draw(img)
            
            # Draw fake letter strokes
            for _ in range(8):
                x1 = random.randint(10, 220)
                y1 = random.randint(10, 50)
                x2 = x1 + random.randint(15, 40)
                y2 = y1 + random.randint(-15, 15)
                # Draw lines or arcs simulating character contours
                draw.line([(x1, y1), (x2, y2)], fill=(40, 40, 40), width=random.randint(2, 4))
                # Add some small loops
                draw.ellipse([x1-3, y1-3, x1+3, y1+3], fill=None, outline=(40, 40, 40), width=2)
                
            img.save(img_path)
            # annotation format: <image_filename> <ground_truth_label>
            # The DataLoader expects the relative path matching the labels file location
            samples.append(f"{img_name} {word}")
            
        with open(labels_file, "w", encoding="utf-8") as f:
            f.write("\n".join(samples) + "\n")
            
        print(f"  Generated {count} samples in {split_dir}/")

    print("\nDataset generation completed successfully.")
    print("You can now run pre-training checks with:")
    print("  python scripts/sanity_check.py")

if __name__ == "__main__":
    generate_synthetic_data()
