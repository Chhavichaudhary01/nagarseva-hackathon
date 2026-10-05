"""
NagarSeva ML Service - Dataset Visualizer
Generates presentation-ready visual artifacts of the training dataset.
Shows sample image grid per class and train/val/test distribution bar charts.
"""

import sys
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

from dataset_manager import CIVIC_CLASSES

def plot_sample_grid():
    """Generates a grid of sample images across all categories."""
    num_classes = len(CIVIC_CLASSES)
    samples_per_class = 3
    
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(9, 2.5 * num_classes))
    fig.suptitle("NagarSeva Grievance Dataset - Training Sample Overview", fontsize=14, fontweight="bold", y=0.995)

    for row_idx, class_name in enumerate(CIVIC_CLASSES):
        class_folder = DATASET_DIR / "train" / class_name
        images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
        
        for col_idx in range(samples_per_class):
            ax = axes[row_idx, col_idx]
            if col_idx < len(images):
                img = Image.open(images[col_idx])
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "No Image", ha="center", va="center")
            
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(class_name.replace("_", "\n"), fontsize=9, fontweight="bold", rotation=0, labelpad=40, va="center")

    plt.tight_layout()
    output_path = ARTIFACTS_DIR / "dataset_overview.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved dataset overview grid to: {output_path}")

def plot_distribution():
    """Generates a grouped bar chart of train/val/test counts per category."""
    splits = ["train", "val", "test"]
    stats = {c: {s: 0 for s in splits} for c in CIVIC_CLASSES}

    for split in splits:
        for c in CIVIC_CLASSES:
            folder = DATASET_DIR / split / c
            if folder.exists():
                stats[c][split] = len(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))

    x = np.arange(len(CIVIC_CLASSES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    
    train_counts = [stats[c]["train"] for c in CIVIC_CLASSES]
    val_counts = [stats[c]["val"] for c in CIVIC_CLASSES]
    test_counts = [stats[c]["test"] for c in CIVIC_CLASSES]

    r1 = ax.bar(x - width, train_counts, width, label="Train (70%)", color="#2563eb")
    r2 = ax.bar(x, val_counts, width, label="Val (15%)", color="#10b981")
    r3 = ax.bar(x + width, test_counts, width, label="Test (15%)", color="#f59e0b")

    ax.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
    ax.set_title("NagarSeva Grievance Dataset Class Distribution", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in CIVIC_CLASSES], rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Label on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f"{height}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    autolabel(r1)
    autolabel(r2)
    autolabel(r3)

    plt.tight_layout()
    output_path = ARTIFACTS_DIR / "class_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved class distribution chart to: {output_path}")

if __name__ == "__main__":
    plot_sample_grid()
    plot_distribution()
