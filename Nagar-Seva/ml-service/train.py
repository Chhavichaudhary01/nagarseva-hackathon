"""
NagarSeva ML Service - Model Training Pipeline
Uses Transfer Learning (MobileNetV2) to train a Civic Grievance Verification & Classification Model.
Computes Loss/Accuracy, Precision, Recall, Macro F1, and Confusion Matrix.
"""

import sys
import os
import json
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
NUM_CLASSES = 6
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_transforms():
    """
    Standard ML Featurization:
    - Train set: aggressive augmentation to simulate shaky phone shots & diverse lighting.
    - Val & Test sets: deterministic resizing and normalization only.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    }
    return data_transforms

def build_model(num_classes: int = NUM_CLASSES):
    """
    MobileNetV2: lightweight, optimized for edge/mobile inference with high accuracy.
    """
    try:
        # Attempt to load pretrained weights
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        print("[INFO] Loaded MobileNetV2 with ImageNet pretrained weights.")
    except Exception as e:
        print(f"[WARN] Loading pretrained weights failed ({e}), initializing standard MobileNetV2.")
        model = models.mobilenet_v2(weights=None)

    # Freeze early feature extraction layers if desired, or fine-tune all
    # Here we fine-tune the final classification head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

def plot_curves(history: dict, save_path: Path):
    """Plots training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Curve
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()
    
    # Accuracy Curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Accuracy (%)")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "g--s", label="Val Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved learning curves to: {save_path}")

def plot_confusion_matrix(cm, class_names, save_path: Path):
    """Plots confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Grievance Verification Confusion Matrix",
        ylabel="True Civic Class",
        xlabel="Predicted Civic Class"
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrix to: {save_path}")

def train():
    print("=" * 60)
    print("NAGARSEVA CIVIC GRIEVANCE MODEL TRAINING")
    print(f"Device: {DEVICE} | Epochs: {NUM_EPOCHS} | Batch Size: {BATCH_SIZE}")
    print("=" * 60)

    transforms_dict = get_data_transforms()
    
    # Load Datasets
    image_datasets = {
        x: datasets.ImageFolder(DATASET_DIR / x, transforms_dict[x])
        for x in ["train", "val", "test"]
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == "train"), num_workers=0)
        for x in ["train", "val", "test"]
    }
    
    class_names = image_datasets["train"].classes
    print(f"[INFO] Classes ({len(class_names)}): {class_names}")
    print(f"[INFO] Dataset sizes: Train={len(image_datasets['train'])}, Val={len(image_datasets['val'])}, Test={len(image_datasets['test'])}")

    model = build_model(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = ARTIFACTS_DIR / "best_grievance_model.pth"

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders["train"]:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()
        epoch_train_loss = running_loss / len(image_datasets["train"])
        epoch_train_acc = running_corrects.double().item() / len(image_datasets["train"])

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(image_datasets["val"])
        epoch_val_acc = val_corrects.double().item() / len(image_datasets["val"])

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.1f}%")

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": best_val_acc,
                "epoch": epoch + 1
            }, best_model_path)

    elapsed = time.time() - start_time
    print(f"\n[INFO] Training complete in {elapsed:.1f}s. Best Val Acc: {best_val_acc*100:.1f}%")
    print(f"[INFO] Best model checkpoint saved to: {best_model_path}")

    # Plot curves
    plot_curves(history, ARTIFACTS_DIR / "loss_accuracy_curve.png")

    # Evaluation on Held-Out Test Set
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST DATASET")
    print("=" * 60)

    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, ARTIFACTS_DIR / "confusion_matrix.png")

    report_str = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    print("\nClassification Report:")
    print(report_str)

    # Save metrics report
    metrics_summary = {
        "classes": class_names,
        "best_val_acc": best_val_acc,
        "test_macro_f1": report_dict.get("macro avg", {}).get("f1-score", 0.0),
        "test_accuracy": report_dict.get("accuracy", 0.0),
        "classification_report": report_dict
    }
    with open(ARTIFACTS_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    with open(ARTIFACTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_str)

    print(f"[INFO] Saved test evaluation report to: {ARTIFACTS_DIR / 'classification_report.txt'}")
    print("=" * 60)
    print("ML Pipeline execution successfully finished!")

if __name__ == "__main__":
    train()
