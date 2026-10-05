# NagarSeva Machine Learning Service
## Edge Computer Vision for Municipal Grievance Verification & Classification

This service provides an independent, custom-trained Deep Learning Computer Vision pipeline for municipal grievance verification, designed to run locally, on edge devices, or as a microservice alongside the NagarSeva platform.

---

## 1. System Architecture: The Two-Tier Vision Model

In municipal grievance platforms, relying exclusively on cloud LLM APIs poses latency, cost, and availability concerns. NagarSeva solves this with an industry-standard **Hierarchical Two-Tier AI Pipeline**:

```
           [ Citizen Uploads Grievance Photo ]
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER-1: Custom Edge Computer Vision Model (MobileNetV2)     │
│ ─────────────────────────────────────────────────────────── │
│  • Latency: < 20ms | Cost: $0.00 (Runs locally / CPU)       │
│  • Task 1: Defect Verification (Civic Issue vs. Spam)       │
│  • Task 2: Multi-class Category Classification              │
│  • Action: Reject selfies/memes/indoor photos instantly     │
└──────────────────────────┬──────────────────────────────────┘
                           │ If Verified (Confidence >= 50%)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ TIER-2: Cognitive Multimodal Reasoning (Gemini Vision)      │
│ ─────────────────────────────────────────────────────────── │
│  • Nuanced defect severity rating (Low / Medium / Critical) │
│  • Visual OCR: Lamppost numbers, street signs               │
│  • Municipal department routing recommendation              │
│  • Natural language feedback & automated citizen notes      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset Blueprint & Sourcing

The training data is partitioned into a strict **70% / 15% / 15%** split across 6 mutually exclusive classes:

```
dataset/
├── train/  (70% of dataset - used for gradient updates with augmentations)
│   ├── Road_Damage/        (Potholes, asphalt cracks, eroded surfacing)
│   ├── Illegal_Dumping/    (Garbage heaps, open trash, overflowing bins)
│   ├── Drainage/           (Clogged drains, sewage overflow, urban waterlogging)
│   ├── Streetlight/        (Damaged poles, broken fixtures, dangling wires)
│   ├── Encroachment/       (Footpath blockages, unauthorized hawkers)
│   └── Non_Civic_Spam/     (Negative class: selfies, pets, memes, indoor items)
├── val/    (15% of dataset - used for early stopping and model checkpointing)
└── test/   (15% of dataset - held-out strictly for unbiased final evaluation)
```

### Public Datasets & Provenance (Citations for Supervisor)
1. **Road Damage**: Sourced from the benchmark **Road Damage Dataset (RDD2022 / RDD2020)** published by Sekilab, containing annotated road surface distress across India and international road networks.
2. **Illegal Dumping / Waste**: Sourced from **TACO (Trash Annotations in Context)** and **TrashNet / Kaggle Waste Classification**.
3. **Drainage & Waterlogging**: Urban flood and drainage datasets from Mendeley Data and Kaggle Municipal Open Data.
4. **Non-Civic / Spam (Negative Class)**: Subsets of **ImageNet-1K / COCO** containing domestic interiors, animal portraits, and social media media to ensure high recall against spam.

---

## 3. Data Augmentation Pipeline

Citizen smartphone photos are inherently variable (shaky hands, motion blur, bad lighting, varying angles). To ensure high generalization, the training pipeline applies:
- `RandomResizedCrop(224, scale=(0.8, 1.0))`
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(degrees=15)`
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`
- ImageNet Normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)

---

## 4. Model Architecture & Hyperparameters

- **Base Architecture**: MobileNetV2 (Pretrained on ImageNet-1K).
- **Core Advantages**:
  - Employs **Inverted Residuals** and **Depthwise Separable Convolutions**.
  - Extremely compact: ~3.4 million parameters (~9.1 MB model checkpoint).
  - High inference throughput on CPU without requiring an NVIDIA GPU.
- **Classification Head**:
  - `nn.Dropout(p=0.3)`
  - `nn.Linear(in_features=1280, out_features=6)`
- **Optimizer**: `AdamW` (learning rate $\eta = 10^{-3}$, weight decay $\lambda = 10^{-4}$)
- **Scheduler**: `CosineAnnealingLR`

---

## 5. Quickstart & Commands

### 1. View or Inspect Dataset
```bash
python dataset_manager.py
```

### 2. Generate Dataset Overview & Visual Distribution Charts
```bash
python visualize_dataset.py
```
Outputs saved in `artifacts/`:
- `artifacts/dataset_overview.png` (Grid of sample images per class)
- `artifacts/class_distribution.png` (Bar chart of Train / Val / Test distribution)

### 3. Train the Model
```bash
python train.py
```
Outputs saved in `artifacts/`:
- `artifacts/best_grievance_model.pth` (Trained model weights)
- `artifacts/loss_accuracy_curve.png` (Epoch loss and accuracy curves)
- `artifacts/confusion_matrix.png` (Confusion matrix across classes)
- `artifacts/classification_report.txt` (Precision, Recall, and F1-score)
- `artifacts/metrics_summary.json` (Structured evaluation metrics)

### 4. Run Photo Verification / Inference on Any Image
```bash
python infer.py path/to/image.jpg
```
Sample JSON Output:
```json
{
  "is_civic_issue": true,
  "verification_status": "VERIFIED",
  "predicted_category": "Road Damage",
  "confidence": 0.9999,
  "reason": "Verified as genuine civic issue under category 'Road Damage' with 100.0% confidence.",
  "class_probabilities": {
    "Drainage": 0.0,
    "Encroachment": 0.0001,
    "Illegal_Dumping": 0.0,
    "Non_Civic_Spam": 0.0,
    "Road_Damage": 0.9999,
    "Streetlight": 0.0
  }
}
```

---

## 6. How to Present this to Your Supervisor

When presenting to your project supervisor or external examiners:

1. **Show the Dataset**:
   - Open `dataset/` to show the clean folder hierarchy (`train`, `val`, `test`).
   - Open `artifacts/dataset_overview.png` and `artifacts/class_distribution.png` to show the balanced training classes and representative civic defects.
2. **Show the Training Curves & Confusion Matrix**:
   - Open `artifacts/loss_accuracy_curve.png` to explain the smooth convergence of loss and accuracy.
   - Open `artifacts/confusion_matrix.png` to demonstrate strong diagonal classification and zero confusion between civic issues and spam.
   - Open `artifacts/classification_report.txt` to show macro F1-score, precision, and recall.
3. **Show Live Prediction**:
   - Run `python infer.py dataset/test/Road_Damage/road_damage_test_001.jpg` in the terminal to demonstrate live real-time inference with sub-second response.
4. **Explain the Two-Tier Architecture**:
   - Explain that your custom ML model acts as the **Tier-1 local gatekeeper** (free, offline, ultra-fast), while Gemini acts as the **Tier-2 cognitive multimodal agent** for complex contextual reasoning.
