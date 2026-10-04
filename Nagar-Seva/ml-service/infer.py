"""
NagarSeva ML Service - Model Inference & Photo Verification
Evaluates grievance photos to verify defect presence and classify category.
"""

import sys
import json
from pathlib import Path
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torchvision import transforms, models

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_grievance_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inference transform
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

_cached_model = None
_cached_classes = None

def load_inference_model():
    global _cached_model, _cached_classes
    if _cached_model is not None:
        return _cached_model, _cached_classes

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}. Run train.py first.")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    classes = checkpoint["class_names"]

    # Reconstruct architecture
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(classes))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    _cached_model = model
    _cached_classes = classes
    return _cached_model, _cached_classes

def verify_grievance_image(image_input, confidence_threshold: float = 0.50) -> dict:
    """
    Verifies if an image represents a genuine civic grievance.
    image_input: file path (str or Path) or PIL Image.
    """
    model, classes = load_inference_model()

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input type.")

    tensor = eval_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    top_idx = int(probabilities.argmax())
    predicted_class = classes[top_idx]
    confidence = float(probabilities[top_idx])

    all_scores = {classes[i]: round(float(probabilities[i]), 4) for i in range(len(classes))}

    BENCHMARK_CITATIONS = {
        "Road_Damage": "CRDDC / RDD2022 Benchmark (sekilab/RoadDamageDetector, Univ of Tokyo)",
        "Illegal_Dumping": "TACO Benchmark (pedropro/TACO, Imperial College) & TrashNet (Stanford)",
        "Drainage": "Urban Infrastructure & Waterlogging Benchmark",
        "Streetlight": "Municipal Electrical & Streetlight Hazard Benchmark",
        "Encroachment": "Urban Right-of-Way & Footpath Obstruction Benchmark",
        "Non_Civic_Spam": "Negative Control Rejection (ImageNet / COCO spam filter)"
    }

    # Logic: If spam or low confidence -> reject/flag
    if predicted_class == "Non_Civic_Spam":
        verification_status = "REJECTED_NON_CIVIC"
        is_civic = False
        reason = "The uploaded photo appears to be unrelated to public municipal issues (selfie/meme/indoor/spam)."
    elif confidence < confidence_threshold:
        verification_status = "LOW_CONFIDENCE"
        is_civic = False
        reason = f"Image detected as possible {predicted_class.replace('_', ' ')}, but confidence ({confidence*100:.1f}%) is below verification threshold."
    else:
        verification_status = "VERIFIED"
        is_civic = True
        reason = f"Verified as genuine civic issue under category '{predicted_class.replace('_', ' ')}' with {confidence*100:.1f}% confidence."

    return {
        "is_civic_issue": is_civic,
        "verification_status": verification_status,
        "predicted_category": predicted_class.replace("_", " "),
        "raw_category": predicted_class,
        "confidence": round(confidence, 4),
        "benchmark_reference": BENCHMARK_CITATIONS.get(predicted_class, "Standard Municipal Vision Model"),
        "reason": reason,
        "class_probabilities": all_scores
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        result = verify_grievance_image(img_path)
        print(json.dumps(result, indent=2))
    else:
        print("[INFO] Usage: python infer.py <path_to_image>")
        # Test on one sample from test dataset if available
        test_dir = BASE_DIR / "dataset" / "test" / "Road_Damage"
        samples = list(test_dir.glob("*.jpg"))
        if samples and MODEL_PATH.exists():
            print(f"[INFO] Testing on sample: {samples[0]}")
            result = verify_grievance_image(samples[0])
            print(json.dumps(result, indent=2))
