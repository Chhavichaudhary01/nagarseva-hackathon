"""
NagarSeva ML Service - REST API (FastAPI)
Exposes the custom-trained Computer Vision model as an HTTP microservice.
"""

import sys
import base64
import io
from pathlib import Path
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from infer import verify_grievance_image, load_inference_model

app = FastAPI(
    title="NagarSeva Civic ML Verification API",
    description="Tier-1 Edge Computer Vision Model for Municipal Grievance Verification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Base64VerifyRequest(BaseModel):
    photoData: str
    expectedCategory: str = None
    confidenceThreshold: float = 0.50

@app.on_event("startup")
def startup_event():
    # Preload model into memory
    load_inference_model()
    print("[INFO] Civic ML Model loaded into memory successfully.")

@app.get("/api/ml/health")
def health_check():
    return {
        "status": "ONLINE",
        "service": "NagarSeva Civic Vision Service",
        "model": "MobileNetV2-Civic-Grievance-v1",
        "num_classes": 6
    }

@app.post("/api/ml/verify")
def verify_photo_base64(request: Base64VerifyRequest):
    try:
        raw_b64 = request.photoData
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        
        img_bytes = base64.b64decode(raw_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        result = verify_grievance_image(pil_img, confidence_threshold=request.confidenceThreshold)
        
        # Check if user specified a category and if it matches
        category_matches = True
        if request.expectedCategory:
            norm_expected = request.expectedCategory.strip().lower().replace(" ", "_")
            norm_predicted = result["raw_category"].strip().lower()
            category_matches = (norm_expected == norm_predicted)
            result["category_match"] = category_matches
            
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

@app.post("/api/ml/verify-file")
async def verify_photo_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        result = verify_grievance_image(pil_img)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Default to port 8085 to avoid any collision with 8080, 5173, 5174, 3001
    uvicorn.run("app:app", host="0.0.0.0", port=8085, reload=False)
