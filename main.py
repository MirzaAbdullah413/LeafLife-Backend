import io
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
from model_utils import load_trained_model, predict_image
from PIL import Image
import logging
import torch
torch.set_num_threads(1)
logger = logging.getLogger("leaflife")
#1 Creating the app
app = FastAPI(title="LeafLife")

#Adding the middleware
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

#loaded the Model 
Model_Path = "leaf_disease_model.pth"
MAX_FILE_SIZE = 5 * 1024 * 1024
model, val_transformation, classes, device = load_trained_model(Model_Path)
print(f"Model loaded on {device} with {len(classes)} classes:{classes}")

#Checking whether the model is good
@app.get("/")
def health_check():
    return {"status":"ok", "num_classes":len(classes)}

#predicting the disease
@app.post("/predict")
def predict(file: UploadFile = File(...)):    #Content Validation 
    if file.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPG/PNG images are supported")

    try:
        contents = file.file.read(MAX_FILE_SIZE + 1)
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Image too large (max 5 MB)")
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        pil_image.thumbnail((640, 640))
    except Exception:
        logging.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Please try another image.")

    try:
        class_name, heatmap = predict_image(
            model, pil_image, val_transformation, classes, device
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Faile: {e}")

    return{
        "disease": class_name,
        "heatmap": f"data:image/jpeg;base64,{heatmap}"
    }
    
