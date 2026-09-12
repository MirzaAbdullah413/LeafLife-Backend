import io
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model_utils import load_trained_model, predict_image
from PIL import Image
#1 Creating the app
app = FastAPI(title="LeafLife")

#Adding the middleware
app.add_middleware(
    CORSMiddleware,
    allow_origin=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_header=["*"],
)

#loaded the Model 
Model_Path = "leaf_disease_model.pth"
model, val_transformation, classes, device = load_trained_model(Model_Path)
print(f"Model loaded on {device} with {len(classes)} classes:{classes}")

#Checking whether the model is good
@app.get("/")
def health_check():
    return {"status":"ok", "num_classes":len(classes)}

#predicting the disease
@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    #Content Validation 
    if file.content_type not in ("image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPG/PNG images are supported")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGD")
    except Exception:
        raise HTTPException(status_code = 400, detail="Could not read the File")

    try:
        class_name, heatmap = predict_image(
            model, pil_image, val_transformation, classes, device
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Faile: {e}")

    return{
        "disease": class_name,
        "heatmap": f"data:image/png;base64,{heatmap}"
    }
    
