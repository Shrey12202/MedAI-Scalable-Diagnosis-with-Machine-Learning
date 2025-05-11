from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List
import os, io, csv
from PIL import Image
import torch
from torchvision import transforms
import mlflow.pytorch

# Constants
LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
]
CSV_PATH = "/mnt/persistent/simulation_data.csv"
IMG_DIR = "/mnt/object/processed/processed_images"

# Load model from MLflow Model Registry
MODEL_NAME = "CheXpertBestModel"
print(f"🔍 Loading model '{MODEL_NAME}' from MLflow Registry...")
MODEL = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/Production")
MODEL.eval()

# Image transformation
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# FastAPI init
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# -------------------- Prediction Endpoint --------------------

@app.post("/predict")
async def predict(
    patient_id: str = Form(...),
    image_name: str = Form(...),
    sex: str = Form(...),
    age: int = Form(...),
    view: str = Form(...),        # Frontal/Lateral
    projection: str = Form(...), # AP/PA
    file: UploadFile = File(...)
):
    try:
        # Save uploaded image
        img_path_local = os.path.join(IMG_DIR, f"patient{patient_id}")
        os.makedirs(img_path_local, exist_ok=True)
        full_img_path = os.path.join(img_path_local, image_name)
        image_bytes = await file.read()
        with open(full_img_path, "wb") as f:
            f.write(image_bytes)

        # Transform and predict
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = TRANSFORM(image).unsqueeze(0)
        with torch.no_grad():
            output = torch.sigmoid(MODEL(input_tensor))[0]
            preds = (output >= 0.5).int().numpy()

        # Prepare result row
        csv_path = f"CheXpert-v1.0/train/patient{patient_id}/{image_name}"
        row = {
            "Path": csv_path,
            "Sex": sex,
            "Age": age,
            "Frontal/Lateral": view,
            "AP/PA": projection,
            **dict(zip(LABELS, preds))
        }

        # Save to CSV
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"] + LABELS)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

        return {"prediction": row, "message": "Prediction saved. You can now correct if needed."}
    
    except Exception as e:
        return {"error": str(e)}

# -------------------- Correction UI --------------------

@app.get("/correct", response_class=HTMLResponse)
def correct_form(request: Request):
    return templates.TemplateResponse("correct.html", {"request": request, "labels": LABELS})

# -------------------- Correction Handler --------------------

@app.post("/correct")
async def submit_correction(
    patient_id: str = Form(...),
    image_name: str = Form(...),
    sex: str = Form(...),
    age: int = Form(...),
    view: str = Form(...),
    projection: str = Form(...),
    labels: List[str] = Form(...)  # multi-select list like ['Edema', 'Pneumonia']
):
    try:
        corrected = {label: 1 if label in labels else 0 for label in LABELS}
        corrected.update({
            "Path": f"CheXpert-v1.0/train/patient{patient_id}/{image_name}",
            "Sex": sex,
            "Age": age,
            "Frontal/Lateral": view,
            "AP/PA": projection
        })

        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"] + LABELS)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(corrected)

        return {"msg": "Correction saved.", "corrected_labels": corrected}
    
    except Exception as e:
        return {"error": str(e)}
