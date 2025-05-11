from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os, io, csv
from PIL import Image
import torch
from torchvision import transforms
import mlflow.pytorch

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Constants
LABELS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
          "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
          "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"]

CSV_PATH = "/mnt/persistent/simulation_data.csv"
IMG_DIR = "/mnt/object/processed/processed_images"
MODEL = mlflow.pytorch.load_model("models:/CheXpertBestModel/latest")
MODEL.eval()

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

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
    # Save image
    img_path_local = os.path.join(IMG_DIR, f"patient{patient_id}")
    os.makedirs(img_path_local, exist_ok=True)
    full_img_path = os.path.join(img_path_local, image_name)
    image_bytes = await file.read()
    with open(full_img_path, "wb") as f:
        f.write(image_bytes)

    # Predict
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        output = torch.sigmoid(MODEL(input_tensor))[0]
        preds = (output >= 0.5).int().numpy()

    # Format path for CSV
    csv_path = f"CheXpert-v1.0/train/patient{patient_id}/{image_name}"

    # Save row to CSV
    row = {
        "Path": csv_path,
        "Sex": sex,
        "Age": age,
        "Frontal/Lateral": view,
        "AP/PA": projection
    }
    row.update(dict(zip(LABELS, preds)))  # 0 or 1

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"] + LABELS)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)

    return {"prediction": row, "message": "Prediction saved. You can now correct if needed."}

@app.get("/correct", response_class=HTMLResponse)
def correct_form(request: Request):
    return templates.TemplateResponse("correct.html", {"request": request, "labels": LABELS})

@app.post("/correct")
async def submit_correction(
    patient_id: str = Form(...),
    image_name: str = Form(...),
    sex: str = Form(...),
    age: int = Form(...),
    view: str = Form(...),
    projection: str = Form(...),
    labels: List[str] = Form(...)  # multi-select: ['Edema', 'Pneumonia', ...]
):
    corrected = {label: 1 if label in labels else 0 for label in LABELS}
    corrected["Path"] = f"CheXpert-v1.0/train/patient{patient_id}/{image_name}"
    corrected["Sex"] = sex
    corrected["Age"] = age
    corrected["Frontal/Lateral"] = view
    corrected["AP/PA"] = projection

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"] + LABELS)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(corrected)

    return {"msg": "Correction saved.", "corrected_labels": corrected}
