from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

import torch
import torch.nn as nn
from torchvision import models, transforms

from PIL import Image
from datetime import datetime
import os
import io
import json
import boto3
from botocore.exceptions import NoCredentialsError

# 🚀 FastAPI app
app = FastAPI(title="CheXpert API", version="1.0")

# 📦 Disease classes
classes = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

# ☁️ MinIO S3 setup
s3 = boto3.client(
    's3',
    endpoint_url='http://host.docker.internal:9000',  # Use 'http://localhost:9000' outside Docker
    aws_access_key_id='your-access-key',
    aws_secret_access_key='your-secret-key',
    region_name='us-east-1'
)
BUCKET_NAME = 'production'

# 🧠 Define the model
class CheXpertModel(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        base = models.densenet121(pretrained=True)
        in_features = base.classifier.in_features
        base.classifier = nn.Linear(in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)

# 🔁 Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 🎯 Load model from disk
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CheXpertModel()
model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")
else:
    print(f"✅ Model file found at: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 🧪 Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 🖼️ Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        # 🔍 Predict
        with torch.no_grad():
            output = model(tensor)
            probs = torch.sigmoid(output).cpu().numpy()[0]
        predictions = {cls: float(round(prob, 4)) for cls, prob in zip(classes, probs)}

        # 📁 Save to MinIO
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_key = f"{timestamp}_{file.filename}"
        json_key = img_key.replace(".jpg", ".json").replace(".png", ".json")

        s3.put_object(Bucket=BUCKET_NAME, Key=img_key, Body=image_bytes)

        result = {
            "filename": img_key,
            "predictions": predictions,
            "timestamp": timestamp
        }
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=json_key,
            Body=json.dumps(result).encode("utf-8"),
            ContentType="application/json"
        )

        return result

    except NoCredentialsError:
        raise HTTPException(status_code=401, detail="❌ Invalid MinIO credentials")
    except Exception as e:
        print("🔥 Error:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

# 📊 Prometheus metrics
Instrumentator().instrument(app).expose(app)
