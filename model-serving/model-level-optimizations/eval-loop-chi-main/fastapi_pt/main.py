from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import boto3
import uuid
from datetime import datetime
from mimetypes import guess_type
import os

app = FastAPI()

# -------------------
# Define your model
# -------------------
class CheXpertModel(nn.Module):
    def __init__(self, num_classes=14):
        super(CheXpertModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224 * 224 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 14
CLASS_NAMES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

model = CheXpertModel(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()
model.to(DEVICE)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------
# Connect to MinIO
# -------------------
s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_URL", "http://localhost:9000"),
    aws_access_key_id=os.getenv("MINIO_USER", "minioadmin"),
    aws_secret_access_key=os.getenv("MINIO_PASSWORD", "minioadmin"),
    region_name="us-east-1"
)

BUCKET_NAME = "production"

def upload_to_minio(img_bytes, predicted_class, confidence):
    prediction_id = str(uuid.uuid4())
    key = f"class_{predicted_class}/{prediction_id}.jpg"

    timestamp = datetime.utcnow().isoformat()
    content_type = guess_type("image.jpg")[0] or "application/octet-stream"

    # Upload image
    s3.upload_fileobj(
        img_bytes,
        Bucket=BUCKET_NAME,
        Key=key,
        ExtraArgs={"ContentType": content_type}
    )

    # Add tags
    s3.put_object_tagging(
        Bucket=BUCKET_NAME,
        Key=key,
        Tagging={"TagSet": [
            {"Key": "predicted_class", "Value": CLASS_NAMES[predicted_class]},
            {"Key": "confidence", "Value": f"{confidence:.3f}"},
            {"Key": "timestamp", "Value": timestamp}
        ]}
    )

    return key

# -------------------
# Inference Endpoint
# -------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).squeeze()
            predicted_index = torch.argmax(probs).item()
            confidence = probs[predicted_index].item()
            predicted_label = CLASS_NAMES[predicted_index]

        # Upload to MinIO
        upload_to_minio(io.BytesIO(contents), predicted_index, confidence)

        return JSONResponse(
            content={
                "predicted_class": predicted_label,
                "confidence": round(confidence, 3)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
