from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io, json, os
import boto3
from botocore.exceptions import NoCredentialsError
import mlflow


# 🚀 FastAPI App
app = FastAPI(
    title="Chexpert Classification API",
    description="Upload X-ray images to predict diseases using a pretrained model.",
    version="2.0.0"
)






# ✅ MinIO S3 client setup
s3 = boto3.client(
    's3',
    endpoint_url='http://129.114.27.181:9000',  # Changed from host.docker.internal to minio
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    region_name='us-east-1'
)
BUCKET_NAME = 'production'  # ✅ Ensure this exists in MinIO

# 🧠 Class labels
classes = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

# 🧱 Model definition
class CheXpertModel(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        base = models.densenet121(pretrained=True)
        in_features = base.classifier.in_features
        base.classifier = nn.Linear(in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)

# 📦 Load model
# 📦 Load model from MLflow
from mlflow.tracking import MlflowClient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    mlflow.set_tracking_uri("http://localhost:5000")  # Change if using remote MLflow

    client = MlflowClient()
    exp = client.get_experiment_by_name("chexpert-jupyter")

    if exp is None:
        raise ValueError("Experiment 'chexpert-jupyter' not found.")

    runs = client.search_runs(exp.experiment_id, filter_string="status = 'FINISHED'")
    if not runs:
        raise RuntimeError("No finished runs found in the 'chexpert-jupyter' experiment.")

    latest_run = sorted(runs, key=lambda r: r.info.start_time, reverse=True)[0]


    print(f"🔄 Loading model from run {latest_run.info.run_id}")
    model_uri = f"runs:/{latest_run.info.run_id}/final_model"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model.eval()

except Exception as e:
    print(f"❌ Failed to load model from MLflow: {e}")
    raise RuntimeError("🔥 Model loading from MLflow failed. Check MLflow tracking URI, run status, or model path.")


# 🔄 Preprocessing
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)





# 🔍 Predict from uploaded file

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        start_time = datetime.now()
        
        # Read uploaded file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Create unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        minio_filename = f"images/{timestamp}_{file.filename}"  # Added images/ prefix
        
        # Upload image to MinIO
        try:
            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=minio_filename,
                Body=image_bytes,
                ContentType='image/jpeg'
            )
            print(f"✅ Image uploaded to MinIO: {minio_filename}")
        except Exception as e:
            print(f"❌ Failed to upload image to MinIO: {str(e)}")
            raise

        # Model prediction code
        input_tensor = preprocess_image(image).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy().tolist()[0]

        # Create predictions dict
        predictions = {cls: round(prob, 4) for cls, prob in zip(classes, probs)}
        
        # Save prediction results to MinIO
        result = {
            "filename": minio_filename,
            "predictions": predictions,
            "timestamp": timestamp,
            "inference_time": (datetime.now() - start_time).total_seconds(),
            "model_device": str(device)
        }
        
        # Save results JSON
        results_filename = f"predictions/{timestamp}_{file.filename}.json"
        try:
            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=results_filename,
                Body=json.dumps(result).encode('utf-8'),
                ContentType='application/json'
            )
            print(f"✅ Predictions saved to MinIO: {results_filename}")
        except Exception as e:
            print(f"❌ Failed to save predictions to MinIO: {str(e)}")
            raise

        return JSONResponse(status_code=200, content=result)


    except Exception as e:
        print("🔥 Prediction error:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})






@app.post("/tag")
async def tag_image(
    filename: str = Form(..., description="Full image key like 'images/20250511_210212_view1_frontal.png'"),
    tag: str = Form(..., description="Tag to associate with the image")
):
    try:
        # ✅ Check if object exists in bucket
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=filename)
        if 'Contents' not in response or not any(obj['Key'] == filename for obj in response['Contents']):
            raise HTTPException(status_code=404, detail="Image not found in MinIO")

        # ✅ Add tag directly to the image
        s3.put_object_tagging(
            Bucket=BUCKET_NAME,
            Key=filename,
            Tagging={
                'TagSet': [
                    {'Key': 'label', 'Value': tag},
                    {'Key': 'timestamp', 'Value': datetime.now().strftime("%Y%m%d_%H%M%S")}
                ]
            }
        )

        print(f"🏷️ Tag '{tag}' attached to image '{filename}' in MinIO.")
        return {
            "message": "Tag added successfully",
            "filename": filename,
            "tag": tag
        }

    except Exception as e:
        print("🔥 Tagging error:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/health")
def health_check():
    return {"status": "ok"}


# 📊 Prometheus metrics
Instrumentator().instrument(app).expose(app)
