import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split, Dataset
import pandas as pd
from PIL import Image
from torchvision import models
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from prometheus_client import start_http_server, Gauge, Summary, CollectorRegistry, Gauge, push_to_gateway

# Set remote MLflow and MinIO endpoints
MLFLOW_TRACKING_URI = "http://129.114.26.3:8000"
MINIO_ENDPOINT = "http://129.114.26.3:9001"

# 🔹 Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("chexpert-jupyter")

# 🔹 Set environment variables for MLflow and MinIO
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["AWS_ACCESS_KEY_ID"] = "your-access-key"         # Use actual access key
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret-key"     # Use actual secret key

# 🔹 Prometheus metrics (avoid duplicate registration)
registry = CollectorRegistry()
train_loss_metric = Gauge('train_loss', 'Training loss per epoch', registry=registry)
val_loss_metric = Gauge('val_loss', 'Validation loss per epoch', registry=registry)
training_duration = Gauge('training_duration', 'Training time in seconds', registry=registry)

os.environ["MINIO_ACCESS_KEY_ID"] = "minioadmin"
os.environ["MINIO_SECRET_ACCESS_KEY"] = "minioadmin"


# -----------------------
# Config
# -----------------------
BATCH_SIZE = 128
EPOCHS = 2
LEARNING_RATE = 1e-4
NUM_CLASSES = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
IMG_ROOT = "/mnt/object/processed/processed_images" # Current dir includes CheXpert-v1.0/

LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
]

# -----------------------
# Log Metrics
# -----------------------
def log_metrics_to_mlflow_and_prometheus(epoch, stage, metrics_dict):
    # Log to MLflow
    for key, value in metrics_dict.items():
        mlflow.log_metric(f"{stage}_{key}", value, step=epoch)

    # Log to Prometheus PushGateway
    registry = CollectorRegistry()
    for key, value in metrics_dict.items():
        g = Gauge(f"{stage}_{key}", f"{stage.capitalize()} {key}", registry=registry)
        g.set(value)

    push_to_gateway("129.114.26.3:9091", job="chexpert_training", registry=registry)

# -----------------------
# Dataset
# -----------------------

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Store image paths and labels
        self.image_paths = self.df["Path"].apply(lambda x: x.replace("CheXpert-v1.0/train/", "")).tolist()
        self.labels = self.df[LABELS].fillna(0).replace(-1, 0).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]  # e.g., "patient123/study1/view1_frontal.jpg"
        img_path = os.path.join(self.root_dir, rel_path).replace(".jpg", ".png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, torch.tensor(label, dtype=torch.float32)

# -----------------------
# Model
# -----------------------
class CheXpertModel(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        base = models.densenet121(pretrained=True)
        in_features = base.classifier.in_features
        base.classifier = nn.Linear(in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)
    
# -----------------------
# Transforms
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------
# Load Data
# -----------------------
train_ds = CheXpertDataset(TRAIN_CSV, IMG_ROOT, transform)
test_ds = CheXpertDataset(TEST_CSV, IMG_ROOT, transform)

train_size = int(0.95 * len(train_ds))
val_size = len(train_ds) - train_size
train_subset, val_subset = random_split(train_ds, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, num_workers=6)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=6)

# -----------------------
# Training Loop
# -----------------------
with training_duration.time():
    with mlflow.start_run():
        # 🔹 Log hyperparameters
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)

        client = MlflowClient()
        model_name = "CheXpertBestModel"
        model = CheXpertModel(NUM_CLASSES).to(DEVICE)

        try:
        # Check if a version of the model exists in MLflow
            latest_versions = client.get_latest_versions(model_name)
            if latest_versions:
                # Get the latest model version URI
                latest_model_uri = f"models:/{model_name}/{latest_versions[0].version}"
                print(f"🔄 Loading model weights from {latest_model_uri}")
                model = mlflow.pytorch.load_model(latest_model_uri).to(DEVICE)
            else:
                print("⚠️ No previous model version found. Starting fresh.")
        except RestException as e:
            print(f"⚠️ MLflow model {model_name} not found. Starting fresh.")

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

        best_val_loss = float("inf")
        best_model_path = "best_model.pth"

        for epoch in range(EPOCHS):
            # ---- Training Phase ----
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"🔁 Epoch {epoch+1}", unit="batch")

            train_preds = []
            train_targets = []

            for imgs, labels in progress_bar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                preds = torch.sigmoid(outputs).detach().cpu().numpy() >= 0.5
                train_preds.extend(preds)
                train_targets.extend(labels.cpu().numpy())

                progress_bar.set_postfix(loss=loss.item())

            avg_train_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")
            
            train_preds = np.array(train_preds)
            train_targets = np.array(train_targets)
            train_accuracy = (train_preds == train_targets).mean()
            train_precision = precision_score(train_targets, train_preds, average='micro', zero_division=0)
            train_recall = recall_score(train_targets, train_preds, average='micro', zero_division=0)
            train_f1 = f1_score(train_targets, train_preds, average='micro', zero_division=0)

            log_metrics_to_mlflow_and_prometheus(epoch, "train", {
                "loss": avg_train_loss,
                "accuracy": train_accuracy,
                "precision": train_precision,
                "recall": train_recall,
                "f1": train_f1
            })

            train_loss_metric.set(avg_train_loss)


            scheduler.step(avg_train_loss)

            # ---- Validation Phase ----
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            val_preds = []
            val_targets = []    

            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc=f"🔁 Epoch {epoch+1}", unit="batch")
    
                for imgs, labels in progress_bar:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    preds = torch.sigmoid(outputs).detach().cpu().numpy() >= 0.5
                    val_preds.extend(preds)
                    val_targets.extend(labels.cpu().numpy())
                    
                    correct += (preds == labels.cpu().numpy()).sum().item()
                    total += preds.size
                    progress_bar.set_postfix(loss=loss.item())

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)
            val_accuracy = (val_preds == val_targets).mean()
            val_precision = precision_score(val_targets, val_preds, average='micro', zero_division=0)
            val_recall = recall_score(val_targets, val_preds, average='micro', zero_division=0)
            val_f1 = f1_score(val_targets, val_preds, average='micro', zero_division=0)

            log_metrics_to_mlflow_and_prometheus(epoch, "val", {
                "loss": avg_val_loss,
                "accuracy": val_accuracy,
                "precision": val_precision,
                "recall": val_recall,
                "f1": val_f1
            })
                        
            val_loss_metric.set(avg_val_loss)


            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ Saved best model at epoch {epoch+1} (Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.4f})")

        # 🔹 Log final artifacts
        # Load best weights back into the model
        model.load_state_dict(torch.load(best_model_path))
        
        # Log best model in MLflow native format
        mlflow.pytorch.log_model(model, "best_model")
        
        # Optional: Register it in the Model Registry
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/best_model",
            name="CheXpertBestModel"
        )

# -----------------------
# Evaluation
# -----------------------
model.load_state_dict(torch.load(best_model_path))
model.eval()
model.eval()
test_loss = 0
test_preds = []
test_targets = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        preds = torch.sigmoid(outputs).detach().cpu().numpy() >= 0.5
        test_preds.extend(preds)
        test_targets.extend(labels.cpu().numpy())

test_preds = np.array(test_preds)
test_targets = np.array(test_targets)
test_accuracy = (test_preds == test_targets).mean()
test_precision = precision_score(test_targets, test_preds, average='micro', zero_division=0)
test_recall = recall_score(test_targets, test_preds, average='micro', zero_division=0)
test_f1 = f1_score(test_targets, test_preds, average='micro', zero_division=0)
avg_test_loss = test_loss / len(test_loader)

log_metrics_to_mlflow_and_prometheus(0, "test", {
    "loss": avg_test_loss,
    "accuracy": test_accuracy,
    "precision": test_precision,
    "recall": test_recall,
    "f1": test_f1
})

print(f"\n Test Accuracy (avg over all labels): {test_accuracy:.4f}")

mlflow.end_run()
