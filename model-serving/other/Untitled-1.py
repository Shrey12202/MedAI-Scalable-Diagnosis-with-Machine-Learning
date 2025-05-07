# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import mlflow
from prometheus_client import start_http_server, Summary, Gauge
from git import Repo
import ray

# Ray check
ray.init(ignore_reinit_error=True)
assert ray.is_initialized(), "Ray is not properly initialized"


# %%
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_CLASSES = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CSV = "sample_train.csv"
TEST_CSV = "sample_test.csv"
IMG_ROOT = "."

LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
]


# %%
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.df[LABELS] = self.df[LABELS].fillna(0).replace(-1, 0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Path"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[LABELS].values.astype("float32"))
        return image, labels


# %%
class CheXpertModel(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        base = models.densenet121(pretrained=True)
        in_features = base.classifier.in_features
        base.classifier = nn.Linear(in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


# %%
train_ds = CheXpertDataset(TRAIN_CSV, IMG_ROOT, transform)
test_ds = CheXpertDataset(TEST_CSV, IMG_ROOT, transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Prometheus Metrics
start_http_server(8001)
training_time = Summary('training_duration_seconds', 'Time spent training model')
train_loss_metric = Gauge('train_loss', 'Training Loss')


# %%
# Git repo settings
repo_path = "./chexpert_model_repo"
repo_url = "https://github.com/Shrey12202/MedAI-Scalable-Diagnosis-with-Machine-Learning"
model_file = "best_model.pth"
model_path = os.path.join(repo_path, model_file)

# Clone/pull latest model
if not os.path.exists(repo_path):
    Repo.clone_from(repo_url, repo_path)
else:
    repo = Repo(repo_path)
    repo.remotes.origin.pull()


# %%
@training_time.time()
def train_model():
    model = CheXpertModel(NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    best_loss = float('inf')

    with mlflow.start_run():
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")
            train_loss_metric.set(avg_loss)
            scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), model_path)
                print(f"✅ Saved new best model at epoch {epoch+1} (Loss: {avg_loss:.4f})")
                # Git commit + push
                repo.index.add([model_path])
                repo.index.commit(f"Best model at epoch {epoch+1}, loss: {avg_loss:.4f}")
                repo.remotes.origin.push()

        mlflow.pytorch.log_model(model, "model")

train_model()


# %%
model = CheXpertModel(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(model_path))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = torch.sigmoid(model(imgs)).cpu()
        all_preds.append(outputs)
        all_labels.append(labels)

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()
preds_binary = (all_preds >= 0.5).astype(int)

# Accuracy
correct = (preds_binary == all_labels).sum()
accuracy = correct / preds_binary.size
print(f"\n✅ Test Accuracy: {accuracy:.4f}")
mlflow.log_metric("test_accuracy", accuracy)

# Disease-level output
print("\n📊 Ground Truth vs Predictions:")
for i in range(min(6, len(all_preds))):
    pred_bin = preds_binary[i]
    gt_bin = all_labels[i].astype(int)

    pred_diseases = [LABELS[j] for j, v in enumerate(pred_bin) if v == 1]
    gt_diseases = [LABELS[j] for j, v in enumerate(gt_bin) if v == 1]

    print(f"\nImage {i+1}:")
    print("Expected (GT):", gt_diseases)
    print("Predicted     :", pred_diseases)



