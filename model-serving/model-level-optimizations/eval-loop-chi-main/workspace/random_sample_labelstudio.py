# %%
import boto3
import requests
from datetime import datetime, timedelta

# %%

# --- 🔧 CONFIG ---

# MinIO S3 config
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "production"


# %%
# Label Studio config
LABEL_STUDIO_URL = "http://129.114.27.181:8080"  # replace with your public IP or domain
LABEL_STUDIO_TOKEN = "ab9927067c51ff279d340d7321e4890dc2841c4a"
PROJECT_ID = 1

# %%

# --- 🌐 GET PUBLIC IP OF INSTANCE (cloud only) ---
#public_ip = requests.get("http://localhost/latest/meta-data/public-ipv4").text.strip()
public_ip = "129.114.27.181"  # 👈 or use your real IP like "192.168.1.24"


# %%
# --- ☁️ SET UP S3 CLIENT ---
s3 = boto3.client(
    "s3",
    endpoint_url=f"http://{public_ip}:9000",  # public IP for signed URLs to work in browser
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    region_name="us-east-1"
)

# %%

# --- 📂 LIST FILES IN BUCKET ---
response = s3.list_objects_v2(Bucket=BUCKET_NAME)
objects = response.get("Contents", [])

# %%

# --- 🔗 CREATE SIGNED URL TASKS FOR LABEL STUDIO ---
signed_tasks = []
for obj in objects:
    key = obj["Key"]
    
    # Create signed URL valid for 7 days
    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": BUCKET_NAME, "Key": key},
        ExpiresIn=7 * 24 * 60 * 60  # 7 days
    )
    
    signed_tasks.append({
        "data": {
            "image": url
        }
    })


# %%
# --- 📨 SEND TASKS TO LABEL STUDIO ---
import_response = requests.post(
    f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/import",
    headers={"Authorization": f"Token {LABEL_STUDIO_TOKEN}"},
    json=signed_tasks
)

# %%
# --- ✅ RESULT ---
if import_response.status_code == 201:
    print(f"✅ Successfully uploaded {len(signed_tasks)} tasks to Label Studio project {PROJECT_ID}")
else:
    print(f"❌ Error uploading tasks:", import_response.text)


