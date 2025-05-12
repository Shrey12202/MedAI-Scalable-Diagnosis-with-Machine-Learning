import os
import subprocess
import time

# Step 1: Install rclone
def install_rclone():
    print("🔧 Installing rclone...")
    subprocess.run("curl https://rclone.org/install.sh | bash", shell=True, check=True)

# Step 2: Setup rclone config
def setup_rclone_config():
    print("🔐 Writing rclone config...")
    os.makedirs("/root/.config/rclone", exist_ok=True)
    with open("/root/.config/rclone/rclone.conf", "w") as f:
        f.write("""[chi_tacc]
type = s3
provider = Other
access_key_id = <your-access-key>
secret_access_key = <your-secret-key>
endpoint = https://objects.tacc.chameleoncloud.org
""")

# Step 3: Mount object store
def mount_bucket():
    print("📦 Mounting Chameleon bucket...")
    os.makedirs("/mnt/chexspert", exist_ok=True)
    subprocess.Popen([
        "rclone", "mount", "chi_tacc:object-persist-project44", "/mnt/chexspert",
        "--daemon", "--allow-other", "--vfs-cache-mode", "writes"
    ])
    time.sleep(5)  # wait for mount to complete

# Step 4: Run training
def run_training():
    print("🚀 Starting training...")
    subprocess.run([
        "python3", "CheXspert MLOps/train/train.py",
        "--csv_path", "/mnt/chexspert/train.csv"
    ], check=True)

if __name__ == "__main__":
    install_rclone()
    setup_rclone_config()
    mount_bucket()
    run_training()
