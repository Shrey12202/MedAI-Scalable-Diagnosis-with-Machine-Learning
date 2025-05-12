FROM python:3.10-slim

WORKDIR /app

# Install packages
RUN apt-get update && apt-get install -y \
    curl \
    fuse \
    unzip \
    ssh \
    && rm -rf /var/lib/apt/lists/*

# Install rclone
RUN curl https://rclone.org/install.sh | bash

# Create mount point
RUN mkdir -p /mnt/chexspert

# Copy training script and runner
COPY run_training.py .
COPY ../train/train.py CheXspert_MLOps/train/train.py

# Optional: Install Python dependencies if needed
RUN pip install -r requirements.txt || true

# Set the container entrypoint to run the full pipeline
CMD ["python3", "run_training.py"]
