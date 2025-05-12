FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    fuse \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://rclone.org/install.sh | bash

COPY . .
COPY ../train/train.py "CheXspert MLOps/train/train.py"

RUN pip install -r requirements.txt || true

RUN mkdir -p /mnt/chexspert

CMD ["python3", "run_training.py"]
