FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ansible \
    sshpass \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

CMD ["ansible-playbook", "-i", "inventory.ini", "site.yml"]
