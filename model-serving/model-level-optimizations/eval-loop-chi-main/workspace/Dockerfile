# Use a minimal Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependencies and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python script
COPY transfer.py .

# Command to run your script
CMD ["python", "user_feedback_labelstudio.py"]
