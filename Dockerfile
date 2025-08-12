# Use a lightweight Python 3.10 slim image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable stdout/stderr buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed for the app, including 'libglib2.0-0' to avoid some GUI errors,
# and clean apt cache to reduce image size
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt to working directory
COPY requirements.txt .

# Upgrade pip to latest version
RUN pip install --upgrade pip

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Upgrade gdown to latest version to avoid issues with Google Drive downloads
RUN pip install --upgrade gdown

# Download TB detection model file from Google Drive using gdown
RUN gdown https://drive.google.com/uc?id=1XHtMgrMMuE9R6lF3eeSS1JBATJy3gO1y -O tb_detection_model.h5

# Download stroke detection model file from Google Drive using gdown
RUN gdown https://drive.google.com/uc?id=1QwjZKcXZK5dtf52I2wGDxUMzyMByhTn5 -O stroke_detection_resnet50.h5

# Copy entire project files into the container
COPY . .

# Expose port 10000 for the app to listen on
EXPOSE 10000

# Run the app with Gunicorn using 2 workers to reduce memory load, bind to all interfaces on port 10000
CMD exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
