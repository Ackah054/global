# Use a lightweight Python 3.10 slim image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable stdout/stderr buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Force CPU-only mode for TensorFlow
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1 \
    libglib2.0-0 \
    wget \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first (to leverage Docker cache)
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only TensorFlow first
RUN pip install --no-cache-dir tensorflow-cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Upgrade gdown to latest version
RUN pip install --no-cache-dir --upgrade gdown

# Download TB detection model
RUN gdown https://drive.google.com/uc?id=1XHtMgrMMuE9R6lF3eeSS1JBATJy3gO1y -O tb_detection_model.h5

# Download stroke detection model
RUN gdown https://drive.google.com/uc?id=1QwjZKcXZK5dtf52I2wGDxUMzyMByhTn5 -O stroke_detection_resnet50.h5

# Copy rest of the app
COPY . .

# Expose port
EXPOSE 10000

# Run app with ONE worker to save memory
CMD exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300
