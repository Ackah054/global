# Use a lightweight Python 3.10 slim image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable stdout/stderr buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Disable GPU usage to avoid CUDA errors on Render
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed for the app
# Replaced libgl1-mesa-glx with libgl1 (Debian Trixie no longer has libgl1-mesa-glx)
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1 \
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

# Expose port 10000 (optional, Render mainly uses $PORT env variable)
EXPOSE 10000

# Run the app with Gunicorn using 2 workers, binding to the port Render provides via $PORT
CMD exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
