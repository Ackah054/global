# Use a lightweight Python 3.10 image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable stdout/stderr buffering
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed for your app and cleanup apt cache to reduce image size
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    wget \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements file to the working directory
COPY requirements.txt .

# Upgrade pip to the latest version and install all Python dependencies
RUN pip install --upgrade pip

# Install all Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Upgrade gdown to the latest version to avoid known bugs with Google Drive URLs
RUN pip install --upgrade gdown

# Download the TB detection model from Google Drive using gdown
RUN gdown https://drive.google.com/uc?id=1XHtMgrMMuE9R6lF3eeSS1JBATJy3gO1y -O tb_detection_model.h5

# Download the stroke detection model from Google Drive using gdown
RUN gdown https://drive.google.com/uc?id=1QwjZKcXZK5dtf52I2wGDxUMzyMByhTn5 -O stroke_detection_resnet50.h5

# Copy the entire project into the working directory
COPY . .

# Expose port 10000 for the application
EXPOSE 10000

# Start the app using gunicorn, binding to 0.0.0.0 on port 10000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
