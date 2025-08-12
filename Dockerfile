# Use official Python 3.10 image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc libgl1-mesa-glx wget && rm -rf /var/lib/apt/lists/*

# Install pip and compatible NumPy first (fix for OpenCV/TensorFlow crash)
RUN pip install --upgrade pip && pip install numpy==1.26.4

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models using gdown
RUN gdown --id 1XHtMgrMMuE9R6lF3eeSS1JBATJy3gO1y -O tb_detection_model.h5
RUN gdown --id 1QwjZKcXZK5dtf52I2wGDxUMzyMByhTn5 -O stroke_detection_resnet50.h5

# Copy all project files
COPY . .

# Expose port
EXPOSE 10000

# Run the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
