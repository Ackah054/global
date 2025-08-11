# Use official Python 3.10 image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies (added wget + unzip for downloading model)
RUN apt-get update && apt-get install -y gcc libgl1-mesa-glx wget unzip && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Download stroke model from Google Drive at build time
RUN wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1QwjZKcXZK5dtf52I2wGDxUMzyMByhTn5' -O stroke_detection_resnet50.h5

# Copy app files
COPY . .

# Expose port
EXPOSE 10000

# Start app using gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
