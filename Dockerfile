FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libgl1-mesa-glx wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install gdown

RUN gdown https://drive.google.com/uc?id=1XHtMgrMMuE9R6lF3eeSS1JBATJy3gO1y -O tb_detection_model.h5
RUN gdown https://drive.google.com/uc?id=1QwjZKcXZK5dtf52I2wGDxUMzyMByhTn5 -O stroke_detection_resnet50.h5

COPY . .

EXPOSE 10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
