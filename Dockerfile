FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIPELINE_DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg tesseract-ocr libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./dir_language_pipeline/
RUN mkdir -p /data
EXPOSE 8000
CMD ["sh", "-c", "uvicorn dir_language_pipeline.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
