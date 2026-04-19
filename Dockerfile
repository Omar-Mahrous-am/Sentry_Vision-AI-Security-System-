# Base image with Python 3.11
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, PyTorch, and TTS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libespeak1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    streamlit \
    python-multipart \
    pyttsx3 \
    ultralytics \
    requests

# Copy project files
COPY . .

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Run backend
CMD ["uvicorn", "backend.system_backend:app", "--host", "0.0.0.0", "--port", "8000"]