FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    python3-espeak \
    alsa-utils \
    libasound2 \
    libasound2-data \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create and set permissions for temp audio directory
RUN mkdir -p /app/temp_audio && chmod 777 /app/temp_audio

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variable for temp audio path
ENV TEMP_AUDIO_PATH=/app/temp_audio

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
