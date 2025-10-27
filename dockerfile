# Use a Python base image suitable for production
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install system dependencies including espeak and sound libraries
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    espeak \
    espeak-data \
    libespeak1 \
    libpulse0 \
    libasound2 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create directory for temporary audio files
RUN mkdir -p /app/temp_audio && \
    chmod 777 /app/temp_audio

# Environment variable for temp audio path
ENV TEMP_AUDIO_PATH=/app/temp_audio

# Render automatically sets the PORT environment variable
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
