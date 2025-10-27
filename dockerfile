# File Name: Dockerfile
# Use a Python base image suitable for production
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# ----------------- 🛠️ FIX START -----------------
# 1. Update package lists and install the system dependency for pyttsx3
#    eSpeak is required by pyttsx3 on Linux environments (like Render).
RUN apt-get update -y && \
    apt-get install -y espeak && \
    rm -rf /var/lib/apt/lists/*
# ----------------- 🛠️ FIX END -----------------

# Copy and install dependencies first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Render automatically sets the PORT environment variable.
# Use $PORT or omit the port entirely; setting "80" is unnecessary and may be overridden.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
