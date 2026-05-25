# Use a lightweight CUDA-enabled Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for psycopg2 and networking
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
# Note: Keep the requirements separate from your scraper's requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the summarizer script
COPY summarizer.py .

# Create a non-root user for security
RUN useradd -m summarizer_user
USER summarizer_user

# The script will be triggered by the K8s CronJob
CMD ["python", "summarizer.py"]
