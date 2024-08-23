# Base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building packages
RUN apt-get update && \
    apt-get install -y \
    swig \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements into the container
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip==24.2 \
    && pip install -r requirements.txt

# Copy the train.py script into the container
COPY train.py .

# Run the train.py script
CMD ["python", "train.py"]