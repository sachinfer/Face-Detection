# Use the official Python image as the base
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgtk2.0-dev \
    libcanberra-gtk3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libtbbmalloc2 \
    libeigen3-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (opencv and others)
RUN pip install --upgrade pip
RUN pip install opencv-python opencv-python-headless numpy

# Copy your Python script and model files into the container
COPY . /app

# Expose port for debugging or viewing results (optional)
EXPOSE 8080

# Set the entry point to the Python script (update this with the correct script name)
ENTRYPOINT ["python", "AgeGender.py"]
