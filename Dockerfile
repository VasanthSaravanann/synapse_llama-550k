# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install xformers for CUDA 12.1 (as mentioned in the notebook)
RUN pip install -U xformers --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth and other dependencies
RUN pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install --no-deps trl peft accelerate bitsandbytes

# Copy the rest of the application
COPY . .

# Expose port (if needed for any web interface)
EXPOSE 8888

# Command to run the application
CMD ["/bin/bash"]