# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Metadata
LABEL maintainer="your.email@university.edu"
LABEL description="RL-Based Adaptive Video Compression for Traffic Sign Detection"
LABEL version="1.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements first (for better caching)
COPY requirements.txt /workspace/

# Install PyTorch with CUDA support
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install -r requirements.txt

# Copy project files
COPY . /workspace/

# Create necessary directories
RUN mkdir -p data/cure-tsd data/yolo_dataset_full data/masks \
    models outputs runs/train runs/rl_training

# Set permissions
RUN chmod +x scripts/*.py

# Expose TensorBoard port
EXPOSE 6006

# Default command
CMD ["/bin/bash"]
