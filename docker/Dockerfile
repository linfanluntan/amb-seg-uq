FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

LABEL maintainer="Renjie He <rhe@mdanderson.org>"
LABEL description="Ambiguity-Aware UQ for Medical Image Segmentation"

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install nnU-Net V2
RUN pip install --no-cache-dir nnunetv2

# Copy project
COPY . /workspace/

# Install project in editable mode
RUN pip install -e .

# Default command
CMD ["bash"]
