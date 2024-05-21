# Use an official Python runtime with CUDA support as a parent image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python
RUN apt-get update && apt-get install -y \
    build-essential libbz2-dev libreadline-dev libsqlite3-dev curl git \
    xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    make libssl-dev zlib1g-dev llvm libncurses5-dev git gcc curl

RUN apt-get install -y python3.11 python3-pip python3.11-dev

# Install Poetry
RUN python3.11 -m pip install poetry 

# Install psutil to avoid poetry building from source
RUN python3.11 -m pip install psutil

# Set the working directory in the container to /app
WORKDIR /app

# Copy only requirements to cache them in docker layer
COPY pyproject.toml /app/

# Install project dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root --no-dev

# Add current directory files to /app in the container
COPY . /app

# Set the working directory to the repository
WORKDIR /app

# Install the package
RUN poetry install

ENV MODEL_PACKAGE_DIR=/app/models/bow_model

# Run inference.py when the container launches
ENTRYPOINT ["python3.11", "scripts/inference.py"]