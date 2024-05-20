# Use an official Python runtime with CUDA support as a parent image
FROM python:3.11-slim

# Install Python
RUN apt-get update && apt-get install -y \
    gcc

# Install Poetry
RUN pip3 install poetry 

# Install psutil to aviod poetry building from source
RUN pip3 install psutil

# Set the working directory in the container to /app
WORKDIR /app
# ENV PYTHONPATH=${PYTHONPATH}:${PWD} 

# Copy only requirements to cache them in docker layer
COPY pyproject.toml /app/

# Install 'psutil' with Poetry without actually installing it
RUN poetry add psutil

# Install project dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root --no-dev

# Add current directory files to /app in the container
COPY . /app

# Set the working directory to the repository
WORKDIR /app

# Install the package
RUN python3 -m pip install .

ENV MODEL_PACKAGE_DIR=/app/models/bow_model

# Run inference.py when the container launches
ENTRYPOINT ["python3", "/app/scripts/inference.py"]