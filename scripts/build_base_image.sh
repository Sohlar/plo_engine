#!/bin/bash
cd ../ai

# Accept optional parameter for CUDA/CPU
BUILD_TYPE=$1

if [ -z "$BUILD_TYPE" ]; then
  # Architecture Detection
  ARCH=$(uname -m)
  if [ "$ARCH" = "x86_64" ]; then
    BUILD_TYPE="cuda"
  elif [ "$ARCH" = "arm64" ]; then
    BUILD_TYPE="cpu"
  else
    echo "Unsupported architecture: $ARCH"
    exit 1
  fi
fi

# Set variables based on build type
if [ "$BUILD_TYPE" = "cuda" ]; then
  BASE_IMAGE="nvidia/cuda:12.1.1-base-ubuntu22.04"
  REQUIREMENTS="requirements_cuda.txt"
elif [ "$BUILD_TYPE" = "cpu" ]; then
  BASE_IMAGE="ubuntu:22.04"
  REQUIREMENTS="requirements_cpu.txt"
else
  echo "Invalid build type. Use 'cuda' or 'cpu'"
  echo "Usage: $0 [cuda|cpu]"
  exit 1
fi

echo "Building with:"
echo "Base Image: $BASE_IMAGE"
echo "Requirements: $REQUIREMENTS"

# Pull the base image first
docker pull "$BASE_IMAGE"

# Move to the docker directory to use it as build context
cd ../docker

# Copy requirements file to build context
cp "../requirements/$REQUIREMENTS" "./requirements.txt"

docker build -t pybase . \
  --build-arg BASE_IMAGE=$BASE_IMAGE \
  --build-arg REQUIREMENTS=requirements.txt \
  --no-cache

# Clean up copied requirements file
rm "./requirements.txt"

cd ../scripts
