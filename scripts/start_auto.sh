#!/bin/bash

docker stop plo_trainer prom_poker || true
docker rm plo_trainer prom_poker || true

docker network rm plo_network || true

docker network create plo_network

# Move to the project root directory to use as build context
cd ..

# Build the specific images
echo "Building plo_trainer..."
if ! docker build -t plo_trainer -f docker/Dockerfile.train .; then
    echo "Failed to build plo_trainer"
    exit 1
fi

echo "Building prom_poker..."
if ! docker build -t prom_poker -f docker/Dockerfile.prom docker/; then
    echo "Failed to build prom_poker"
    exit 1
fi

# Run Prometheus in detached mode
echo "Starting Prometheus..."
docker run -d --name prom_poker --network plo_network -p 9090:9090 prom_poker

# Run the trainer in interactive mode
echo "Starting PLO trainer in interactive mode..."
docker run -it \
    --name plo_trainer \
    --network plo_network \
    -v ./models:/app/models \
    -p 8000:8000 \
    plo_trainer python3 src/train.py -i -n 50000

echo "Access Prometheus at http://localhost:9090"
