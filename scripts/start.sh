#!/bin/bash

docker stop plo_trainer prom_poker || true
docker rm plo_trainer prom_poker || true

docker network rm plo_network || true

docker network create plo_network

docker build -t plo_trainer -f Dockerfile.train .
docker build -t prom_poker -f Dockerfile.prom .

docker run -d --name prom_poker --network plo_network -p 9090:9090 prom_poker

echo "PLO Trainer and Prometheus are now running."
echo "Access Prometheus at http://localhost:9090"
