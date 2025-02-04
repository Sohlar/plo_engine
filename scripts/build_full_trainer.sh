#!/bin/bash
cd ../ai
docker build -t pybase . -f ./Dockerfile --no-cache
docker build -t plo_trainer . -f ./Dockerfile.train --no-cache
cd ../scripts