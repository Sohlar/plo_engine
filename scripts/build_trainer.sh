#!/bin/bash
cd ../ai
docker build -t plo_trainer . -f ./Dockerfile.train --no-cache
cd ../scripts