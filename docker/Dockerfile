ARG BASE_IMAGE=nvidia/cuda:12.1.1-base-ubuntu22.04
FROM ${BASE_IMAGE}

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y curl vim python3 pip

WORKDIR /app

RUN mkdir logs models

ARG REQUIREMENTS=requirements.txt
RUN echo "Using requirements file: $REQUIREMENTS"
COPY $REQUIREMENTS /app/requirements.txt

RUN pip install -r requirements.txt
