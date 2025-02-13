#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Merge multiple PyTorch models"
    echo
    echo "Options:"
    echo "  --model-dir PATH   Directory containing model files (required)"
    echo "  --output PATH      Output path for merged model (required)"
    echo "  --device TYPE      Device to use (cpu/cuda) (default: cpu)"
    echo
    echo "Example:"
    echo "  $0 --model-dir ./models/batch1 --output ./models/merged/final.pth"
    exit 1
}

# Default value for device
DEVICE="cpu"

# Check for minimum arguments
if [ "$#" -lt 4 ]; then
    usage
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --device)
            if [ -z "$2" ]; then
                echo "Error: --device requires a value"
                usage
            fi
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_DIR" ] || [ -z "$OUTPUT" ]; then
    echo "Error: Both --model-dir and --output are required"
    usage
fi

# Move to project root directory
cd "$(dirname "$0")/.."

# Get absolute paths for mounting
MODEL_DIR=$(realpath "$MODEL_DIR")
OUTPUT_DIR=$(dirname "$(realpath "$OUTPUT")")
OUTPUT_FILE=$(basename "$OUTPUT")

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Building merge container..."
if ! docker build -f "$(pwd)/docker/Dockerfile.merge" -t plo-merge . ; then
    echo "Error: Failed to build container"
    exit 1
fi

echo "Starting model merge..."
echo "Model directory: $MODEL_DIR"
echo "Output path: $OUTPUT_DIR/$OUTPUT_FILE"
echo "Device: $DEVICE"

# Run the Docker container with volume mounts using the merge image
docker run --rm \
    -v "$(pwd):/app" \
    -v "$MODEL_DIR:/app/input_models" \
    -v "$OUTPUT_DIR:/app/output" \
    plo-merge \
    --model-dir /app/input_models \
    --output "/app/output/$OUTPUT_FILE" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo "Model merge completed successfully"
else
    echo "Error: Model merging failed"
    exit 1
fi 