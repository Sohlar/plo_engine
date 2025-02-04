# PLO AI Training and CLI Game

This project contains the AI components for a Pot Limit Omaha (PLO) poker trainer, a Deep Q-Network (DQN) agent, a training script, and metrics collection.

# Build and run the trainer:

    `cd plogame/scripts`

    `./build_base_image.sh`

    `cd ../ai && ./start.sh`

    CUDA
    `docker run -it --gpus all -v ./models:/app/models --name plo_trainer --network plo_network -p 8000:8000 plo_trainer`

    CPU
    `docker run -it -v ./models:/app/models --name plo_trainer --network plo_network -p 8000:8000 plo_trainer`

    `python3 ./train.py`

## CUDA Support

This project supports CUDA for GPU acceleration. To use CUDA:

1. Ensure you have NVIDIA GPU drivers installed on your host machine.
2. Install CUDA on your host system. The version used in this project is 12.1.
3. Install the NVIDIA Container Toolkit:
   ```
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
4. Use the `--gpus all` flag when running the Docker container to enable GPU support.

Note: Make sure your NVIDIA drivers and CUDA version on the host system are compatible with CUDA 12.1 used in the Docker container.

## Files

1. agent.py: Defines the DQN agent for AI decision-making.
2. train.py: Provides functionality to train the AI and play against it.
3. metrics.py: Sets up metrics collection for monitoring AI performance and system resources.

## Customization

- Adjust hyperparameters in agent.py to optimize AI performance
- Modify the network architecture in the DQN class for different model complexities
- Add or remove metrics in metrics.py as needed for your monitoring setup
