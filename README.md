# PLO AI Training and CLI Game

This project contains the AI components for a Pot Limit Omaha (PLO) poker trainer, including a Deep Q-Network (DQN) agent, training script, and metrics collection.

## Quick Start

1. Build the base image:
    ```bash
    cd plogame/scripts
    ./build_base_image.sh
    ```

2. Start Prometheus and the trainer:
    ```bash
    ./start.sh
    ```

3. Once inside the container, run the training script:
    ```bash
    python3 src/train.py
    ```

## Training Options

When running train.py, you'll be presented with a menu:
```
PLO AI Training and Game Menu
1. Train AI
2. Play Against AI
3. Exit
```

### Training Modes
When selecting "Train AI", you have several options:
1. Train both positions (OOP and IP)
2. Train OOP only
3. Train IP only
4. Train against existing model

### Playing Against AI
When selecting "Play Against AI":
1. Choose your position (OOP or IP)
2. Select a pre-trained model
3. Play hands interactively against the AI

## CUDA Support

This project supports CUDA for GPU acceleration. To use CUDA:

1. Ensure you have NVIDIA GPU drivers installed on your host machine
2. Install CUDA on your host system (version 12.1)
3. Install the NVIDIA Container Toolkit:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

4. Run the container with GPU support:
    ```bash
    docker run -it --gpus all \
        -v ./models:/app/models \
        --name plo_trainer \
        --network plo_network \
        -p 8000:8000 \
        plo_trainer
    ```

## Project Structure

### Key Files
- `agent.py`: Defines the DQN agent for AI decision-making
- `train.py`: Main script for training the AI and playing against it
- `metrics.py`: Sets up metrics collection for monitoring AI performance
- `ai_trainer.py`: Contains the core poker game logic and training environment

### Metrics and Monitoring
- Prometheus metrics available on port 8000
- Access Prometheus UI at http://localhost:9090

## Customization

- Adjust hyperparameters in agent.py to optimize AI performance
- Modify the network architecture in the DQN class for different model complexities
- Add or remove metrics in metrics.py as needed for your monitoring setup

## Models

Trained models are saved in the `./models` directory with timestamps. When playing against the AI or training against an existing model, you can select from these saved models.
