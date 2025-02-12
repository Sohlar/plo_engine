import torch
import os
from pathlib import Path
import argparse
from typing import List
import logging
from agent2 import PLONetwork
from constants import STATE_SIZE  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models(model_dir: str) -> List[torch.nn.Module]:
    """
    Load all .pth models from the specified directory
    """
    models = []
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Directory {model_dir} does not exist")
    
    for model_path in model_dir.glob("*.pth"):
        try:
            # Create a new network instance
            network = PLONetwork(STATE_SIZE)
            # Load the state dict
            loaded_data = torch.load(model_path, map_location='cpu')
            
            # Handle different types of saved data
            if isinstance(loaded_data, dict):
                # If it's already a state dict
                network.load_state_dict(loaded_data)
            elif hasattr(loaded_data, 'state_dict'):
                # If it's a full model
                network.load_state_dict(loaded_data.state_dict())
            else:
                raise ValueError(f"Unrecognized model format in {model_path}")
                
            models.append(network)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            continue
    
    if not models:
        raise ValueError("No valid models found in directory")
    
    return models

def merge_models(models: List[torch.nn.Module], device: str) -> torch.nn.Module:
    """
    Merge multiple models by averaging their parameters
    Args:
        models: List of models to merge
        device: Device to perform merging on
    """
    if not models:
        raise ValueError("No models provided for merging")
    
    # Create new model on the specified device
    merged_model = PLONetwork(STATE_SIZE).to(device)
    merged_state_dict = merged_model.state_dict()
    
    # Move all models to the specified device first
    models = [model.to(device) for model in models]
    
    # For each parameter, compute the average across all models
    for key in merged_state_dict:
        # Stack parameters from all models
        params = torch.stack([model.state_dict()[key] for model in models])
        # Compute mean along the model dimension
        merged_state_dict[key] = torch.mean(params, dim=0)
    
    # Load the merged parameters into the new model
    merged_model.load_state_dict(merged_state_dict)
    return merged_model

def main():
    parser = argparse.ArgumentParser(description="Merge multiple PyTorch models")
    parser.add_argument("--model-dir", type=str,
                       default=os.environ.get('MODEL_DIR'),
                       help="Directory containing .pth model files")
    parser.add_argument("--output", type=str,
                       default=os.environ.get('OUTPUT_PATH'),
                       help="Output path for merged model")
    parser.add_argument("--device", type=str, 
                       default=os.environ.get('DEVICE', "cuda" if torch.cuda.is_available() else "cpu"),
                       help="Device to use for merging (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.model_dir or not args.output:
        parser.error("Both --model-dir and --output are required (either as arguments or environment variables)")
    
    try:
        # Load all models
        logger.info(f"Loading models from {args.model_dir}")
        models = load_models(args.model_dir)
        
        # Merge the models
        logger.info(f"Merging {len(models)} models on {args.device}")
        merged_model = merge_models(models, args.device)
        
        # Save the merged model's state dict
        torch.save(merged_model.state_dict(), args.output)
        logger.info(f"Saved merged model to {args.output}")
        
    except Exception as e:
        logger.error(f"Error during model merging: {str(e)}")
        raise

if __name__ == "__main__":
    main() 