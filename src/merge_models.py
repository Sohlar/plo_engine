import torch
import os
from pathlib import Path
import argparse
from typing import List
import logging

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
            model = torch.load(model_path)
            models.append(model)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            continue
    
    if not models:
        raise ValueError("No valid models found in directory")
    
    return models

def merge_models(models: List[torch.nn.Module]) -> torch.nn.Module:
    """
    Merge multiple models by averaging their parameters
    """
    if not models:
        raise ValueError("No models provided for merging")
    
    merged_state_dict = models[0].state_dict()
    
    # For each additional model
    for i in range(1, len(models)):
        current_state_dict = models[i].state_dict()
        
        # Verify that the architectures match
        if current_state_dict.keys() != merged_state_dict.keys():
            raise ValueError(f"Model {i} has different architecture than the first model")
        
        # Average the parameters
        for key in merged_state_dict:
            merged_state_dict[key] = (merged_state_dict[key] * i + current_state_dict[key]) / (i + 1)
    
    # Apply the merged parameters to the first model
    models[0].load_state_dict(merged_state_dict)
    return models[0]

def main():
    parser = argparse.ArgumentParser(description="Merge multiple PyTorch models")
    parser.add_argument("--model-dir", type=str, required=True, 
                       help="Directory containing .pth model files")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for merged model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for merging (cuda/cpu)")
    
    args = parser.parse_args()
    
    try:
        # Load all models
        logger.info(f"Loading models from {args.model_dir}")
        models = load_models(args.model_dir)
        
        # Move models to specified device
        for model in models:
            model.to(args.device)
        
        # Merge the models
        logger.info(f"Merging {len(models)} models")
        merged_model = merge_models(models)
        
        # Save the merged model
        torch.save(merged_model, args.output)
        logger.info(f"Saved merged model to {args.output}")
        
    except Exception as e:
        logger.error(f"Error during model merging: {str(e)}")
        raise

if __name__ == "__main__":
    main() 