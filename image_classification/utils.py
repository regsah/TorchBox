"""
Contains utility functions for training a PyTorch model
"""
import torch
from pathlib import Path

def save_model(model, target_dir, model_name):
    """
    Saves a PyTorch model to a given directory.

    Args:
    model (torch.nn.Module): The model.
    target_dir (str): the path to the directory.
    model_name (str): name of the file of the model. (should include ".pth" or ".pt" as the file extension)
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create the path for model file
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
