"""
Contains utility functions for training a PyTorch model
"""
import torch
import torchvision
from torchvision import transforms

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

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



import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from typing import List, Tuple

from PIL import Image

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

def pred_and_plot_image(model, class_names, image_path, image_size=(224, 224), transform=None, device=device):
    """
    Predicts and plots an image using a specified model.

    Args:
        model (torch.nn.Module): Model to predict on an image.
        class_names (List[str]): class names of the data.
        image_path (str): Path to the image.
        image_size (Tuple[int, int], optional): Size that the image will be transformed to.
        transform (torchvision.transforms, optional): Transform to perform on image before prediction.
        device (torch.device, optional): The device that the computation will be made on ("cpu" or "cuda").
    """

    # Open image
    img = Image.open(image_path)

    # set the image_transform to the given transform or create it from scratch
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # Get the model on the device
    model.to(device)

    # Get the model to the eval mode
    model.eval()

    # Turn on the inference_mode
    with torch.inference_mode():
        # Add an extra dimension to the image since it requires the following format : [batch_size, color_channels, height, width]
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Make the prediction
        target_image_pred = model(transformed_image.to(device))

    # Convert logits to probabilities
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert probabilities to labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)
