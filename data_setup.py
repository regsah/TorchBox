"""
Contains create_dataloaders which is used for creating PyTorch DataLoaders for 
image classification.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders( train_dir, test_dir, transform, 
                        batch_size, num_workers=NUM_WORKERS):
  """
  Contains create_dataloaders, which is used for creating PyTorch DataLoaders for 
  image classification.

  Args:
      train_dir (str): Path to the training data directory.
      test_dir (str): Path to the testing data directory.
      transform (transforms.Compose): PyTorch transforms to be applied to the data.
      batch_size (int): Number of data samples in a single batch.
      num_workers (int, optional): Number of workers per DataLoader.

  Returns:
      tuple: A tuple containing:
          - train_dataloader (DataLoader): DataLoader for the training dataset.
          - test_dataloader (DataLoader): DataLoader for the testing dataset.
          - class_names (list): List of class names.

  Example:
      train_dataloader, test_dataloader, class_names = create_dataloaders(
          train_dir='path/to/train',
          test_dir='path/to/test',
          transform=custom_transform,
          batch_size=64,
          num_workers=2
      )
  """

  # ImageFolder of torchvision is used to create the datasets
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get classes
  class_names = train_data.classes

  # Create data loaders themselves
  train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True)

  test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)


  return train_dataloader, test_dataloader, class_names
