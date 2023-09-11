# TorchBox

TorchBox is a collection of python scripts created to provide ease of usage when training PyTorch models. 
Currently only has scripts supporting image classification.

## Image Classification (image_classification)

The `image_classification` folder contains scripts and utilities specifically made for image classification tasks. Here's what you'll find in this folder:

- `data_setup.py`: Provides functions for setting up PyTorch DataLoaders for image classification datasets.
- `engine.py`: Contains functions for training and evaluating image classification models.
- `utils.py`: Includes various utility functions to assist with image classification tasks.

### Importing the Scripts

You can import these scripts into your project using the following code:

```
try:
    from image_classification import data_setup, engine, utils
except:
    print("[INFO] Couldn't find image_classification scripts... downloading them from GitHub.")
    !git clone https://github.com/regsah/TorchBox/
    !mv TorchBox/image_classification .
    !rm -rf TorchBox
    from image_classification import data_setup, engine, utils
```
