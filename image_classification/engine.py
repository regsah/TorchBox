"""
Contains train_step, test_step and train functions. 
Handles everything related to training a given model.
"""
import torch

from tqdm.auto import tqdm

def train_step(model, dataloader, loss_fn, optimizer, device="cpu"):
  """
  Handles a single training step.

  Args:
    model (torch.nn.Module): The PyTorch model that is going to be trained
    dataloader (torch.utils.data.DataLoader): the training DataLoader
    loss_fn (torch.nn.Module): loss function.
    optimizer (torch.optim.Optimizer): optimizer.
    device (torch.device or str): the device that will be used for computation ("cuda" or "cpu").

  Returns:
    tuple: A tuple containing:
      - train_loss (float): the total loss of a training step.
      - train_acc (float): the total accuracy of a training step.
  """
  # Put the model in train mode
  model.train()

  # Initialize train_loss and train_acc
  train_loss, train_acc = 0, 0

  # Main loop of the training step
  for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # Forward pass
      y_pred = model(X)

      # Calculate loss
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # Zero_grad the Optimizer to prevent accumulation
      optimizer.zero_grad()

      # Backward propagation
      loss.backward()

      # Step the Optimizer
      optimizer.step()

      # Calculate accuracy
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # get the average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)

  return train_loss, train_acc



def test_step(model, dataloader, loss_fn, device="cpu"):
  """
  Handles a single test step.

  Args:
    model (torch.nn.Module): The PyTorch model that is going to be tested.
    dataloader (torch.utils.data.DataLoader): The testing DataLoader.
    loss_fn (torch.nn.Module): loss function.
    device (torch.device or str): the device that will be used for computation ("cuda" or "cpu").

  Returns:
    tuple: A tuple containing:
      - test_loss (float): the total loss of a test step.
      - test_acc (float): the total accuracy of a test step.
  """
  # Put model in eval mode
  model.eval() 

  # Initialize test_loss and test_acc
  test_loss, test_acc = 0, 0

  # Turn on inference context manager
  with torch.inference_mode():
      # Main loop of the test step
      for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.to(device), y.to(device)

          # Forward pass
          test_pred_logits = model(X)

          # Calculate loss
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()

          # Calculate accuracy
          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  # get the average loss and accuracy per batch 
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)

  return test_loss, test_acc



def train(model, train_dataloader, test_dataloader, 
          optimizer, loss_fn, epochs, device):
  """
  Handles training and testing a model.

  Args:
    model (torch.nn.Module): The model that is going to be trained and tested.
    train_dataloader (torch.utils.data.DataLoader): A DataLoader for training.
    test_dataloader (torch.utils.data.DataLoader):A DataLoader for testing.
    optimizer (torch.optim.Optimizer): Optimizer.
    loss_fn (torch.nn.Module): Loss function.
    epochs (int): The number of epochs that the model will be trained for.
    device (torch.device or str): The device that the computation will be made on ("cuda" or "cpu").

  Returns:
    dictionary: A dictionary in the following form:
      {train_loss: [], train_acc: [], test_loss: [], test_acc: []}
  """
  # Create the results dictionary
  results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

  # Main loop for training and testing
  for epoch in tqdm(range(epochs)):

      #Call train_step and test_step
      train_loss, train_acc = train_step(model=model,
                                         dataloader=train_dataloader,
                                         loss_fn=loss_fn,
                                         optimizer=optimizer,
                                         device=device)

      test_loss, test_acc = test_step(model=model,
                                      dataloader=test_dataloader,
                                      loss_fn=loss_fn,
                                      device=device)

      # Print out this epoch's results
      print(f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}")

      # Update the results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  return results
