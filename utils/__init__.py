import os
import torch
import numpy as np


class SimpleLogger:
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, message):
        with open(os.path.join(self.log_path, "stdout.txt"), "a") as f:
            f.write(message + "\n")

    def save_model_checkpoint(self, model, epoch):
        torch.save(model.state_dict(), os.path.join(self.log_path, f"best_model_epoch_{epoch}.pt"))


class ToTensorTransform:
    def __call__(self, data):
        # Convert the data to a NumPy array if it's not already
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        # Convert to a PyTorch tensor
        return torch.from_numpy(data)


def load_latest_model(model, log_path):
    # List all files in the log_path directory
    model_files = [f for f in os.listdir(log_path) if f.startswith("best_model_epoch_") and f.endswith(".pt")]

    # Extract epoch numbers from filenames and find the largest epoch
    if not model_files:
        raise FileNotFoundError("No model files found in the specified directory.")

    # Parse out the epoch numbers from filenames
    epochs = [int(f.split("best_model_epoch_")[1].split(".pt")[0]) for f in model_files]
    max_epoch = max(epochs)

    # Construct the filepath of the latest model
    latest_model_path = os.path.join(log_path, f"best_model_epoch_{max_epoch}.pt")

    # Load the model state
    model.load_state_dict(torch.load(latest_model_path))
    print(f"Loaded model from {latest_model_path}")

    return model
