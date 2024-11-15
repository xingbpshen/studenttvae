from torch.utils.data import Dataset
import torch


class StatlogDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        # Calculate mean and std deviation for each feature for standardization
        self.mean = self.features.mean(dim=0)
        self.std = self.features.std(dim=0)

    def z_score(self, x):
        """Apply standardization to a feature vector."""
        return (x - self.mean) / self.std

    def min_max(self, x):
        """Apply min-max scaling to a feature vector."""
        min = 27
        max = 157
        return (x - min) / (max - min)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        # Apply normalization
        x = self.min_max(x)
        return x, y
