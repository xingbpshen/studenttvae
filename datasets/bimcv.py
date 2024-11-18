from PIL import Image
from torch.utils.data import Dataset


class BIMCVDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        Custom dataset for BIMCV.
        Args:
            df (DataFrame): DataFrame with image paths and labels.
            transform (callable, optional): Transform to apply to the images.
        """
        self.df = df
        self.transform = transform
        self.image_paths = df['path'].tolist()
        self.labels = df['label'].values

        # Load all images into memory
        self.images = []
        for img_path in self.image_paths:
            image = Image.open(img_path).convert('RGB')  # Convert to RGB to ensure consistency
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image and label
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]

        return image, label
