import os
from PIL import Image
from torch.utils.data import Dataset


class CelebAGenderDataset(Dataset):
    def __init__(self, img_dir, df, transform=None):
        """
        Custom dataset for CelebA with a flat directory structure.
        Args:
            img_dir (str): Directory containing all images.
            df (DataFrame): DataFrame with image filenames as index and labels.
            transform (callable, optional): Transform to apply to the images.
        """
        self.img_dir = img_dir
        self.df = df
        self.transform = transform
        self.image_paths = [str(idx) for idx in df.index.tolist()]  # Ensure all paths are strings
        self.labels = df['Male'].values

        # Load all images into memory
        self.images = []
        for img_path in self.image_paths:
            full_img_path = os.path.join(self.img_dir, img_path)
            image = Image.open(full_img_path).convert('RGB')  # Convert to RGB to ensure consistency
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
