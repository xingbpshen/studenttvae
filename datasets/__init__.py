from sklearn.model_selection import train_test_split
import pandas as pd
from torchvision import transforms
from datasets.celeba_gender import CelebAGenderDataset
from datasets.statlog import StatlogDataset
from datasets.bimcv import BIMCVDataset
import os
import numpy as np
from sklearn.utils import shuffle


def get_dataset(args, config):
    if config.data.dataset == "CelebAGender":
        # Path to CelebA dataset and attributes file
        celeba_path = os.path.join(config.data.data_dir, "img_align_celeba")
        attr_path = os.path.join(config.data.data_dir, "list_attr_celeba.txt")

        # Load attribute data
        attributes = pd.read_csv(attr_path, sep='\s+', header=1)
        attributes['Male'] = (attributes['Male'] == 1).astype(int)  # Binary label: Male=1, Female=0

        # Gather all females and males
        all_females = attributes[attributes['Male'] == 0]
        all_males = attributes[attributes['Male'] == 1]
        num_females = len(all_females)

        # Calculate the number of males needed
        num_males_needed = int(num_females * (config.data.male_ratio / config.data.female_ratio))
        selected_males = all_males.sample(n=num_males_needed, random_state=42)
        remaining_males = all_males.drop(selected_males.index)

        # Combine selected male and female data
        combined_data = pd.concat([all_females, selected_males])

        # Define split ratios
        train_ratio = config.data.train_ratio
        val_ratio = config.data.val_ratio
        test_ratio = config.data.test_ratio

        # Split combined data into train, validation, and test sets
        train_data, test_val_data = train_test_split(combined_data, train_size=train_ratio, random_state=42)
        val_data, test_data_A = train_test_split(test_val_data, train_size=val_ratio / (val_ratio + test_ratio),
                                                 random_state=42)

        # Create test_dataset_B by sampling from remaining males
        test_data_B = remaining_males.sample(n=len(test_data_A), random_state=42)

        # Shuffle the final splits
        train_data = train_data.sample(frac=1, random_state=42)
        val_data = val_data.sample(frac=1, random_state=42)
        test_data_A = test_data_A.sample(frac=1, random_state=42)
        test_data_B = test_data_B.sample(frac=1, random_state=42)

        # Confirm split sizes
        print(f"Training set size: {len(train_data)}")
        print(f"Validation set size: {len(val_data)}")
        print(f"Testing set A size: {len(test_data_A)}")
        print(f"Testing set B size: {len(test_data_B)}")

        # Function to load images using torchvision transforms
        transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((64, 64)),  # Resize to 64x64 as specified
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

        # Load datasets
        train_dataset = CelebAGenderDataset(img_dir=celeba_path, df=train_data, transform=transform)
        val_dataset = CelebAGenderDataset(img_dir=celeba_path, df=val_data, transform=transform)
        test_dataset_A = CelebAGenderDataset(img_dir=celeba_path, df=test_data_A, transform=transform)
        test_dataset_B = CelebAGenderDataset(img_dir=celeba_path, df=test_data_B, transform=transform)

        return train_dataset, val_dataset, test_dataset_A, test_dataset_B

    elif config.data.dataset == "BIMCV":
        # Path to BIMCV dataset
        bimcv_path = config.data.data_dir

        # Gather all Normal and Tuberculosis images
        all_normals = pd.DataFrame(
            {'path': [os.path.join(bimcv_path, 'Normal', f) for f in os.listdir(os.path.join(bimcv_path, 'Normal'))],
             'label': 0})
        all_covid19s = pd.DataFrame({'path': [os.path.join(bimcv_path, 'COVID19', f) for f in
                                                   os.listdir(os.path.join(bimcv_path, 'COVID19'))], 'label': 1})
        num_normals = len(all_normals)

        # Calculate the number of Tuberculosis needed
        num_covid19s_needed = int(num_normals * (config.data.covid19_ratio / config.data.normal_ratio))
        selected_covid19s = all_covid19s.sample(n=num_covid19s_needed, random_state=42)
        remaining_covid19s = all_covid19s.drop(selected_covid19s.index)

        # Combine selected Normal and Tuberculosis data
        combined_data = pd.concat([all_normals, selected_covid19s])

        # Define split ratios
        train_ratio = config.data.train_ratio
        val_ratio = config.data.val_ratio
        test_ratio = config.data.test_ratio

        # Split combined data into train, validation, and test sets
        train_data, test_val_data = train_test_split(combined_data, train_size=train_ratio, random_state=42)
        val_data, test_data_A = train_test_split(test_val_data, train_size=val_ratio / (val_ratio + test_ratio),
                                                 random_state=42)

        # Create test_dataset_B by sampling from remaining Tuberculosis
        test_data_B = remaining_covid19s.sample(n=len(test_data_A), random_state=42)

        # Shuffle the final splits
        train_data = train_data.sample(frac=1, random_state=42)
        val_data = val_data.sample(frac=1, random_state=42)
        test_data_A = test_data_A.sample(frac=1, random_state=42)
        test_data_B = test_data_B.sample(frac=1, random_state=42)

        # Confirm split sizes
        print(f"Training set size: {len(train_data)}")
        print(f"Validation set size: {len(val_data)}")
        print(f"Testing set A size: {len(test_data_A)}")
        print(f"Testing set B size: {len(test_data_B)}")

        # Function to load images using torchvision transforms
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize to 64x64 as specified
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

        # Load datasets
        train_dataset = BIMCVDataset(df=train_data, transform=transform)
        val_dataset = BIMCVDataset(df=val_data, transform=transform)
        test_dataset_A = BIMCVDataset(df=test_data_A, transform=transform)
        test_dataset_B = BIMCVDataset(df=test_data_B, transform=transform)

        return train_dataset, val_dataset, test_dataset_A, test_dataset_B

    elif config.data.dataset == "Statlog":
        # Load the Statlog dataset
        # load npz file
        arrays = np.load(os.path.join(config.data.data_dir, 'statlog_landsat_satellite.npz'))
        X, y = [arrays[i] for i in arrays.files]
        # Shuffle X and y together to maintain correspondence
        X, y = shuffle(X, y, random_state=42)

        # Define split ratios
        train_ratio = config.data.train_ratio
        val_ratio = config.data.val_ratio
        test_ratio = config.data.test_ratio

        # Calculate the number of samples for each split
        total_samples = len(X)
        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)
        test_size = total_samples - train_size - val_size  # Ensure all samples are used

        # Split the data
        X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
        y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]
        # Confirm split sizes
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Testing set size: {len(X_test)}")

        # Create dataset instances for each split
        train_dataset = StatlogDataset(X_train, y_train)
        val_dataset = StatlogDataset(X_val, y_val)
        test_dataset = StatlogDataset(X_test, y_test)

        return train_dataset, val_dataset, test_dataset

    else:
        raise ValueError(f"Dataset {config.data.dataset} not recognized.")
