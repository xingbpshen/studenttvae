from sklearn.model_selection import train_test_split
import pandas as pd
from torchvision import transforms
from datasets.celeba_gender import CelebAGenderDataset
from datasets.statlog import StatlogDataset
import os
import numpy as np
from sklearn.utils import shuffle
from utils import ToTensorTransform


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

        # # Path to CelebA dataset and attributes file
        # celeba_path = os.path.join(config.data.data_dir, "img_align_celeba")
        # attr_path = os.path.join(config.data.data_dir, "list_attr_celeba.txt")
        #
        # # Load attribute data
        # attributes = pd.read_csv(attr_path, sep='\s+', header=1)
        # attributes['Male'] = (attributes['Male'] == 1).astype(int)  # Binary label: Male=1, Female=0
        #
        # # Separate male and female images
        # female_data = attributes[attributes['Male'] == 0]
        # male_data = attributes[attributes['Male'] == 1]
        #
        # # Define split ratios
        # train_ratio = config.data.train_ratio
        # val_ratio = config.data.val_ratio
        # test_ratio = config.data.test_ratio
        #
        # # Define gender balance
        # female_ratio = config.data.female_ratio
        # male_ratio = config.data.male_ratio
        #
        # # Split female data
        # train_female, test_val_female = train_test_split(female_data, train_size=train_ratio * female_ratio,
        #                                                  random_state=42)
        # val_female, test_female = train_test_split(test_val_female,
        #                                            train_size=(val_ratio / (val_ratio + test_ratio)) * female_ratio,
        #                                            random_state=42)
        #
        # # Split male data
        # train_male, test_val_male = train_test_split(male_data, train_size=train_ratio * male_ratio, random_state=42)
        # val_male, test_male = train_test_split(test_val_male,
        #                                        train_size=(val_ratio / (val_ratio + test_ratio)) * male_ratio,
        #                                        random_state=42)
        #
        # # Combine male and female splits
        # train_data = pd.concat([train_female, train_male])
        # val_data = pd.concat([val_female, val_male])
        # test_data_A = pd.concat([test_female, test_male])
        # test_data_B = test_male  # 100% male for Test Set B
        #
        # # Shuffle the final splits
        # train_data = train_data.sample(frac=1, random_state=42)
        # val_data = val_data.sample(frac=1, random_state=42)
        # test_data_A = test_data_A.sample(frac=1, random_state=42)
        # test_data_B = test_data_B.sample(frac=1, random_state=42)
        #
        # # Confirm split sizes
        # print(f"Training set (f+m) size: {len(train_data)}")
        # print(f"Validation set (f+m) size: {len(val_data)}")
        # print(f"Testing set A (f+m) size: {len(test_data_A)}")
        # print(f"Testing set B (m) size: {len(test_data_B)}")
        #
        # # Function to load images using torchvision transforms
        # transform = transforms.Compose([
        #     transforms.CenterCrop(178),
        #     transforms.Resize((64, 64)),  # Resize to 64x64 as specified
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        # ])
        #
        # # Load datasets
        # train_dataset = CelebAGenderDataset(img_dir=celeba_path, df=train_data, transform=transform)
        # val_dataset = CelebAGenderDataset(img_dir=celeba_path, df=val_data, transform=transform)
        # test_dataset_A = CelebAGenderDataset(img_dir=celeba_path, df=test_data_A, transform=transform)
        # test_dataset_B = CelebAGenderDataset(img_dir=celeba_path, df=test_data_B, transform=transform)
        #
        # return train_dataset, val_dataset, test_dataset_A, test_dataset_B

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
