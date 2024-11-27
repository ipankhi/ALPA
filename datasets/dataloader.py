import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from .augmentation import get_aug
from utils import get_args_parser_aptos, get_model

# Define CustomDataset for training and validation
class CustomDataset(Dataset):
    def __init__(self, args, df, labels=None, transform=None):
        """
        Initialize the CustomDataset.

        Parameters:
        - args: Argument parser containing various configuration settings.
        - df: DataFrame containing image identifiers.
        - labels: Series containing labels corresponding to images (optional).
        - transform: Transformations to be applied on the images.
        """
        self.args = args
        self.df = df
        self.labels = labels
        self.transform = transform

        if self.labels is not None:
            # Initialize and fit the LabelEncoder and OneHotEncoder with labels
            self.label_encoder = LabelEncoder()
            self.one_hot_encoder = OneHotEncoder()
            self.labels_encoded = self.label_encoder.fit_transform(
                self.labels).reshape(-1, 1)
            self.one_hot_encoder.fit(self.labels_encoded)

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding label (if available) by index.
        
        Parameters:
        - idx: Index to retrieve data from.
        
        Returns:
        - img: Transformed image.
        - one_hot_label: One-hot encoded label if labels are provided.
        """
        # Retrieve image path from DataFrame
        img_path = self.df.iloc[idx]
        img_path_with_extension = f"{img_path}.png"
        full_img_path = os.path.join(self.args.img_path, img_path_with_extension)

        # Read image using OpenCV
        cv2_img = cv2.imread(full_img_path, cv2.IMREAD_COLOR)

        # Apply transformations to the image
        img = self.transform(cv2_img)

        if self.labels is not None:
            # Encode label and convert to one-hot representation
            label = self.labels.iloc[idx]
            label_encoded = self.label_encoder.transform([label])[0]
            one_hot_label = self.one_hot_encoder.transform(
                [[label_encoded]]).toarray().squeeze()

            return img, one_hot_label

        return img

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.df)

# Define a separate CustomDataset for test data
class TestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.df['diagnosis'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding label by index.
        
        Parameters:
        - idx: Index to retrieve data from.
        
        Returns:
        - image: Transformed image.
        - label: Encoded label.
        """
        img_name = os.path.join(self.img_dir, self.df.iloc[idx]["id_code"] + '.png')
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Utility functions
def k_fold_indices(n_samples, n_splits=5, shuffle=True, random_state=None):
    """
    Generate train and validation indices for k-fold cross-validation.
    
    Parameters:
    - n_samples: Total number of samples.
    - n_splits: Number of folds for cross-validation.
    - shuffle: Whether to shuffle data before splitting.
    - random_state: Seed for reproducibility.
    
    Returns:
    - Generator of train and validation indices for each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return kf.split(np.arange(n_samples))

def get_kfold_loader(args, train_X, train_Y, valid_X, valid_Y):
    """
    Create DataLoader objects for training and validation sets.
    
    Parameters:
    - args: Argument parser containing various configuration settings.
    - train_X: Training image identifiers.
    - train_Y: Training labels.
    - valid_X: Validation image identifiers.
    - valid_Y: Validation labels.
    
    Returns:
    - train_dl: DataLoader for the training set.
    - valid_dl: DataLoader for the validation set.
    """
    # Get augmentation strategies for training and validation sets
    train_aug, valid_aug = get_aug(args)

    # Create datasets for training and validation sets
    train_dataset = CustomDataset(args, train_X, train_Y, train_aug)
    valid_dataset = CustomDataset(args, valid_X, valid_Y, valid_aug)

    # Create DataLoader objects
    train_dl = DataLoader(
        train_dataset, batch_size=args.batchsize, num_workers=4, shuffle=False, pin_memory=True
    )
    valid_dl = DataLoader(
        valid_dataset, batch_size=args.batchsize, num_workers=4, shuffle=False, pin_memory=True
    )

    return train_dl, valid_dl

def get_dataloader(args, k_fold_splits=5):
    """
    Create DataLoader objects for each fold in k-fold cross-validation.
    
    Parameters:
    - args: Argument parser containing various configuration settings.
    - k_fold_splits: Number of folds for cross-validation.
    
    Returns:
    - train_dl_list: List of DataLoader objects for training sets across all folds.
    - valid_dl_list: List of DataLoader objects for validation sets across all folds.
    - train_class_counts: List of class counts in training sets for each fold.
    - valid_class_counts: List of class counts in validation sets for each fold.
    """
    print("Initializing KFold dataloader...")

    # Read data from CSV file
    train_df = pd.read_csv(args.csv_path)
    train_X, train_Y = train_df['id_code'], train_df['diagnosis']
    n_samples = len(train_X)

    # Lists to store DataLoader objects and class counts for each fold
    train_dl_list, valid_dl_list = [], []
    train_class_counts, valid_class_counts = [], []

    # Perform k-fold cross-validation
    for train_index, valid_index in k_fold_indices(n_samples, n_splits=k_fold_splits, random_state=args.seed):
        # Split data into training and validation sets for the current fold
        train_X_fold, valid_X_fold = train_X.iloc[train_index], train_X.iloc[valid_index]
        train_Y_fold, valid_Y_fold = train_Y.iloc[train_index], train_Y.iloc[valid_index]

        # Count occurrences of each class in both folds
        train_counts = train_Y_fold.value_counts(sort=False).tolist()
        valid_counts = valid_Y_fold.value_counts(sort=False).tolist()

        # Store class counts for this fold
        train_class_counts.append(train_counts)
        valid_class_counts.append(valid_counts)

        # Create DataLoader objects for this fold and store them
        train_dl, valid_dl = get_kfold_loader(args, train_X_fold, train_Y_fold, valid_X_fold, valid_Y_fold)
        train_dl_list.append(train_dl)
        valid_dl_list.append(valid_dl)

    return train_dl_list, valid_dl_list, train_class_counts, valid_class_counts
