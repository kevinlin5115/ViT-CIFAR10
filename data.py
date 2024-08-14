import os
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        # (3072, ) -> (3, 32, 32)
        image = image.reshape(3, 32, 32)

        image = Image.fromarray(image.transpose(1, 2, 0))

        if self.transform:
            image = self.transform(image)

        return image, label
    
def load_cifar_10_batch(file):
    """
    Load a single batch from the CIFAR-10 dataset.
    """
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        return data, labels
    
def load_cifar_10_data(data_dir):
    """
    Load all CIFAR-10 data.
    """
    train_data = []
    train_labels = []
    for i in range(1, 6):
        data, labels = load_cifar_10_batch(os.path.join(data_dir, f'data_batch_{i}'))
        train_data.append(data)
        train_labels.append(labels)

    # Combine training batches
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    # Load test data
    test_data, test_labels = load_cifar_10_batch(os.path.join(data_dir, 'test_batch'))

    return (train_data, train_labels), (test_data, test_labels)

def get_dataloaders(data_dir, batch_size, val_split=0.1, img_size=32, num_workers=2):
    """
    Prepare CIFAR-10 DataLoaders with a training/validation split.
    """
    (train_data, train_labels), (test_data, test_labels) = load_cifar_10_data(data_dir)

    # Create a dataset
    full_train_dataset = CIFAR10Dataset(train_data, train_labels)

    # Calculatelengths for training and validation sets
    val_len = int(len(full_train_dataset) * val_split)
    train_len = len(full_train_dataset) - val_len

    # Split the dataset
    train_dataset, val_dataset = random_split(full_train_dataset, [train_len, val_len])

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Apply transformations to train and validation datasets
    train_dataset.dataset.transform = transform
    val_dataset.dataset.transform = transform

    # Prepare the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataset = CIFAR10Dataset(test_data, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader