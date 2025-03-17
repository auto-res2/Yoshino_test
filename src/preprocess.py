import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_mnist(data_dir='./data', batch_size=64, test_batch_size=1000):
    """
    Load and preprocess the MNIST dataset.
    
    Args:
        data_dir (str): Directory to store the dataset
        batch_size (int): Batch size for training
        test_batch_size (int): Batch size for testing
        
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for training, validation and testing
    """
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    
    # Split training data into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    
    # Download and load test data
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def load_cifar10(data_dir='./data', batch_size=64, test_batch_size=1000):
    """
    Load and preprocess the CIFAR-10 dataset.
    
    Args:
        data_dir (str): Directory to store the dataset
        batch_size (int): Batch size for training
        test_batch_size (int): Batch size for testing
        
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for training, validation and testing
    """
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download and load training data
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    
    # Split training data into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    
    # Download and load test data
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_dataloaders(config):
    """
    Get data loaders based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for training, validation and testing
    """
    dataset = config.get('dataset', 'mnist')
    data_dir = config.get('data_dir', './data')
    batch_size = config.get('batch_size', 64)
    test_batch_size = config.get('test_batch_size', 1000)
    
    if dataset.lower() == 'mnist':
        return load_mnist(data_dir, batch_size, test_batch_size)
    elif dataset.lower() == 'cifar10':
        return load_cifar10(data_dir, batch_size, test_batch_size)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

if __name__ == "__main__":
    # Test the preprocessing functions
    train_loader, val_loader, test_loader = load_mnist()
    print(f"MNIST - Training samples: {len(train_loader.dataset)}")
    print(f"MNIST - Validation samples: {len(val_loader.dataset)}")
    print(f"MNIST - Test samples: {len(test_loader.dataset)}")
    
    train_loader, val_loader, test_loader = load_cifar10()
    print(f"CIFAR10 - Training samples: {len(train_loader.dataset)}")
    print(f"CIFAR10 - Validation samples: {len(val_loader.dataset)}")
    print(f"CIFAR10 - Test samples: {len(test_loader.dataset)}")
