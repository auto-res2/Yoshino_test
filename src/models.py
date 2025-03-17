import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvNet(nn.Module):
    """
    A simple convolutional neural network for image classification.
    """
    def __init__(self, num_classes=10, input_channels=1):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CIFAR10Net(nn.Module):
    """
    A convolutional neural network for CIFAR-10 classification.
    """
    def __init__(self, num_classes=10):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_model(config):
    """
    Get model based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        model: PyTorch model
    """
    dataset = config.get('dataset', 'mnist')
    
    if dataset.lower() == 'mnist':
        return SimpleConvNet(num_classes=10, input_channels=1)
    elif dataset.lower() == 'cifar10':
        return CIFAR10Net(num_classes=10)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
