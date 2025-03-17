import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam, SGD
from optimizer import AdamWAlignOptimizer
from models import get_model

def train_epoch(model, device, train_loader, optimizer, epoch, log_interval=100):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        device: Device to train on
        train_loader: DataLoader for training data
        optimizer: Optimizer to use
        epoch: Current epoch number
        log_interval: How often to log training progress
        
    Returns:
        train_loss: Average training loss for the epoch
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f} '
                  f'Accuracy: {100. * correct / total:.2f}%')
    
    train_loss /= len(train_loader)
    accuracy = 100. * correct / total
    
    print(f'Train set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    
    return train_loss, accuracy

def validate(model, device, val_loader):
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        device: Device to validate on
        val_loader: DataLoader for validation data
        
    Returns:
        val_loss: Average validation loss
        accuracy: Validation accuracy
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.cross_entropy(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss /= len(val_loader)
    accuracy = 100. * correct / total
    
    print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    
    return val_loss, accuracy

def train_model(config, train_loader, val_loader, device='cuda'):
    """
    Train a model with the specified configuration.
    
    Args:
        config: Configuration dictionary
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on
        
    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accuracies: List of training accuracies
        val_accuracies: List of validation accuracies
    """
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA is not available. Using CPU instead.")
    device = torch.device(device)
    
    # Get model
    model = get_model(config).to(device)
    
    # Get optimizer
    optimizer_name = config.get('optimizer', 'adam_walign')
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0)
    
    if optimizer_name.lower() == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        momentum = config.get('momentum', 0.9)
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adam_walign':
        align_threshold = config.get('align_threshold', 0.1)
        caution_factor = config.get('caution_factor', 0.5)
        optimizer = AdamWAlignOptimizer(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            align_threshold=align_threshold,
            caution_factor=caution_factor
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Training parameters
    epochs = config.get('epochs', 10)
    log_interval = config.get('log_interval', 100)
    save_dir = config.get('save_dir', './models')
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, epoch, log_interval)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, device, val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(save_dir, f"{config['dataset']}_{optimizer_name}_best.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")
    
    # Save final model
    model_path = os.path.join(save_dir, f"{config['dataset']}_{optimizer_name}_final.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies

if __name__ == "__main__":
    # Test the training function with a simple configuration
    from src.preprocess import load_mnist
    
    config = {
        'dataset': 'mnist',
        'optimizer': 'adam_walign',
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'align_threshold': 0.1,
        'caution_factor': 0.5,
        'epochs': 2,
        'log_interval': 100,
        'save_dir': './models'
    }
    
    train_loader, val_loader, _ = load_mnist()
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        config, train_loader, val_loader, device='cpu'
    )
    
    print("Training completed successfully!")
