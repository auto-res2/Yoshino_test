import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models import get_model

def evaluate_model(model, device, test_loader):
    """
    Evaluate the model on the test set.
    
    Args:
        model: PyTorch model
        device: Device to evaluate on
        test_loader: DataLoader for test data
        
    Returns:
        test_loss: Average test loss
        accuracy: Test accuracy
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    
    return test_loss, accuracy

def evaluate_optimizers(config, test_loader, device='cuda'):
    """
    Evaluate and compare different optimizers.
    
    Args:
        config: Configuration dictionary
        test_loader: DataLoader for test data
        device: Device to evaluate on
        
    Returns:
        results: Dictionary with evaluation results for each optimizer
    """
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA is not available. Using CPU instead.")
    device = torch.device(device)
    
    # Get optimizers to compare
    optimizers = config.get('optimizers_to_compare', ['adam', 'sgd', 'adam_walign'])
    
    # Results dictionary
    results = {}
    
    for optimizer_name in optimizers:
        print(f"\nEvaluating {optimizer_name.upper()} optimizer...")
        
        # Load model
        model = get_model(config).to(device)
        model_path = os.path.join(config['save_dir'], f"{config['dataset']}_{optimizer_name}_best.pt")
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
            
            # Evaluate model
            test_loss, accuracy = evaluate_model(model, device, test_loader)
            
            # Store results
            results[optimizer_name] = {
                'test_loss': test_loss,
                'accuracy': accuracy
            }
        else:
            print(f"Model file {model_path} not found. Skipping evaluation.")
    
    return results

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, optimizer_name, save_dir='./results'):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accuracies: List of training accuracies
        val_accuracies: List of validation accuracies
        optimizer_name: Name of the optimizer
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {optimizer_name.upper()}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{optimizer_name}_loss.png'))
    
    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Training and Validation Accuracy - {optimizer_name.upper()}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{optimizer_name}_accuracy.png'))

def compare_optimizers(results, save_dir='./results'):
    """
    Compare different optimizers and plot results.
    
    Args:
        results: Dictionary with evaluation results for each optimizer
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    optimizers = list(results.keys())
    test_losses = [results[opt]['test_loss'] for opt in optimizers]
    accuracies = [results[opt]['accuracy'] for opt in optimizers]
    
    # Plot test losses
    plt.figure(figsize=(10, 5))
    plt.bar(optimizers, test_losses)
    plt.xlabel('Optimizer')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Comparison')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(save_dir, 'optimizer_test_loss_comparison.png'))
    
    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.bar(optimizers, accuracies)
    plt.xlabel('Optimizer')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(save_dir, 'optimizer_accuracy_comparison.png'))
    
    # Print comparison table
    print("\nOptimizer Comparison:")
    print("-" * 50)
    print(f"{'Optimizer':<15} {'Test Loss':<15} {'Accuracy (%)':<15}")
    print("-" * 50)
    for opt in optimizers:
        print(f"{opt:<15} {results[opt]['test_loss']:<15.4f} {results[opt]['accuracy']:<15.2f}")
    print("-" * 50)

if __name__ == "__main__":
    # Test the evaluation functions with a simple configuration
    from src.preprocess import load_mnist
    from src.train import train_model
    
    config = {
        'dataset': 'mnist',
        'optimizer': 'adam_walign',
        'optimizers_to_compare': ['adam', 'sgd', 'adam_walign'],
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'align_threshold': 0.1,
        'caution_factor': 0.5,
        'epochs': 2,
        'log_interval': 100,
        'save_dir': './models'
    }
    
    _, _, test_loader = load_mnist()
    
    # Evaluate optimizers
    results = evaluate_optimizers(config, test_loader, device='cpu')
    
    # Compare optimizers
    if results:
        compare_optimizers(results)
    else:
        print("No results to compare. Train models first.")
