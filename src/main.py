import os
import argparse
import json
import torch
import matplotlib.pyplot as plt
from preprocess import get_dataloaders
from train import train_model
from evaluate import evaluate_optimizers, compare_optimizers, plot_training_curves

def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config, config_path):
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def main(args):
    """
    Main function to run the experiment.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Default configuration
        config = {
            'dataset': 'mnist',
            'optimizer': 'adam_walign',
            'optimizers_to_compare': ['adam', 'sgd', 'adam_walign'],
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'align_threshold': 0.1,
            'caution_factor': 0.5,
            'epochs': 5,
            'batch_size': 64,
            'test_batch_size': 1000,
            'log_interval': 100,
            'save_dir': './models',
            'results_dir': './results',
            'data_dir': './data'
        }
        
        # Save default configuration
        os.makedirs('./config', exist_ok=True)
        save_config(config, './config/default_config.json')
    
    # Update configuration with command line arguments
    if args.dataset:
        config['dataset'] = args.dataset
    if args.optimizer:
        config['optimizer'] = args.optimizer
    if args.epochs:
        config['epochs'] = args.epochs
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Get data loaders
    print(f"Loading {config['dataset']} dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Train model
    print(f"Training model with {config['optimizer']} optimizer...")
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        config, train_loader, val_loader, device=device
    )
    
    # Plot training curves
    print("Plotting training curves...")
    plot_training_curves(
        train_losses, val_losses, train_accuracies, val_accuracies, 
        config['optimizer'], save_dir=config['results_dir']
    )
    
    # Evaluate and compare optimizers
    if args.compare or config.get('compare', False):
        print("Comparing optimizers...")
        # Train models with different optimizers if not already trained
        for optimizer_name in config['optimizers_to_compare']:
            if optimizer_name != config['optimizer']:
                print(f"\nTraining model with {optimizer_name} optimizer...")
                config_copy = config.copy()
                config_copy['optimizer'] = optimizer_name
                train_model(config_copy, train_loader, val_loader, device=device)
        
        # Evaluate optimizers
        results = evaluate_optimizers(config, test_loader, device=device)
        
        # Compare optimizers
        if results:
            compare_optimizers(results, save_dir=config['results_dir'])
    
    print("Experiment completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ADAM-WALIGN experiment')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], help='Dataset to use')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd', 'adam_walign'], help='Optimizer to use')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--compare', action='store_true', help='Compare optimizers')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    main(args)
