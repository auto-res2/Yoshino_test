# ADAM-WALIGN Optimizer

This repository contains the implementation and evaluation of the ADAM-WALIGN optimizer, which integrates key concepts from ADOPT and cautious optimization to enhance update stability and convergence integrity.

## Structure
- `.github/workflows`: Contains CI/CD configuration
- `config`: Configuration files for experiments
- `data`: Data used for model training and evaluation
- `models`: Pre-trained and trained models
- `paper`: Research paper related files
- `src`: Source code for the implementation
  - `train.py`: Scripts for training models
  - `evaluate.py`: Script to evaluate the model
  - `preprocess.py`: Script for preprocessing data
  - `main.py`: Scripts for running the experiment

## Installation
```bash
pip install -r requirements.txt
```

## Running the Experiment
```bash
python src/main.py
```
