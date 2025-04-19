#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train and evaluate lung cancer detection model.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lung_model import build_lung_model
from utils.data_loader import load_lung_data, get_training_augmentation
from utils.metrics import evaluate_classification
from utils.visualization import plot_training_history, plot_gradcam_comparison

def parse_args():
    parser = argparse.ArgumentParser(description='Train lung cancer detection model')
    parser.add_argument('--data_dir', type=str, default='../data/lung',
                        help='Path to lung cancer dataset directory')
    parser.add_argument('--output_dir', type=str, default='../models/saved',
                        help='Directory to save model and results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for input')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load and preprocess data
    print("Loading lung cancer dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_lung_data(
        args.data_dir, 
        img_size=(args.img_size, args.img_size)
    )
    
    print(f"Train set: {X_train.shape} | {y_train.shape}")
    print(f"Val set: {X_val.shape} | {y_val.shape}")
    print(f"Test set: {X_test.shape} | {y_test.shape}")
    
    # Get data generator for augmentation
    datagen = get_training_augmentation()
