#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train and evaluate skin cancer classification model.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.skin_model import build_skin_model, fine_tune_skin_model
from utils.data_loader import load_skin_data, get_training_augmentation
from utils.metrics import evaluate_classification
from utils.visualization import plot_training_history, plot_gradcam_comparison

def parse_args():
    parser = argparse.ArgumentParser(description='Train skin cancer classification model')
    parser.add_argument('--data_dir', type=str, default='../data/skin',
                        help='Path to skin cancer dataset directory')
    parser.add_argument('--output_dir', type=str, default='../models/saved',
                        help='Directory to save model and results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--fine_tune_epochs', type=int, default=20,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for input')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5,
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
    print("Loading skin cancer dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_skin_data(
        args.data_dir, 
        img_size=(args.img_size, args.img_size)
    )
    
    print(f"Train set: {X_train.shape} | {y_train.shape}")
    print(f"Val set: {X_val.shape} | {y_val.shape}")
    print(f"Test set: {X_test.shape} | {y_test.shape}")
    
    # Get data generator for augmentation
    datagen = get_training_augmentation()
    
    # Build model
    print("Building skin cancer classification model...")
    model = build_skin_model(input_shape=(args.img_size, args.img_size, 3))
    model.summary()
    
    # Set up callbacks
    callbacks = []
    if args.early_stopping:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
    
    # Model checkpoint
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(args.output_dir, 'skin_model_checkpoint.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True
    )
    callbacks.append(model_checkpoint)
    
    # Train model (first phase - frozen base model)
    print(f"Training model for {args.epochs} epochs...")
    history1
