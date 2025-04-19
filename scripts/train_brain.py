#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train and evaluate brain tumor detection model.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.brain_model import build_brain_model
from utils.data_loader import load_brain_data, get_training_augmentation
from utils.metrics import evaluate_classification
from utils.visualization import plot_training_history, plot_gradcam_comparison

def parse_args():
    parser = argparse.ArgumentParser(description='Train brain tumor detection model')
    parser.add_argument('--data_dir', type=str, default='../data/brain',
                        help='Path to brain tumor dataset directory')
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
    print("Loading brain tumor dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_brain_data(
        args.data_dir, 
        img_size=(args.img_size, args.img_size)
    )
    
    print(f"Train set: {X_train.shape} | {y_train.shape}")
    print(f"Val set: {X_val.shape} | {y_val.shape}")
    print(f"Test set: {X_test.shape} | {y_test.shape}")
    
    # Get data generator for augmentation
    datagen = get_training_augmentation()
    
    # Build model
    print("Building brain tumor detection model...")
    model = build_brain_model(input_shape=(args.img_size, args.img_size, 1))
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
        os.path.join(args.output_dir, 'brain_model_checkpoint.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True
    )
    callbacks.append(model_checkpoint)
    
    # Train model
    print(f"Training model for {args.epochs} epochs...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=args.batch_size),
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    # Plot training history
    print("Plotting training history...")
    fig = plot_training_history(history)
    fig.savefig(os.path.join(args.output_dir, 'brain_training_history.png'))
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).squeeze()
    
    # Calculate and print metrics
    metrics = evaluate_classification(
        y_test, 
        y_pred, 
        y_pred_prob, 
        classes=['Healthy', 'Tumor']
    )
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'brain_metrics.txt'), 'w') as f:
        f.write("Brain Tumor Detection Model Metrics\n")
        f.write("==================================\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall (Sensitivity): {metrics['recall']:.4f}\n")
        f.write(f"AUC: {metrics['auc']:.4f}\n")
    
    # Save model
    print("Saving model...")
    model.save(os.path.join(args.output_dir, 'brain_model.h5'))
    
    # Visualize Grad-CAM for a few examples
    print("Generating Grad-CAM visualizations...")
    for i in range(min(5, len(X_test))):
        fig = plot_gradcam_comparison(
            model, 
            X_test[i], 
            title=f"Brain Tumor Detection Grad-CAM (True: {y_test[i]}, Pred: {y_pred[i]})"
        )
        fig.savefig(os.path.join(args.output_dir, f'brain_gradcam_example_{i}.png'))
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()
