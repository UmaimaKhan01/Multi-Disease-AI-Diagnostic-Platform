import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils.gradcam import compute_gradcam, overlay_gradcam

def plot_image(img, title=None, cmap=None, figsize=(8, 8)):
    """Plot a single image."""
    plt.figure(figsize=figsize)
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = img.squeeze()
    plt.imshow(img, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_comparison(original, processed, titles=None, cmaps=None, figsize=(12, 6)):
    """Plot two images side-by-side for comparison."""
    if cmaps is None:
        cmaps = [None, None]
    if titles is None:
        titles = ['Original', 'Processed']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Handle grayscale images
    if len(original.shape) == 3 and original.shape[2] == 1:
        original = original.squeeze()
    if len(processed.shape) == 3 and processed.shape[2] == 1:
        processed = processed.squeeze()
    
    axes[0].imshow(original, cmap=cmaps[0])
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    
    axes[1].imshow(processed, cmap=cmaps[1])
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_gradcam_comparison(model, image, class_idx=None, layer_name=None, alpha=0.4, title=None):
    """Plot original image and Grad-CAM heatmap overlay side-by-side."""
    # Compute Grad-CAM
    heatmap = compute_gradcam(model, image, layer_name, class_idx)
    
    # Prepare visualizations
    if len(image.shape) == 4:  # 3D volume
        # Extract middle slice for visualization
        middle_slice_idx = image.shape[0] // 2
        image_for_display = image[middle_slice_idx, :, :, 0]
    else:
        image_for_display = image.squeeze() if image.shape[-1] == 1 else image
    
    # Create heatmap overlay
    overlay = overlay_gradcam(image_for_display, heatmap, alpha)
    
    # Create figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original image
    if len(image_for_display.shape) == 2 or (len(image_for_display.shape) == 3 and image_for_display.shape[2] == 1):
        axes[0].imshow(image_for_display, cmap='gray')
    else:
        axes[0].imshow(image_for_display)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Display heatmap overlay
    axes[1].imshow(overlay)
    axes[1].set_title('Grad-CAM Overlay')
    axes[1].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_multiple_gradcam_slices(model, volume, n_slices=3, class_idx=None, layer_name=None, alpha=0.4):
    """Plot Grad-CAM overlays for multiple slices of a 3D volume."""
    if len(volume.shape) != 4:
        raise ValueError("Expected a 3D volume with shape (depth, height, width, channels)")
    
    # Compute Grad-CAM (will average across depth dimension)
    heatmap = compute_gradcam(model, volume, layer_name, class_idx)
    
    # Select n_slices evenly spaced slices from the volume
    depth = volume.shape[0]
    slice_indices = np.linspace(0, depth-1, n_slices, dtype=int)
    
    # Create figure and subplots
    fig, axes = plt.subplots(2, n_slices, figsize=(n_slices*4, 8))
    
    for i, idx in enumerate(slice_indices):
        # Extract slice
        slice_img = volume[idx, :, :, 0]
        
        # Display original slice
        axes[0, i].imshow(slice_img, cmap='gray')
        axes[0, i].set_title(f'Slice {idx}')
        axes[0, i].axis('off')
        
        # Create and display heatmap overlay
        overlay = overlay_gradcam(slice_img, heatmap, alpha)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'Grad-CAM Slice {idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_training_history(history):
    """Plot training and validation metrics from Keras history object."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return fig
