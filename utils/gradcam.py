import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt

def compute_gradcam(model, image, layer_name=None, class_idx=None):
    """
    Compute Grad-CAM heatmap for a given model and image.
    
    Args:
        model: The trained model
        image: Input image (should have shape expected by model, excluding batch dimension)
        layer_name: Name of the layer to compute Grad-CAM on (default: last conv layer)
        class_idx: Index of the class to compute Grad-CAM for (default: highest predicted class)
        
    Returns:
        heatmap: Grad-CAM heatmap (same spatial dimensions as image)
    """
    # Add batch dimension if needed
    if len(image.shape) == 3 and image.shape[-1] in [1, 3]:
        # 2D image
        img_tensor = np.expand_dims(image, axis=0)
    elif len(image.shape) == 4 and image.shape[-1] == 1:
        # 3D volume with single channel
        img_tensor = np.expand_dims(image, axis=0)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    
    # Find last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Conv3D)):
                layer_name = layer.name
                break
    
    # Get model output and extract the target class
    preds = model.predict(img_tensor)
    if class_idx is None:
        if len(preds.shape) == 2:  # Multi-class output
            class_idx = np.argmax(preds[0])
        else:  # Binary output
            class_idx = 0  # Just use the single output
    
    # Create a model that maps the input image to the activations of the target layer
    # and to the model outputs
    grad_model = models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if len(predictions.shape) == 2:  # Multi-class
            loss = predictions[:, class_idx]
        else:  # Binary
            loss = predictions
    
    # Extract gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Handle 3D case by taking average over depth dimension
    if len(conv_outputs.shape) == 5:  # 3D case: (batch, d, h, w, channels)
        # Average over depth dimension for visualization
        conv_outputs = tf.reduce_mean(conv_outputs, axis=1)
        grads = tf.reduce_mean(grads, axis=1)
    
    # Global average pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the gradient values
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Apply ReLU to the heatmap
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
    
    # Resize heatmap to match image dimensions
    if len(image.shape) == 3:  # 2D image
        target_size = image.shape[:2]
    else:  # 3D volume - we'll only visualize a 2D slice
        # For 3D data, we need to decide which 2D slice to visualize
        target_size = image.shape[1:3]  # Assuming (d, h, w, c) format
    
    heatmap = cv2.resize(heatmap.numpy(), target_size)
    
    return heatmap

def overlay_gradcam(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on the original image for visualization.
    
    Args:
        image: Original image
        heatmap: Grad-CAM heatmap
        alpha: Transparency factor for overlay
        colormap: OpenCV colormap for heatmap
        
    Returns:
        superimposed_img: Image with overlaid heatmap
    """
    # Ensure heatmap is in range [0, 255]
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap to create visualization
    colored_heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        image = np.squeeze(image)
        image = cv2.cvtColor(np.uint8(image * 255), cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        image = np.uint8(image * 255)
    else:
        # For 3D volumes, we should extract a 2D slice first
        raise ValueError("Unexpected image format. For 3D volumes, extract a 2D slice before overlay.")
    
    # Superimpose the heatmap on the image
    superimposed_img = cv2.addWeighted(image, 1-alpha, colored_heatmap, alpha, 0)
    
    return superimposed_img

def visualize_3d_gradcam(model, volume, slices_indices=None, class_idx=None, layer_name=None, alpha=0.4):
    """
    Compute and visualize Grad-CAM for a 3D volume across multiple slices.
    
    Args:
        model: The trained 3D model
        volume: Input 3D volume (d, h, w, c)
        slices_indices: List of slice indices to visualize
        class_idx: Index of the class to compute Grad-CAM for
        layer_name: Name of the layer to compute Grad-CAM on
        alpha: Transparency factor for overlay
        
    Returns:
        fig: Matplotlib figure containing the visualizations
    """
    # Compute Grad-CAM
    heatmap_3d = compute_gradcam(model, volume, layer_name, class_idx)
    
    # If slice indices not provided, select evenly spaced slices
    if slices_indices is None:
        depth = volume.shape[0]
        slices_indices = [depth // 4, depth // 2, (3 * depth) // 4]
    
    # Set up the plot
    num_slices = len(slices_indices)
    fig, axes = plt.subplots(2, num_slices, figsize=(4 * num_slices, 8))
    
    for i, slice_idx in enumerate(slices_indices):
        # Extract slice from volume
        slice_img = volume[slice_idx, :, :, 0]
        
        # Display original slice
        axes[0, i].imshow(slice_img, cmap='gray')
        axes[0, i].set_title(f'Slice {slice_idx}')
        axes[0, i].axis('off')
        
        # Extract corresponding heatmap
        slice_heatmap = heatmap_3d[slice_idx]
        
        # Overlay heatmap
        overlay = overlay_gradcam(slice_img, slice_heatmap, alpha)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'GradCAM Slice {slice_idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig
