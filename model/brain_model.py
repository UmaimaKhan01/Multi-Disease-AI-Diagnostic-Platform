import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

def spatial_attention(x):
    """
    Soft spatial attention module to focus on important regions.
    
    Args:
        x: Input feature maps
        
    Returns:
        x_attn: Feature maps with spatial attention applied
    """
    # Average pooling across channels
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    # Max pooling across channels
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    
    # Concatenate pooled features
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    
    # Create attention map with convolution
    attn_map = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    
    # Apply attention map to input feature maps
    x_attn = layers.multiply([x, attn_map])
    
    return x_attn

def build_brain_model(input_shape=(224, 224, 1), weights=None):
    """
    Build a brain tumor detection model using VGG16 with spatial attention.
    
    Args:
        input_shape: Input shape (height, width, channels)
        weights: Optional path to pre-trained weights
        
    Returns:
        model: Compiled Keras model
    """
    # Create input layer
    inputs = layers.Input(shape=input_shape)
    
    # Convert grayscale to RGB if needed (for VGG16 compatibility)
    if input_shape[-1] == 1:
        x = layers.Concatenate()([inputs, inputs, inputs])  # Duplicate grayscale to RGB
    else:
        x = inputs
    
    # Load VGG16 with pre-trained ImageNet weights
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Apply base model
    x = base_model(x, training=False)
    
    # Apply spatial attention
    x = spatial_attention(x)
    
    # Global pooling and classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = models.Model(inputs, outputs, name="BrainTumorDetector")
    
    # Load weights if provided
    if weights:
        model.load_weights(weights)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Recall(name='sensitivity'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.MeanIoU(num_classes=2, name='iou')
        ]
    )
    
    return model

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calculate Dice coefficient for evaluation.
    
    Args:
        y_true: Ground truth
        y_pred: Predictions
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        dice: Dice coefficient
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    return dice
