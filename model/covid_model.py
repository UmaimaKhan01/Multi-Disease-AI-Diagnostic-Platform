import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def se_block(x, reduction=16):
    """
    Squeeze-and-Excitation (SE) block for channel-wise attention.
    
    Args:
        x: Input feature maps
        reduction: Reduction ratio for the bottleneck
        
    Returns:
        x: Feature maps with channel attention applied
    """
    # Get number of channels/filters
    channels = x.shape[-1]
    
    # Squeeze: Global average pooling
    se = layers.GlobalAveragePooling2D()(x)
    
    # Excitation: Two FC layers with bottleneck
    se = layers.Reshape((1, 1, channels))(se)
    se = layers.Dense(channels // reduction, activation='relu', 
                     kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(channels, activation='sigmoid', 
                     kernel_initializer='he_normal', use_bias=False)(se)
    
    # Scale the input feature maps
    x = layers.Multiply()([x, se])
    
    return x

def build_covid_model(input_shape=(224, 224, 1), weights=None):
    """
    Build a COVID-19 detection model with Squeeze-and-Excitation attention.
    
    Args:
        input_shape: Input shape (height, width, channels)
        weights: Optional path to pre-trained weights
        
    Returns:
        model: Compiled Keras model
    """
    # Create input layer
    inputs = layers.Input(shape=input_shape)
    
    # First convolutional block with SE attention
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = se_block(x)
    
    # Second convolutional block with SE attention
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = se_block(x)
    
    # Third convolutional block with SE attention
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = se_block(x)
    
    # Fourth convolutional block with SE attention
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = se_block(x)
    
    # Global average pooling and dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Binary classification output
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = models.Model(inputs, outputs, name="COVID19Detector")
    
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
            tf.keras.metrics.Precision(name='precision')
        ]
    )
    
    return model
