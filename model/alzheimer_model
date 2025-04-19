import tensorflow as tf
from tensorflow.keras import layers, models

def build_alzheimer_model(input_shape=(64, 64, 64, 1), num_classes=4, weights=None):
    """
    Build a 3D CNN model for Alzheimer's disease classification.
    
    Args:
        input_shape: Input shape (depth, height, width, channels)
        num_classes: Number of output classes (4 for NonDemented, VeryMildDemented, MildDemented, ModerateDemented)
        weights: Optional path to pre-trained weights
        
    Returns:
        model: Compiled Keras model
    """
    # Create input layer
    inputs = layers.Input(shape=input_shape)
    
    # First 3D convolutional block
    x = layers.Conv3D(16, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 2))(x)
    
    # Second 3D convolutional block
    x = layers.Conv3D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 2))(x)
    
    # Third 3D convolutional block
    x = layers.Conv3D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 2))(x)
    
    # Fourth 3D convolutional block
    x = layers.Conv3D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 2))(x)
    
    # Global pooling and classification head
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer depends on number of classes
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'
    
    # Create model
    model = models.Model(inputs, outputs, name="AlzheimersClassifier")
    
    # Load weights if provided
    if weights:
        model.load_weights(weights)
    
    # Compile model with appropriate metrics
    metrics = ['accuracy']
    
    if num_classes == 2:
        metrics.extend([
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Recall(name='sensitivity'),
            tf.keras.metrics.Precision(name='precision')
        ])
    else:
        metrics.extend([
            tf.keras.metrics.AUC(name='auc', multi_label=True),
            tf.keras.metrics.Recall(name='sensitivity'),
            tf.keras.metrics.Precision(name='precision')
        ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=loss,
        metrics=metrics
    )
    
    return model

def multi_plane_fusion(coronal_model, sagittal_model, axial_model, weights=None):
    """
    Create a fusion model that combines predictions from three orthogonal planes.
    
    Args:
        coronal_model: Pre-trained model for coronal view
        sagittal_model: Pre-trained model for sagittal view
        axial_model: Pre-trained model for axial view
        weights: Optional path to pre-trained weights
        
    Returns:
        fusion_model: Compiled fusion model
    """
    # Feature layers from each model
    coronal_features = coronal_model.layers[-3].output  # Before the final classification layer
    sagittal_features = sagittal_model.layers[-3].output
    axial_features = axial_model.layers[-3].output
    
    # Create fusion model
    concatenated = layers.concatenate([coronal_features, sagittal_features, axial_features])
    x = layers.Dense(256, activation='relu')(concatenated)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    
    # Create fusion model
    fusion_model = models.Model(
        inputs=[coronal_model.input, sagittal_model.input, axial_model.input],
        outputs=outputs
    )
    
    # Load weights if provided
    if weights:
        fusion_model.load_weights(weights)
    
    # Compile model
    fusion_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc', multi_label=True)
        ]
    )
    
    return fusion_model
