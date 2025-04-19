import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

def build_skin_model(input_shape=(224, 224, 3), weights=None):
    """
    Build a skin cancer classification model based on EfficientNetB0.
    
    Args:
        input_shape: Input shape (height, width, channels)
        weights: Optional path to pre-trained weights
        
    Returns:
        model: Compiled Keras model
    """
    # Load EfficientNetB0 with pre-trained ImageNet weights as base model
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create model with custom classification head
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    
    # Add custom classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = models.Model(inputs, outputs, name="SkinCancerClassifier")
    
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

def fine_tune_skin_model(model, learning_rate=1e-5):
    """
    Fine-tune the EfficientNetB0 base model.
    
    Args:
        model: The pre-trained model
        learning_rate: Learning rate for fine-tuning
        
    Returns:
        model: Fine-tuned model
    """
    # Unfreeze the base model
    base_model = model.layers[1]
    base_model.trainable = True
    
    # Freeze the first several layers to prevent overfitting
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Recompile model with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Recall(name='sensitivity'),
            tf.keras.metrics.Precision(name='precision')
        ]
    )
    
    return model
