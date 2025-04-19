import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report, precision_score, recall_score

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Calculate Dice coefficient for segmentation tasks."""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    """Calculate IoU (Intersection over Union) score for segmentation tasks."""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def evaluate_classification(y_true, y_pred, y_prob=None, classes=None):
    """Evaluate classification model and print metrics."""
    if classes is None:
        if len(np.unique(y_true)) == 2:
            classes = ['Negative', 'Positive']
        else:
            classes = [f'Class {i}' for i in range(len(np.unique(y_true)))]
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    # Handle F1 for multi-class
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # ROC AUC - handle multi-class and binary differently
    if y_prob is not None:
        try:
            if y_prob.ndim > 1 and y_prob.shape[1] > 1:  # multi-class
                auc = roc_auc_score(np.eye(len(classes))[y_true], y_prob, multi_class='ovr')
            else:  # binary
                prob_values = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                auc = roc_auc_score(y_true, prob_values)
        except Exception as e:
            print(f"Unable to calculate AUC: {e}")
            auc = np.nan
    else:
        auc = np.nan
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Precision and recall
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    # Print results
    print("\n--- Classification Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    if not np.isnan(auc):
        print(f"AUC: {auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    # Return metrics for logging or further use
    metrics = {
        'accuracy': acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }
    
    return metrics

def evaluate_segmentation(y_true, y_pred):
    """Evaluate segmentation model and print metrics."""
    # Calculate metrics
    dice = dice_coefficient(y_true, y_pred)
    iou = iou_score(y_true, y_pred)
    
    # Print results
    print("\n--- Segmentation Metrics ---")
    print(f"Dice Coefficient: {dice:.4f}")
    print(f"IoU Score: {iou:.4f}")
    
    # Return metrics for logging or further use
    metrics = {
        'dice': dice,
        'iou': iou
    }
    
    return metrics
