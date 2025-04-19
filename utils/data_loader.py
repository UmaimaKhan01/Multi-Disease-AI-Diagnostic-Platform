import os
import numpy as np
import cv2
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Utility: CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def segment_lungs(image):
    """
    Segment lungs from chest X-ray using basic thresholding.
    A full implementation would use a pretrained U-Net.
    """
    # This is a simplified placeholder
    # In a real implementation, use a pretrained U-Net for actual lung segmentation
    _, binary = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for the largest contours (excluding the background)
    mask = np.zeros_like(image)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[1:3]:  # Assuming lungs are 2nd and 3rd largest
        cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Apply mask to original image
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return segmented

def load_covid_data(data_dir, img_size=(224,224)):
    """Load COVID-19 X-ray images and labels."""
    classes = ['normal', 'covid']  # assume subfolders
    X, y = [], []
    for label in classes:
        folder = os.path.join(data_dir, label)
        for fname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            # Apply CLAHE and histogram normalization
            img = clahe.apply(img)
            img = cv2.equalizeHist(img)
            # Lung segmentation
            img_masked = segment_lungs(img)
            X.append(img_masked[..., np.newaxis])  # add channel dim
            y.append(0 if label=='normal' else 1)
    X = np.array(X)/255.0
    y = np.array(y, dtype=np.int32)
    # Split into train/val/test (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_skin_data(data_dir, img_size=(224,224)):
    """Load Skin Cancer images (HAM10000). Assume data_dir has subfolders 'benign' and 'malignant'."""
    X, y = [], []
    for label in ['benign', 'malignant']:
        folder = os.path.join(data_dir, label)
        for fname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            # Apply CLAHE to each channel (convert to LAB for better effect on luminance)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            # Hair removal (morphological filtering)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
            blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
            _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
            img[np.where(mask==255)] = [0,0,0]
            X.append(img)
            y.append(0 if label=='benign' else 1)
    X = np.array(X, dtype=np.float32)/255.0
    y = np.array(y, dtype=np.int32)
    # Train/val/test split...
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_brain_data(data_dir, img_size=(224,224)):
    """Load Brain MRI slices for tumor vs no-tumor classification."""
    # Assume data_dir has subfolders 'tumor' and 'healthy', each containing 2D slice images (.jpg or .png).
    X, y = [], []
    for label in ['healthy', 'tumor']:
        folder = os.path.join(data_dir, label)
        for fname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            # Optional skull stripping: assume images already skull-stripped or apply threshold to remove background
            _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
            img = cv2.bitwise_and(img, img, mask=thresh)  # simple removal of dark background
            X.append(img[..., np.newaxis])
            y.append(0 if label=='healthy' else 1)
    X = np.array(X)/255.0
    y = np.array(y, dtype=np.int32)
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_alzheimer_data(data_dir, target_shape=(64,64,64)):
    """Load Alzheimer MRI volumes (.nii files) and labels."""
    # Assume subfolders 'NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented'
    X, y = [], []
    classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    for ci, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        for fname in os.listdir(folder):
            if fname.endswith(('.nii', '.nii.gz', '.mgz')):
                try:
                    img_nii = nib.load(os.path.join(folder, fname))
                    vol = img_nii.get_fdata()
                    # N4 bias field correction would go here using SimpleITK
                    # skull stripping would go here (assume volumes pre-stripped for simplicity)
                    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)  # normalize 0-1
                    # Resize to target shape
                    vol_resized = np.zeros(target_shape)
                    # Get central slices if volume is larger than target
                    s_i, s_j, s_k = vol.shape
                    t_i, t_j, t_k = target_shape
                    # Copy the central portion
                    i_offset = max(0, s_i//2 - t_i//2)
                    j_offset = max(0, s_j//2 - t_j//2)
                    k_offset = max(0, s_k//2 - t_k//2)
                    # Get min dimensions
                    i_end = min(s_i, i_offset + t_i)
                    j_end = min(s_j, j_offset + t_j)
                    k_end = min(s_k, k_offset + t_k)
                    # Copy the volume (portion that fits)
                    i_copy = min(t_i, i_end - i_offset)
                    j_copy = min(t_j, j_end - j_offset)
                    k_copy = min(t_k, k_end - k_offset)
                    vol_resized[:i_copy, :j_copy, :k_copy] = vol[i_offset:i_offset+i_copy, 
                                                                j_offset:j_offset+j_copy, 
                                                                k_offset:k_offset+k_copy]
                    X.append(vol_resized[..., np.newaxis].astype(np.float32))
                    y.append(ci)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
                    continue
    X = np.array(X)
    y = np.array(y, dtype=np.int32)
    # Split into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_lung_data(data_dir, img_size=(224,224)):
    """Load Lung Cancer X-ray images (binary classification)."""
    # Assume subfolders 'normal' and 'cancer'
    X, y = [], []
    for label in ['normal', 'cancer']:
        folder = os.path.join(data_dir, label)
        for fname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            # CLAHE for contrast enhancement
            img = clahe.apply(img)
            # Median filtering to remove noise
            img = cv2.medianBlur(img, 3)
            X.append(img[..., np.newaxis])
            y.append(0 if label=='normal' else 1)
    X = np.array(X)/255.0
    y = np.array(y, dtype=np.int32)
    # Split into train/val/test
    X_train, X_temp, y_train, y
