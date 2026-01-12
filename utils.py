import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# --- SETTINGS ---
TARGET_CLASSES = [
    "Apple", "Banana", "Orange", "Mango", "Grapes",
    "Pineapple", "Watermelon", "Pomegranate", "Strawberry", "Lemon"
]
IMG_SIZE = (96, 96)
CACHE_DIR = "features_cache"


def get_fruit_dataset(split='train'):
    """
    Loads and filters the fruit-recognition dataset from Hugging Face.
    """
    dataset = load_dataset("ysif9/fruit-recognition")
    # Filter Classes
    all_class_names = dataset['train'].features['label'].names
    target_ids = [all_class_names.index(name) for name in TARGET_CLASSES if name in all_class_names]
    dataset = dataset.filter(lambda example: example['label'] in target_ids)

    return dataset[split], all_class_names


def get_test_images():
    """
    Returns the test dataset images and labels for visualization purposes.
    Returns: ds_test (iterable of examples), all_class_names
    """
    return get_fruit_dataset('test')


def get_data_and_extract_features():
    """
    Loads fruit recognition dataset, extracts MobileNetV2 features, and caches them.
    Returns: X_train, y_train, X_test, y_test, encoded_target_names
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    paths = {
        'X_train': os.path.join(CACHE_DIR, 'X_train.npy'),
        'y_train': os.path.join(CACHE_DIR, 'y_train.npy'),
        'X_test': os.path.join(CACHE_DIR, 'X_test.npy'),
        'y_test': os.path.join(CACHE_DIR, 'y_test.npy'),
        'classes': os.path.join(CACHE_DIR, 'classes.npy')
    }

    # Check cache
    if all(os.path.exists(p) for p in paths.values()):
        print("Loading features from cache...")
        X_train = np.load(paths['X_train'])
        y_train = np.load(paths['y_train'])
        X_test = np.load(paths['X_test'])
        y_test = np.load(paths['y_test'])
        encoded_target_names = np.load(paths['classes'], allow_pickle=True).tolist()

        return X_train, y_train, X_test, y_test, encoded_target_names

    print("Cache not found. Loading Dataset from Hugging Face...")
    ds_train, all_class_names = get_fruit_dataset('train')
    ds_test, _ = get_fruit_dataset('test')

    print("Loading MobileNetV2 (Pre-trained on ImageNet)...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                             pooling='avg')

    def process_and_extract(split_name):
        print(f"Processing {split_name} data...")
        images = []
        labels = []
        if split_name == 'train':
            ds_split = ds_train
        else:
            ds_split = ds_test

        raw_images_batch = []
        label_batch = []

        for i, example in enumerate(ds_split):
            img = example['image']
            if img.mode != 'RGB': img = img.convert('RGB')
            img = img.resize(IMG_SIZE)

            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)

            raw_images_batch.append(img_array)
            label_batch.append(example['label'])

            # Predict in batches
            if len(raw_images_batch) >= 500:
                batch_arr = np.array(raw_images_batch)
                features = base_model.predict(batch_arr, verbose=0)
                images.append(features)
                labels.extend(label_batch)
                raw_images_batch = []
                label_batch = []
                print(f"  Processed {i + 1}/{len(ds_split)} images...", end='\r')

        if raw_images_batch:
            batch_arr = np.array(raw_images_batch)
            features = base_model.predict(batch_arr, verbose=0)
            images.append(features)
            labels.extend(label_batch)

        X = np.vstack(images)
        y = np.array(labels)
        return X, y

    X_train_raw, y_train_raw = process_and_extract('train')
    X_test_raw, y_test_raw = process_and_extract('test')

    # Get original names for the raw IDs
    y_train_names = [all_class_names[label_id] for label_id in y_train_raw]
    y_test_names = [all_class_names[label_id] for label_id in y_test_raw]

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_names)
    y_test_encoded = le.transform(y_test_names)
    encoded_target_names = list(le.classes_)

    # Save to cache
    np.save(paths['X_train'], X_train_raw)
    np.save(paths['y_train'], y_train_encoded)
    np.save(paths['X_test'], X_test_raw)
    np.save(paths['y_test'], y_test_encoded)
    np.save(paths['classes'], np.array(encoded_target_names))

    print(f"\nSaved features to {CACHE_DIR}")
    return X_train_raw, y_train_encoded, X_test_raw, y_test_encoded, encoded_target_names


def evaluate_model(model, X_test, y_test, class_names, model_name="Model"):
    """
    Predicts using the model, prints accuracy & classification report, and plots confusion matrix.
    """
    print(f"\n=== Evaluation: {model_name} ===")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    print(f"Accuracy: {acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()


def visualize_model_errors(model, X_test, y_test, class_names, num_samples=5):
    """
    Identifies misclassified images and visualizes them.
    """
    print(f"\n=== Error Analysis: {num_samples} Misclassified Samples ===")
    y_pred = model.predict(X_test)

    # Find indices of errors
    error_indices = np.where(y_pred != y_test)[0]

    if len(error_indices) == 0:
        print("No errors found on the test set!")
        return

    # Randomly select a few errors
    selected_indices = np.random.choice(error_indices, size=min(len(error_indices), num_samples), replace=False)

    # Load test dataset for visualization
    ds_test, _ = get_test_images()

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(selected_indices):
        true_label_idx = y_test[idx]
        pred_label_idx = y_pred[idx]

        true_label = class_names[true_label_idx]
        pred_label = class_names[pred_label_idx]

        # Get image from dataset
        img_data = ds_test[int(idx)]['image']  # idx is numpy int

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img_data)
        plt.title(f"True: {true_label}\nPred: {pred_label}", color='red')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
