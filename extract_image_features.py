import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator # New import for augmentation
from tqdm import tqdm # For progress bar

print("Script started: Extracting image features with augmentation for training set...")

# --- Configuration (ensure consistency with prepare_odir_data.py) ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
TARGET_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# --- Define the core single image feature extraction logic ---
def _extract_single_image_features(img_array_processed, feature_extractor_model):
    """
    Extracts features from a single, pre-processed image array.
    """
    try:
        features = feature_extractor_model.predict(img_array_processed, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None

# --- Modified feature extraction function to include augmentation ---
def extract_image_features_with_augmentation(dataframe, feature_extractor_model, img_height, img_width, is_training_set=False, augment_factor=5):
    """
    Extracts features from images in the dataframe, with optional augmentation for training set.
    For augmented images, it creates 'augment_factor' new feature vectors per original image.
    """
    all_patient_ids = []
    all_image_features = []
    all_original_image_filenames = [] # To link back to original image in dataframe

    # Setup ImageDataGenerator only if it's the training set and augmentation is requested
    datagen = None
    if is_training_set and augment_factor > 0:
        datagen = ImageDataGenerator(
            rotation_range=20,     # Rotate images by up to 20 degrees
            width_shift_range=0.1, # Shift images horizontally by up to 10%
            height_shift_range=0.1,# Shift images vertically by up to 10%
            shear_range=0.1,       # Apply shear transformations
            zoom_range=0.1,        # Apply random zooms
            horizontal_flip=True,  # Randomly flip images horizontally
            fill_mode='nearest'    # Fill newly created pixels after rotation/shift
        )
        print(f"Applying data augmentation (factor: {augment_factor}) for training images.")
    else:
        print("No data augmentation for this set.")

    # Removed image_base_path as it's no longer a parameter
    print(f"Processing {len(dataframe)} images...")
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Extracting Image Features"):
        patient_id = row['patient_id']
        image_filename = row['image_path'] # Assuming 'Image_Path' column holds the filename

        img_path = image_filename

        # 1. Process Original Image
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image at {img_path}. Skipping.")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for Keras
            img_resized = cv2.resize(img_rgb, (img_width, img_height))
            img_array_original = np.expand_dims(img_resized, axis=0) # Add batch dimension
            img_array_original_processed = preprocess_input(img_array_original)

            features_original = _extract_single_image_features(img_array_original_processed, feature_extractor_model)
            if features_original is not None:
                all_patient_ids.append(patient_id)
                all_image_features.append(features_original)
                all_original_image_filenames.append(image_filename)

        except Exception as e:
            print(f"Error processing original image {img_path}: {e}")
            continue

        # 2. Process Augmented Images (only for training set and if augmentation is enabled)
        if is_training_set and datagen:
            i = 0
            # Flow from a numpy array to generate 'augment_factor' augmented images
            # This generates augmented images in memory
            for batch in datagen.flow(img_array_original, batch_size=1, shuffle=False):
                if i >= augment_factor:
                    break # Stop after generating 'augment_factor' augmented images

                augmented_img_array = batch[0] # Get the single augmented image from the batch
                
                # Preprocess the augmented image for ResNet
                augmented_img_array_processed = preprocess_input(np.expand_dims(augmented_img_array, axis=0))

                features_augmented = _extract_single_image_features(augmented_img_array_processed, feature_extractor_model)
                if features_augmented is not None:
                    all_patient_ids.append(patient_id) # Link augmented features to original patient ID
                    all_image_features.append(features_augmented)
                    all_original_image_filenames.append(image_filename) # Keep track of original image source
                i += 1

    # Create DataFrame from extracted features
    feature_cols = [f'img_feat_{i}' for i in range(feature_extractor_model.output_shape[1])]
    image_features_df = pd.DataFrame(all_image_features, columns=feature_cols)
    
    # Add patient IDs and original image filenames
    image_features_df['ID'] = all_patient_ids
    image_features_df['Original_Image_Path'] = all_original_image_filenames # Keep track of original image path

    return image_features_df

# --- Main script execution ---
if __name__ == "__main__":
    # Define paths
    DATA_DIR = 'data/ODIR-5K/ODIR-5K'
    TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, 'Training Images')
    TEST_IMAGE_DIR = os.path.join(DATA_DIR, 'Testing Images')

    # Load the base ResNet50 model for feature extraction
    print("Loading ResNet50 model for feature extraction...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    feature_extractor = Model(inputs=base_model.input,
                              outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))
    print("ResNet50 model loaded.")

    # --- Load DataFrames ---
    # Load training and testing patient info CSVs (from prepare_odir_data.py)
    try:
        train_df = pd.read_csv('train_patients_df.csv')
        test_df = pd.read_csv('test_patients_df.csv')
        print("Training and testing patient dataframes loaded.")
    except FileNotFoundError:
        print("Error: train_patients_df.csv or test_patients_df.csv not found.")
        print("Please ensure you have run prepare_odir_data.py first to create these files.")
        exit()

    # --- Extract features for training set (with augmentation) ---
    print("\n--- Extracting features for Training Set (with Augmentation) ---")
    train_image_features_df = extract_image_features_with_augmentation(
        train_df,
        feature_extractor,
        IMG_HEIGHT,
        IMG_WIDTH,
        is_training_set=True,
        augment_factor=5
    )

    print(f"Extracted {len(train_image_features_df)} feature vectors for the training set (original + augmented).")

    # --- Extract features for testing set (NO augmentation) ---
    print("\n--- Extracting features for Testing Set (NO Augmentation) ---")
    test_image_features_df = extract_image_features_with_augmentation(
        test_df,
        feature_extractor,
        IMG_HEIGHT,
        IMG_WIDTH,
        is_training_set=False,
        augment_factor=0
    )

    print(f"Extracted {len(test_image_features_df)} feature vectors for the testing set.")

    # --- Save extracted features ---
    # These CSVs will be loaded by train_multi_class_model.py
    train_image_features_df.to_csv('train_image_features_augmented.csv', index=False)
    test_image_features_df.to_csv('test_image_features.csv', index=False)

    print("\nImage feature extraction complete. Features saved to CSV files.")