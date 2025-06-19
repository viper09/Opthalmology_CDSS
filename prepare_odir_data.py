import pandas as pd
import numpy as np
import os
import cv2 # OpenCV for image loading
import matplotlib.pyplot as plt

# --- Configuration ---
# Adjust this path based on where you extracted the ODIR-5K dataset
DATA_DIR = 'data/ODIR-5K/ODIR-5K'
# Assuming the annotation file is an .xlsx and named something like 'data.xlsx' or 'ODIR-5K.xlsx'
# Please verify the exact name of your Excel file within the ODIR-5K folder
EXCEL_FILE = os.path.join(DATA_DIR, 'data.xlsx') # <--- **ADJUST THIS FILENAME IF YOURS IS DIFFERENT!**

# Image directories are now subfolders within DATA_DIR, WITH CORRECTED NAMES
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, 'Training Images')
TEST_IMAGE_DIR = os.path.join(DATA_DIR, 'Testing Images')


print(f"Loading data from: {EXCEL_FILE}")
print(f"Looking for training images in: {TRAIN_IMAGE_DIR}")
print(f"Looking for testing images in: {TEST_IMAGE_DIR}")

# --- 1. Load Tabular Data (from Excel) ---
try:
    # Changed from pd.read_csv to pd.read_excel
    df_raw = pd.read_excel(EXCEL_FILE)
    print("\nTabular data loaded successfully from Excel.")
    print("Shape:", df_raw.shape)
    print("\nFirst 5 rows of tabular data:")
    print(df_raw.head())
    print("\nInfo of tabular data:")
    df_raw.info()
except FileNotFoundError:
    print(f"Error: Excel file not found at {EXCEL_FILE}. Please check your DATA_DIR and EXCEL_FILE path.")
    exit()
except Exception as e:
    print(f"Error reading Excel file: {e}. Ensure you have 'openpyxl' installed (pip install openpyxl).")
    exit()

# --- 2. Initial Data Cleaning and Feature Selection for Tabular Data ---
# ODIR-5k's Excel usually has columns like 'ID', 'Patient Age', 'Patient Sex',
# 'Left-Fundus', 'Right-Fundus', 'Left-Diagnostic Keywords', 'Right-Diagnostic Keywords'
# We are interested in 'Patient Age', 'Patient Sex', and the diagnostic keywords.

# Function to map diagnostic keywords to our target classes
def map_diagnosis(keywords):
    keywords = str(keywords).lower()
    if 'normal' in keywords:
        return 'Normal'
    elif 'diabetic retinopathy' in keywords or 'dr' in keywords:
        return 'Diabetes'
    elif 'glaucoma' in keywords:
        return 'Glaucoma'
    elif 'cataract' in keywords:
        return 'Cataract'
    else:
        # For cases that don't fit our 4 main classes or are complex
        # We might exclude these or categorize them as 'Other'
        return 'Other' # Or np.nan if we want to drop them

df = df_raw.copy()

# Rename columns for easier access if they are different in your Excel file
# Common ODIR-5k column names from .xlsx:
df = df.rename(columns={
    'Patient Age': 'age',
    'Patient Sex': 'sex',
    'Left-Fundus': 'left_fundus',
    'Right-Fundus': 'right_fundus',
    'Left-Diagnostic Keywords': 'left_diagnostic_keywords',
    'Right-Diagnostic Keywords': 'right_diagnostic_keywords'
})

# Apply diagnosis mapping
df['left_diagnosis'] = df['left_diagnostic_keywords'].apply(map_diagnosis)
df['right_diagnosis'] = df['right_diagnostic_keywords'].apply(map_diagnosis)

# Filter out 'Other' diagnoses for now
df = df[df['left_diagnosis'] != 'Other']
df = df[df['right_diagnosis'] != 'Other']

print("\nDiagnosis distribution for left eye:")
print(df['left_diagnosis'].value_counts())
print("\nDiagnosis distribution for right eye:")
print(df['right_diagnosis'].value_counts())

# Create a row for each eye, including constructing correct image paths
records = []
for index, row in df.iterrows():
    # Helper to find the correct image path (in training or testing)
    def find_image_path(image_filename):
        # Check in training folder first
        path_train = os.path.join(TRAIN_IMAGE_DIR, image_filename)
        if os.path.exists(path_train):
            return path_train
        # Then check in testing folder
        path_test = os.path.join(TEST_IMAGE_DIR, image_filename)
        if os.path.exists(path_test):
            return path_test
        return None # Image not found in either

    # Left eye record
    left_img_path = find_image_path(row['left_fundus'])
    if left_img_path: # Only add if image path is found
        records.append({
            'patient_id': row['ID'], # Assuming 'ID' column for patient ID
            'age': row['age'],
            'sex': row['sex'],
            'eye_side': 'left',
            'image_path': left_img_path,
            'diagnosis': row['left_diagnosis']
        })
    else:
        print(f"Warning: Left image not found for ID {row['ID']}: {row['left_fundus']}")

    # Right eye record
    right_img_path = find_image_path(row['right_fundus'])
    if right_img_path: # Only add if image path is found
        records.append({
            'patient_id': row['ID'],
            'age': row['age'],
            'sex': row['sex'],
            'eye_side': 'right',
            'image_path': right_img_path,
            'diagnosis': row['right_diagnosis']
        })
    else:
        print(f"Warning: Right image not found for ID {row['ID']}: {row['right_fundus']}")

df_processed = pd.DataFrame(records)
print("\nProcessed DataFrame (one row per eye):")
print(df_processed.head())
print("Shape:", df_processed.shape)
print("\nDiagnosis distribution in processed data:")
print(df_processed['diagnosis'].value_counts())

# --- 3. Verify Image Paths and Load a Sample Image ---
# This step is crucial to ensure your paths are correct and images can be read.

if not df_processed.empty:
    sample_record = df_processed.iloc[0]
    sample_image_path = sample_record['image_path']
    print(f"\nAttempting to load a sample image from: {sample_image_path}")

    try:
        sample_image = cv2.imread(sample_image_path)
        if sample_image is None:
            raise FileNotFoundError("Image not found or could not be loaded by OpenCV.")

        print(f"Sample image loaded successfully! Shape: {sample_image.shape}, Type: {sample_image.dtype}")
        plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)) # OpenCV loads as BGR, matplotlib expects RGB
        plt.title(f"Sample Image (Diagnosis: {sample_record['diagnosis']})")
        plt.axis('off')
        plt.show()

    except FileNotFoundError as e:
        print(f"Error loading sample image: {e}")
        print("Please ensure the 'training' and 'testing' image folders are correctly located relative to your script and the Excel paths are accurate.")
        print("Expected image path structure: YOUR_PROJECT_ROOT/data/ODIR-5K/{training|testing}/image_filename.jpg")
    except Exception as e:
        print(f"An unexpected error occurred while loading image: {e}")
else:
    print("No processed data found to load a sample image from. Check data filtering or path issues.")

print("\nData loading and initial processing script finished.")
# At this point, your df_processed dataframe contains tabular data and verified image paths,
# ready for image feature extraction in the next phase.

import pandas as pd
import numpy as np
import os
import cv2 # OpenCV for image loading
import matplotlib.pyplot as plt

# --- Configuration ---
# Adjust this path based on where you extracted the ODIR-5K dataset
DATA_DIR = 'data/ODIR-5K/ODIR-5K'
# Assuming the annotation file is an .xlsx and named something like 'data.xlsx' or 'ODIR-5K.xlsx'
# Please verify the exact name of your Excel file within the ODIR-5K folder
EXCEL_FILE = os.path.join(DATA_DIR, 'data.xlsx') # <--- **ADJUST THIS FILENAME IF YOURS IS DIFFERENT!**

# Image directories are now subfolders within DATA_DIR, WITH CORRECTED NAMES
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, 'Training Images')
TEST_IMAGE_DIR = os.path.join(DATA_DIR, 'Testing Images')


print(f"Loading data from: {EXCEL_FILE}")
print(f"Looking for training images in: {TRAIN_IMAGE_DIR}")
print(f"Looking for testing images in: {TEST_IMAGE_DIR}")

# --- 1. Load Tabular Data (from Excel) ---
try:
    # Changed from pd.read_csv to pd.read_excel
    df_raw = pd.read_excel(EXCEL_FILE)
    print("\nTabular data loaded successfully from Excel.")
    print("Shape:", df_raw.shape)
    print("\nFirst 5 rows of tabular data:")
    print(df_raw.head())
    print("\nInfo of tabular data:")
    df_raw.info()
except FileNotFoundError:
    print(f"Error: Excel file not found at {EXCEL_FILE}. Please check your DATA_DIR and EXCEL_FILE path.")
    exit()
except Exception as e:
    print(f"Error reading Excel file: {e}. Ensure you have 'openpyxl' installed (pip install openpyxl).")
    exit()

# --- 2. Initial Data Cleaning and Feature Selection for Tabular Data ---
# ODIR-5k's Excel usually has columns like 'ID', 'Patient Age', 'Patient Sex',
# 'Left-Fundus', 'Right-Fundus', 'Left-Diagnostic Keywords', 'Right-Diagnostic Keywords'
# We are interested in 'Patient Age', 'Patient Sex', and the diagnostic keywords.

# Function to map diagnostic keywords to our target classes
def map_diagnosis(keywords):
    keywords = str(keywords).lower()
    if 'normal' in keywords:
        return 'Normal'
    elif 'diabetic retinopathy' in keywords or 'dr' in keywords:
        return 'Diabetes'
    elif 'glaucoma' in keywords:
        return 'Glaucoma'
    elif 'cataract' in keywords:
        return 'Cataract'
    else:
        # For cases that don't fit our 4 main classes or are complex
        # We might exclude these or categorize them as 'Other'
        return 'Other' # Or np.nan if we want to drop them

df = df_raw.copy()

# Rename columns for easier access if they are different in your Excel file
# Common ODIR-5k column names from .xlsx:
df = df.rename(columns={
    'Patient Age': 'age',
    'Patient Sex': 'sex',
    'Left-Fundus': 'left_fundus',
    'Right-Fundus': 'right_fundus',
    'Left-Diagnostic Keywords': 'left_diagnostic_keywords',
    'Right-Diagnostic Keywords': 'right_diagnostic_keywords'
})

# Apply diagnosis mapping
df['left_diagnosis'] = df['left_diagnostic_keywords'].apply(map_diagnosis)
df['right_diagnosis'] = df['right_diagnostic_keywords'].apply(map_diagnosis)

# Filter out 'Other' diagnoses for now
df = df[df['left_diagnosis'] != 'Other']
df = df[df['right_diagnosis'] != 'Other']

print("\nDiagnosis distribution for left eye:")
print(df['left_diagnosis'].value_counts())
print("\nDiagnosis distribution for right eye:")
print(df['right_diagnosis'].value_counts())

# Create a row for each eye, including constructing correct image paths
records = []
for index, row in df.iterrows():
    # Helper to find the correct image path (in training or testing)
    def find_image_path(image_filename):
        # Check in training folder first
        path_train = os.path.join(TRAIN_IMAGE_DIR, image_filename)
        if os.path.exists(path_train):
            return path_train
        # Then check in testing folder
        path_test = os.path.join(TEST_IMAGE_DIR, image_filename)
        if os.path.exists(path_test):
            return path_test
        return None # Image not found in either

    # Left eye record
    left_img_path = find_image_path(row['left_fundus'])
    if left_img_path: # Only add if image path is found
        records.append({
            'patient_id': row['ID'], # Assuming 'ID' column for patient ID
            'age': row['age'],
            'sex': row['sex'],
            'eye_side': 'left',
            'image_path': left_img_path,
            'diagnosis': row['left_diagnosis']
        })
    else:
        print(f"Warning: Left image not found for ID {row['ID']}: {row['left_fundus']}")

    # Right eye record
    right_img_path = find_image_path(row['right_fundus'])
    if right_img_path: # Only add if image path is found
        records.append({
            'patient_id': row['ID'],
            'age': row['age'],
            'sex': row['sex'],
            'eye_side': 'right',
            'image_path': right_img_path,
            'diagnosis': row['right_diagnosis']
        })
    else:
        print(f"Warning: Right image not found for ID {row['ID']}: {row['right_fundus']}")

df_processed = pd.DataFrame(records)
print("\nProcessed DataFrame (one row per eye):")
print(df_processed.head())
print("Shape:", df_processed.shape)
print("\nDiagnosis distribution in processed data:")
print(df_processed['diagnosis'].value_counts())

# --- 3. Verify Image Paths and Load a Sample Image ---
# This step is crucial to ensure your paths are correct and images can be read.

if not df_processed.empty:
    sample_record = df_processed.iloc[0]
    sample_image_path = sample_record['image_path']
    print(f"\nAttempting to load a sample image from: {sample_image_path}")

    try:
        sample_image = cv2.imread(sample_image_path)
        if sample_image is None:
            raise FileNotFoundError("Image not found or could not be loaded by OpenCV.")

        print(f"Sample image loaded successfully! Shape: {sample_image.shape}, Type: {sample_image.dtype}")
        plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)) # OpenCV loads as BGR, matplotlib expects RGB
        plt.title(f"Sample Image (Diagnosis: {sample_record['diagnosis']})")
        plt.axis('off')
        plt.show()

    except FileNotFoundError as e:
        print(f"Error loading sample image: {e}")
        print("Please ensure the 'training' and 'testing' image folders are correctly located relative to your script and the Excel paths are accurate.")
        print("Expected image path structure: YOUR_PROJECT_ROOT/data/ODIR-5K/{training|testing}/image_filename.jpg")
    except Exception as e:
        print(f"An unexpected error occurred while loading image: {e}")
else:
    print("No processed data found to load a sample image from. Check data filtering or path issues.")

print("\nData loading and initial processing script finished.")
# At this point, your df_processed dataframe contains tabular data and verified image paths,
# ready for image feature extraction in the next phase.

# --- 4. Split Data into Training and Testing Sets ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("\nSplitting data into training and testing sets...")

# Encode the 'diagnosis' column into numerical labels
# This is crucial for consistency between prepare_odir_data.py and train_multi_class_model.py
label_encoder = LabelEncoder()
df_processed['diagnosis_encoded'] = label_encoder.fit_transform(df_processed['diagnosis'])

# Get the mapping for debugging/understanding
diagnosis_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(f"Diagnosis mapping: {diagnosis_mapping}")

# Split df_processed into train and test sets
# Stratify by 'diagnosis_encoded' to maintain class distribution in both sets
train_df, test_df = train_test_split(
    df_processed,
    test_size=0.2, # 20% for testing, 80% for training
    random_state=42,
    stratify=df_processed['diagnosis_encoded'] # Ensure balanced classes in splits
)

print(f"Training set size: {len(train_df)} samples")
print(f"Testing set size: {len(test_df)} samples")

print("\nClass distribution in training set:")
print(train_df['diagnosis_encoded'].value_counts())
print("\nClass distribution in testing set:")
print(test_df['diagnosis_encoded'].value_counts())


# --- 5. Save Processed DataFrames ---
print("\nSaving processed dataframes...")
train_df.to_csv('train_patients_df.csv', index=False)
test_df.to_csv('test_patients_df.csv', index=False)
print("train_patients_df.csv and test_patients_df.csv saved successfully to the project root.")

print("\nPrepare data script finished.")