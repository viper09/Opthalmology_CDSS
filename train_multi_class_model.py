import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # New import for encoding string labels
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Script started: Training multi-class model for ODIR-5K diseases...")

# --- Configuration ---
PROCESSED_DATA_DIR = 'data/processed_odir'
PROCESSED_DF_PATH = os.path.join(PROCESSED_DATA_DIR, 'odir_processed_data.csv')
IMAGE_FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, 'odir_image_features.npy')

MODEL_SAVE_PATH = 'catboost_multi_class_odir_model.cbm' # New name for multi-class model

# --- 1. Load Data ---
try:
    print(f"Loading processed DataFrame from: {PROCESSED_DF_PATH}")
    df_processed = pd.read_csv(PROCESSED_DF_PATH)
    print(f"Loaded DataFrame shape: {df_processed.shape}")

    print(f"\nLoading image features from: {IMAGE_FEATURES_PATH}")
    image_features_np = np.load(IMAGE_FEATURES_PATH)
    print(f"Loaded image features shape: {image_features_np.shape}")

    # Create a DataFrame for image features, including the image_path to merge
    df_image_features = pd.DataFrame(image_features_np, columns=[f'img_feat_{i}' for i in range(image_features_np.shape[1])])
    df_image_features['image_path'] = df_processed['image_path'].tolist() # Assuming perfect order match

    print("\nMerging tabular data with image features...")
    combined_data = pd.merge(df_processed, df_image_features, on='image_path', how='inner')
    print(f"Combined data shape after merge: {combined_data.shape}")

except FileNotFoundError as e:
    print(f"Error loading files: {e}. Please ensure '{PROCESSED_DATA_DIR}' exists and contains the necessary files.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading or merging: {e}")
    exit()

# --- 2. Prepare Target Variable (Multi-class Diagnosis) ---
# We will use the 'diagnosis' column directly as the target
# First, encode the string labels to numerical labels
print("\nEncoding target labels (Normal, Diabetes, Glaucoma, Cataract)...")
label_encoder = LabelEncoder()
combined_data['diagnosis_encoded'] = label_encoder.fit_transform(combined_data['diagnosis'])
print(f"Original diagnoses: {label_encoder.classes_}")
print(f"Encoded labels: {label_encoder.transform(label_encoder.classes_)}")
print("\nEncoded Diagnosis distribution:")
print(combined_data['diagnosis_encoded'].value_counts().sort_index())


# --- 3. Separate Features (X) and Target (y) ---
# X will contain 'age', 'sex', 'eye_side', and all 'img_feat_X' columns
# y will contain the newly encoded 'diagnosis_encoded'

image_feature_columns = [col for col in combined_data.columns if col.startswith('img_feat_')]
X_columns = ['age', 'sex', 'eye_side'] + image_feature_columns
X = combined_data[X_columns]
y = combined_data['diagnosis_encoded'] # Use the encoded target

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# --- 4. Identify Categorical Features for CatBoost ---
categorical_feature_names = ['sex', 'eye_side'] # Still just these two for now
categorical_features_indices = [X.columns.get_loc(col) for col in categorical_feature_names]

print(f"\nIdentified Categorical Features (by name): {categorical_feature_names}")
print(f"Identified Categorical Features (by index): {categorical_features_indices}")

# --- 5. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # Stratify is important for multi-class imbalance
)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print("\nClass distribution in training set:")
print(y_train.value_counts().sort_index())
print("\nClass distribution in testing set:")
print(y_test.value_counts().sort_index())

# --- 6. Prepare Data for CatBoost using Pool ---
train_pool = Pool(X_train, y_train, cat_features=categorical_feature_names)
test_pool = Pool(X_test, y_test, cat_features=categorical_feature_names)

print("\nData preparation complete. Ready for multi-class model training.")

# --- 7. Initialize and Train the CatBoost Model for Multi-class ---
print("\nStarting CatBoost multi-class model training...")

# Define class weights to handle imbalance
# These weights are calculated based on the inverse of class frequencies
# The order corresponds to diagnosis_encoded: 0 (Cataract), 1 (Diabetes), 2 (Glaucoma), 3 (Normal)
class_weights = [
    12.74, # Weight for class 0 (Cataract: 2700/212)
    8.18,  # Weight for class 1 (Diabetes: 2700/330)
    13.24, # Weight for class 2 (Glaucoma: 2700/204)
    1.38   # Weight for class 3 (Normal: 2700/1954)
]

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    loss_function='MultiClass', # Changed for multi-class classification
    eval_metric='Accuracy', # Changed for multi-class
    random_seed=42,
    verbose=50,
    early_stopping_rounds=50,
    class_weights=class_weights, # <--- ADD THIS LINE HERE!
)

model.fit(train_pool, eval_set=test_pool)

print("\nCatBoost multi-class model training complete.")

# --- 8. Save the Trained Model ---
model.save_model(MODEL_SAVE_PATH)
print(f"CatBoost multi-class model saved to {MODEL_SAVE_PATH}")


# --- 9. Evaluate the Multi-class Model ---
print("\nStarting multi-class model evaluation...")

y_pred_proba = model.predict_proba(X_test) # Now returns probabilities for all classes
y_pred = model.predict(X_test).flatten() # Ensure predictions are 1D array of encoded labels

print("\nEvaluation Metrics:")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# For multi-class, precision, recall, f1-score need 'average' parameter
# 'weighted' accounts for class imbalance by weighting metrics by support (number of true instances for each label)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
print(f"Precision (Weighted): {precision:.4f}")

recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall (Weighted): {recall:.4f}")

f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-Score (Weighted): {f1:.4f}")

# For ROC AUC in multi-class, use 'multi_class' and 'average' parameters
# 'ovr' (one-vs-rest) is a common strategy
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
print(f"ROC AUC Score (Weighted One-vs-Rest): {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Rows: True, Columns: Predicted):")
print(cm)

# Visualize the Confusion Matrix for Multi-class
print("\nGenerating Multi-class Confusion Matrix Plot...")
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=label_encoder.classes_, # Use original string labels for ticks
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for ODIR-5K Multi-class Prediction')
plt.show()

print("\nModel evaluation complete.")
print("Script finished.")