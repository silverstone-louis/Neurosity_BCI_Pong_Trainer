import os
import sys
import json
import pickle
import random
import numpy as np
import xgboost as xgb
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss

# --- Configuration ---

# Path to your NEW 3-class EEG data (JSON Lines format)
# Each line: {"features": "<pickled 8x8 cov matrix>", "label": <0-4 int label>}
NEW_EEG_DATA_PATH = r"X:\drone_research\custom_models\xgboost\left_right_both\combined_5_class_eeg_data_proportional_rest.jsonl" # From user log

# Path to save the newly trained 3-class model and its scaler
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
MODEL_SAVE_DIR = r"X:\drone_research\custom_models\xgboost\alternate_3_class_left_right_both" # Directory to save model and scaler
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
NEW_3_class_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f'5_class_eeg_xgb_model_{timestamp}.json')
NEW_SCALER_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f'5_class_eeg_scaler_{timestamp}.pkl')

# --- Label Definition for the New 3-class Problem ---
# Ensures consistency in interpreting labels
CLASS_LABELS_MAP = {
    "Rest": 0,
    "LeftFist": 1,
    "RightFist": 2,
}
NUM_MODEL_CLASSES = 3
# Create target names for classification report in the correct order
TARGET_NAMES_FOR_REPORT = [name for name, index in sorted(CLASS_LABELS_MAP.items(), key=lambda item: item[1])]

# --- Data Augmentation Configuration ---
CREATE_SYNTHETIC_DATA = True # Set to False to skip augmentation
SYNTHETIC_SAMPLES_PER_REAL = 50
# Noise level as a ratio of each feature's standard deviation.
# E.g., 0.02 means noise std dev will be 2% of the feature's std dev.
NOISE_STD_DEV_RATIO = 0.04

# XGBoost training parameters
PARAMS = {
    'objective': 'multi:softprob', 
    'eval_metric': 'mlogloss',     
    'eta': 0.02,                   
    'max_depth': 5, # Consider increasing if dataset becomes much larger and more complex
    'subsample': 0.4,              
    'colsample_bytree': 0.8,       
    'seed': 42,                    
    'device': 'cuda:0', # Change to 'cuda' or 'cuda:0' if GPU available
    'num_class': NUM_MODEL_CLASSES,
    'tree_method': 'hist'          
}
NUM_BOOST_ROUND = 3500 # May need adjustment with augmented data
TEST_SET_SIZE = 0.2 # 20% of (potentially augmented) data for testing
EARLY_STOPPING_ROUNDS = 50 

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Helper Functions ---
def load_eeg_jsonl_data(filepath):
    """Loads EEG covariance data from a JSON Lines file."""
    features_list = []
    labels_list = []
    skipped_lines = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    pickled_features_str = item.get("features")
                    label = item.get("label")

                    if pickled_features_str is None or label is None:
                        logging.warning(f"Skipping line {line_number} in {filepath}: missing 'features' or 'label'.")
                        skipped_lines += 1
                        continue
                    
                    if not isinstance(label, int) or not (0 <= label < NUM_MODEL_CLASSES):
                        logging.warning(f"Skipping line {line_number} in {filepath}: label '{label}' is not a valid integer for {NUM_MODEL_CLASSES} classes.")
                        skipped_lines += 1
                        continue

                    cov_matrix_8x8 = pickle.loads(pickled_features_str.encode('latin1'))
                    
                    if not isinstance(cov_matrix_8x8, np.ndarray) or cov_matrix_8x8.shape != (8, 8):
                        logging.warning(f"Skipping line {line_number} in {filepath}: decoded features are not an 8x8 NumPy array. Shape: {getattr(cov_matrix_8x8, 'shape', 'N/A')}")
                        skipped_lines += 1
                        continue
                    
                    features_flat_64 = cov_matrix_8x8.flatten()
                    if features_flat_64.shape[0] != 64:
                        logging.warning(f"Skipping line {line_number} in {filepath}: flattened features not 64 elements. Shape: {features_flat_64.shape}")
                        skipped_lines += 1
                        continue
                        
                    features_list.append(features_flat_64)
                    labels_list.append(label)

                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping invalid JSON line {line_number} in {filepath}: {e}")
                    skipped_lines += 1
                except (pickle.UnpicklingError, TypeError, AttributeError, EOFError) as decode_e:
                    logging.warning(f"Skipping line {line_number} due to pickle decoding error: {decode_e}")
                    skipped_lines += 1
                except Exception as e_item:
                    logging.warning(f"Skipping line {line_number} due to unexpected error: {e_item}")
                    skipped_lines += 1
        
        if skipped_lines > 0:
            logging.warning(f"Total skipped lines during data loading: {skipped_lines}")
        logging.info(f"Successfully loaded and processed {len(features_list)} genuine samples from {filepath}")
        
        if not features_list: 
            return np.array([]), np.array([])
            
        return np.array(features_list), np.array(labels_list)

    except FileNotFoundError:
        logging.error(f"Data file not found: {filepath}")
        return None, None
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        return None, None

def augment_data_with_noise(X_original, y_original, num_synthetic_per_real, noise_std_ratio):
    """
    Augments data by adding Gaussian noise to create synthetic samples.

    Args:
        X_original (np.ndarray): Original features (n_samples, n_features).
        y_original (np.ndarray): Original labels (n_samples,).
        num_synthetic_per_real (int): Number of synthetic samples to generate per real sample.
        noise_std_ratio (float): Ratio of feature's std dev to use for noise generation.

    Returns:
        tuple: (X_augmented, y_augmented)
    """
    if X_original.size == 0 or num_synthetic_per_real == 0:
        logging.info("No augmentation performed (original data empty or num_synthetic_per_real is 0).")
        return X_original, y_original

    logging.info(f"Starting data augmentation: {num_synthetic_per_real} synthetic samples per real sample.")
    
    n_samples_orig, n_features = X_original.shape
    
    # Calculate standard deviation for each feature across the original dataset
    # Add a small epsilon to prevent division by zero or zero std dev for constant features
    feature_std_devs = np.std(X_original, axis=0) + 1e-9 
    
    X_synthetic_list = []
    y_synthetic_list = []

    for i in range(n_samples_orig):
        original_sample = X_original[i]
        original_label = y_original[i]
        for _ in range(num_synthetic_per_real):
            # Generate noise based on the std dev of each feature
            noise = np.random.normal(loc=0.0, scale=feature_std_devs * noise_std_ratio, size=n_features)
            synthetic_sample = original_sample + noise
            X_synthetic_list.append(synthetic_sample)
            y_synthetic_list.append(original_label)
    
    X_synthetic = np.array(X_synthetic_list)
    y_synthetic = np.array(y_synthetic_list)
    
    logging.info(f"Generated {X_synthetic.shape[0]} synthetic samples.")

    # Combine original and synthetic data
    X_augmented = np.vstack((X_original, X_synthetic))
    y_augmented = np.concatenate((y_original, y_synthetic))
    
    logging.info(f"Total data size after augmentation: {X_augmented.shape[0]} samples.")
    return X_augmented, y_augmented

# --- Main Training Logic ---
if __name__ == "__main__":
    logging.info("--- Starting 3-class EEG XGBoost Model Training & Evaluation Script (with Augmentation) ---")

    # 1. Load new 3-class EEG data
    logging.info(f"Loading genuine 3-class EEG data from: {NEW_EEG_DATA_PATH}")
    if not os.path.exists(NEW_EEG_DATA_PATH):
        logging.error(f"New EEG data file not found: {NEW_EEG_DATA_PATH}. Please update path.")
        sys.exit(1)
        
    X_genuine_data, y_genuine_data = load_eeg_jsonl_data(NEW_EEG_DATA_PATH)

    if X_genuine_data is None or y_genuine_data is None or X_genuine_data.size == 0:
        logging.error("Failed to load genuine EEG data or data is empty. Exiting.")
        sys.exit(1)
    
    logging.info(f"Loaded {X_genuine_data.shape[0]} genuine samples with {X_genuine_data.shape[1]} features each.")

    # 2. Augment Data (Optional)
    X_all_data = X_genuine_data
    y_all_data = y_genuine_data
    if CREATE_SYNTHETIC_DATA:
        X_all_data, y_all_data = augment_data_with_noise(
            X_genuine_data, 
            y_genuine_data,
            SYNTHETIC_SAMPLES_PER_REAL,
            NOISE_STD_DEV_RATIO
        )
    else:
        logging.info("Skipping data augmentation as CREATE_SYNTHETIC_DATA is False.")


    # 3. Fit Scaler on the full (potentially augmented) dataset
    logging.info("Fitting a new StandardScaler on the full (potentially augmented) 3-class EEG data.")
    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(X_all_data) # X_all_data is now potentially augmented
    logging.info("New scaler fitted and applied to the full data.")
    logging.info(f"Saving newly fitted scaler to: {NEW_SCALER_SAVE_PATH}")
    try:
        with open(NEW_SCALER_SAVE_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        logging.info("New scaler saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save new scaler: {e}")

    # 4. Split data into Training and Test sets
    logging.info(f"Splitting data into training and test sets (Test size: {TEST_SET_SIZE*100}%)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_all, 
        y_all_data, 
        test_size=TEST_SET_SIZE, 
        random_state=42, 
        stratify=y_all_data 
    )
    logging.info(f"Training set size: {X_train.shape[0]} samples")
    logging.info(f"Test set size: {X_test.shape[0]} samples")

    # 5. Prepare DMatrices
    logging.info("Creating DMatrices for training and testing...")
    try:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        logging.info(f"DMatrix for training created: {dtrain.num_row()} rows, {dtrain.num_col()} columns.")
        logging.info(f"DMatrix for testing created: {dtest.num_row()} rows, {dtest.num_col()} columns.")
    except Exception as e:
        logging.error(f"Failed to create DMatrices: {e}")
        sys.exit(1)

    # 6. Train the 3-class XGBoost model
    logging.info(f"Training a new {NUM_MODEL_CLASSES}-class XGBoost model for up to {NUM_BOOST_ROUND} rounds...")
    logging.info(f"Using XGBoost parameters: {PARAMS}")
    
    watchlist = [(dtrain, 'train'), (dtest, 'eval')] 
    
    bst_3_class = None 
    try:
        bst_3_class = xgb.train(
            PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=watchlist,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, 
            verbose_eval=100 
        )
        logging.info("New 3-class model training completed.")
        if bst_3_class: # Check if model object exists (early stopping might not return if it fails badly)
             logging.info(f"Best iteration: {bst_3_class.best_iteration}, Best score (mlogloss on eval): {bst_3_class.best_score}")

    except Exception as e:
        logging.error(f"Error during 3-class model training: {e}")
        sys.exit(1)

    # 7. Save the newly trained 3-class model
    if bst_3_class:
        logging.info(f"Saving newly trained 3-class model to: {NEW_3_class_MODEL_PATH}")
        try:
            bst_3_class.save_model(NEW_3_class_MODEL_PATH)
            logging.info("New 3-class model saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save new 3-class model: {e}")
    else:
        logging.error("Training resulted in no model object. Model not saved.")
        sys.exit(1) 

    # 8. Evaluate the model on the Test Set (Benchmark)
    logging.info("\n--- Model Evaluation on Test Set ---")
    try:
        y_pred_proba_test = bst_3_class.predict(dtest, iteration_range=(0, bst_3_class.best_iteration + 1))
        y_pred_labels_test = np.argmax(y_pred_proba_test, axis=1)

        accuracy = accuracy_score(y_test, y_pred_labels_test)
        test_log_loss = log_loss(y_test, y_pred_proba_test) 
        conf_matrix = confusion_matrix(y_test, y_pred_labels_test)
        class_report = classification_report(y_test, y_pred_labels_test, target_names=TARGET_NAMES_FOR_REPORT, zero_division=0)

        logging.info(f"Test Set Accuracy: {accuracy:.4f}")
        logging.info(f"Test Set Log Loss: {test_log_loss:.4f}")
        
        logging.info("\nTest Set Confusion Matrix:")
        conf_matrix_str = "\nPredicted ->\n"
        header = "Actual | " + " | ".join([f"{name[:7]:^7}" for name in TARGET_NAMES_FOR_REPORT]) + "\n"
        conf_matrix_str += header
        conf_matrix_str += "-" * len(header) + "\n"
        for i, row in enumerate(conf_matrix):
            conf_matrix_str += f"{TARGET_NAMES_FOR_REPORT[i][:6]:<6} | " + " | ".join([f"{x:^7}" for x in row]) + "\n"
        print(conf_matrix_str) 

        logging.info("\nTest Set Classification Report:")
        print(class_report) 

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")

    logging.info("--- 3-class EEG XGBoost Model Training & Evaluation Script Finished ---")
