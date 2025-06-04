import os
import sys
import json
import pickle
import random
import numpy as np
import xgboost as xgb
import logging
from datetime import datetime

# --- Configuration ---
# Paths to your existing model and scaler
EXISTING_MODEL_PATH = r"X:\clean_copy\three_command_training\round_one_98_percent_acc_three_command_xgboost.json"
SCALER_PATH = r"X:\clean_copy\three_command_training\clean_three_command_scaler_softprob.pkl"

# Path to the new data collected from successful Pong hits
# This file should contain scaled features already (assumed 1x64)
# IMPORTANT: Ensure this path is correct for your system.
SUCCESS_DATA_FILE = r"X:\clean_copy\three_command_retained_training_data\pong_successful_hits_3_class_model.jsonl"

# Path to your original large dataset (contains pickled, unscaled features)
# IMPORTANT: Ensure this path is correct for your system.
ORIGINAL_DATA_PATH = r'F:\sw-kinesis-ai-main\sw-kinesis-dataset\output-balanced-classes-10000-or-more.json'

# Path to save the newly retrained model
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# Ensure the directory for the retrained model exists or is created if needed.
# For simplicity, this example saves to the script's directory.
RETRAINED_MODEL_PATH = f'round_two_xgboost_model_{timestamp}.json'

# --- Label Mapping ---
# This maps string labels from SUCCESS_DATA_FILE to the model's target numerical indices.
# These target indices must align with how the original model was trained.
# Original model's LABEL_MAPPING during its training was:
#   2 (Original Numeric Label for Left Arm) -> Class 0 (Model's Index)
#   8 (Original Numeric Label for Push)     -> Class 1 (Model's Index)
#   0 (Original Numeric Label for Rest)     -> Class 2 (Model's Index)
NEW_DATA_STRING_TO_MODEL_LABEL = {
    "Success_Left_Arm": 0,  # "Left Arm" was mapped to Class 0 in the original model
    "Success_Push": 1       # "Push" was mapped to Class 1 in the original model
}

# This maps original numeric labels (from 'y' in ORIGINAL_DATA_PATH, specifically for 'Rest')
# to the model's target numerical indices.
ORIGINAL_NUMERIC_TO_MODEL_LABEL = {
    0: 2  # Original 'Rest' (numeric label 0) is mapped to Class 2 in the model
    # You could add other original numeric labels here if you planned to sample them
    # e.g., 2: 0, 8: 1, if you were also sampling unscaled Left Arm/Push from original data
}

# Get the target index for 'Rest' data to be added
# The original numeric label for 'Rest' in your dataset was 0.
REST_MODEL_LABEL_INDEX = ORIGINAL_NUMERIC_TO_MODEL_LABEL.get(0)
if REST_MODEL_LABEL_INDEX is None:
    logging.error("Configuration error: 'Rest' (original numeric label 0) not found in ORIGINAL_NUMERIC_TO_MODEL_LABEL mapping.")
    sys.exit(1)

# Get the target indices for Pong success labels
# These will be used when processing SUCCESS_DATA_FILE
LEFT_ARM_MODEL_LABEL_INDEX = NEW_DATA_STRING_TO_MODEL_LABEL.get("Success_Left_Arm")
PUSH_MODEL_LABEL_INDEX = NEW_DATA_STRING_TO_MODEL_LABEL.get("Success_Push")

if LEFT_ARM_MODEL_LABEL_INDEX is None or PUSH_MODEL_LABEL_INDEX is None:
    logging.error("Configuration error: 'Success_Left_Arm' or 'Success_Push' not found in NEW_DATA_STRING_TO_MODEL_LABEL mapping.")
    sys.exit(1)

# Total number of classes the model is trained on.
# This MUST match the num_class parameter of the original model.
# Your original model has 3 classes (0, 1, 2).
NUM_MODEL_CLASSES = 3

# How many 'Rest' samples to randomly select from the original dataset
NUM_REST_SAMPLES_TO_ADD = 1000 # Or your desired number

# XGBoost training parameters

PARAMS = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'eta': 0.01, # Lower learning rate for fine-tuning
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'device': 'cuda:0', # Change to 'cuda' if using GPU
    'num_class': NUM_MODEL_CLASSES, # Crucial: ensure this is 3
    'tree_method': 'hist' # Use 'hist' for faster training on larger datasets
    # Consider adding GPU parameters if applicable and used in original training
    # 'device': 'cuda:0' # or 'gpu_hist' for tree_method if using GPU
}
NUM_BOOST_ROUND = 20000 # Number of boosting rounds for this retraining step

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Helper Functions ---

def load_jsonl(filepath):
    """Loads data from a JSON Lines file."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping invalid JSON line {line_number} in {filepath}: {e}")
        logging.info(f"Successfully loaded {len(data)} lines from {filepath}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        return None

def load_json(filepath):
    """Loads data from a standard JSON file (expecting a dict)."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f: # Try utf-8 first
             data = json.load(f)
        logging.info(f"Successfully loaded JSON with utf-8 encoding from {filepath}.")
        return data
    except UnicodeDecodeError:
        logging.info(f"UTF-8 decoding failed for {filepath}, trying latin-1...")
        try:
             with open(filepath, 'r', encoding='latin-1') as f:
                 data = json.load(f)
             logging.info(f"Successfully loaded JSON with latin-1 encoding from {filepath}.")
             return data
        except Exception as e_latin1:
             logging.error(f"Error loading JSON file {filepath} with latin-1 encoding: {e_latin1}")
             return None
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError as e_json:
        logging.error(f"Invalid JSON format in {filepath}: {e_json}")
        return None
    except Exception as e_other:
        logging.error(f"Error loading {filepath}: {e_other}")
        return None

# --- Main Retraining Logic ---
if __name__ == "__main__":
    logging.info("--- Starting XGBoost Model Retraining Script ---")

    # 1. Load existing model
    logging.info(f"Loading existing model from: {EXISTING_MODEL_PATH}")
    if not os.path.exists(EXISTING_MODEL_PATH):
        logging.error(f"Existing model file not found: {EXISTING_MODEL_PATH}")
        sys.exit(1)
    try:
        bst = xgb.Booster()
        bst.load_model(EXISTING_MODEL_PATH)
        logging.info("Existing model loaded successfully.")
    except xgb.core.XGBoostError as e:
        logging.error(f"Failed to load XGBoost model: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred loading the model: {e}")
        sys.exit(1)

    # 2. Load existing scaler
    logging.info(f"Loading scaler from: {SCALER_PATH}")
    if not os.path.exists(SCALER_PATH):
        logging.error(f"Scaler file not found: {SCALER_PATH}")
        sys.exit(1)
    scaler = None
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        logging.info("Scaler loaded successfully.")
        if not hasattr(scaler, 'transform'):
             logging.error("Loaded scaler object does not have a 'transform' method.")
             sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to load scaler: {e}")
        sys.exit(1)

    # 3. Load new Pong success data (assumed pre-scaled 1x64 features)
    logging.info(f"Loading new Pong success data from: {SUCCESS_DATA_FILE}")
    success_data = load_jsonl(SUCCESS_DATA_FILE)
    if success_data is None: # load_jsonl returns None on critical error
        sys.exit(1)
    if not success_data: # Empty list
        logging.warning("No new success data found or loaded. Retraining will only use added 'Rest' samples if available.")

    # 4. Prepare features and labels from success data
    pong_features_list = []
    pong_labels_list = []
    success_count_mapped = {'Left_Arm': 0, 'Push': 0} # To count successfully mapped new data
    for item_idx, item in enumerate(success_data):
        label_str = item.get("label")
        features = item.get("features")

        if not isinstance(features, list) or len(features) != 64:
            logging.warning(f"Skipping item {item_idx} from success data: features are not a list of 64 numbers. Features: {str(features)[:100]}...")
            continue
        if not label_str:
            logging.warning(f"Skipping item {item_idx} from success data: missing 'label'.")
            continue

        label_index = -1 # Default to invalid
        if label_str == "Success_Left_Arm":
            label_index = LEFT_ARM_MODEL_LABEL_INDEX
            success_count_mapped['Left_Arm'] += 1
        elif label_str == "Success_Push":
            label_index = PUSH_MODEL_LABEL_INDEX
            success_count_mapped['Push'] += 1

        if label_index != -1: # Check if a valid mapping was found
             pong_features_list.append(features)
             pong_labels_list.append(label_index)
        else:
             logging.warning(f"Skipping success data item {item_idx} with unmapped label: '{label_str}'")

    logging.info(f"Prepared {len(pong_features_list)} samples from success data:")
    logging.info(f"  - Mapped Left_Arm (to class {LEFT_ARM_MODEL_LABEL_INDEX}): {success_count_mapped['Left_Arm']}")
    logging.info(f"  - Mapped Push (to class {PUSH_MODEL_LABEL_INDEX}): {success_count_mapped['Push']}")


    # 5. Load original data, extract 'Rest' trials, decode, flatten, and prepare for scaling
    logging.info(f"Loading original data from: {ORIGINAL_DATA_PATH}")
    original_data_dict = load_json(ORIGINAL_DATA_PATH)
    if original_data_dict is None:
        sys.exit(1)
    if not isinstance(original_data_dict, dict):
        logging.error(f"Original data file {ORIGINAL_DATA_PATH} did not load as a dictionary (type: {type(original_data_dict)}). Check file structure.")
        sys.exit(1)

    rest_features_flattened_unscaled = []
    logging.info(f"Filtering original data for 'Rest' (Original Numeric Label: 0, Target Model Index: {REST_MODEL_LABEL_INDEX})...")

    items_processed = 0
    skipped_decode_errors = 0
    skipped_other_errors = 0
    # Iterate through the *values* of the loaded dictionary
    for item_key, item_value in original_data_dict.items():
        items_processed += 1
        if not isinstance(item_value, dict):
            if skipped_other_errors == 0: # Log only once per run to avoid flooding console
                 logging.warning(f"Item value for key '{item_key}' is not a dictionary (type: {type(item_value)}). Further non-dict items will be skipped silently.")
            skipped_other_errors += 1
            continue

        original_label_numeric = item_value.get("y")
        features_encoded = item_value.get("x")

        # Check if this item is 'Rest' (original numeric label 0)
        if original_label_numeric == 0 and isinstance(features_encoded, str):
            try:
                decoded_array = pickle.loads(features_encoded.encode('latin1'))
                if decoded_array.shape == (8, 8):
                    features_flat = decoded_array.reshape(1, -1) # Reshape to (1, 64)
                    if features_flat.shape[1] == 64:
                        rest_features_flattened_unscaled.append(features_flat[0].tolist()) # Append the list of 64 numbers
                    else:
                        if skipped_other_errors == 0: logging.warning(f"Flattened array shape mismatch for 'Rest' key '{item_key}'. Expected 64, got {features_flat.shape[1]}. Skipping.")
                        skipped_other_errors += 1
                else:
                    if skipped_other_errors == 0: logging.warning(f"Decoded array shape mismatch for 'Rest' key '{item_key}'. Expected (8, 8), got {decoded_array.shape}. Skipping.")
                    skipped_other_errors += 1
            except (pickle.UnpicklingError, TypeError, AttributeError, EOFError) as decode_e:
                if skipped_decode_errors == 0: logging.warning(f"Pickle decoding error for 'Rest' key '{item_key}': {decode_e}. Further decode errors will be skipped silently.")
                skipped_decode_errors += 1
            except Exception as other_e:
                 if skipped_other_errors == 0: logging.warning(f"Unexpected error processing 'Rest' item for key '{item_key}': {other_e}. Further errors will be skipped silently.")
                 skipped_other_errors += 1

    logging.info(f"Processed {items_processed} total items from original data dictionary.")
    if skipped_decode_errors > 0: logging.warning(f"Skipped {skipped_decode_errors} items due to decoding errors.")
    if skipped_other_errors > 0: logging.warning(f"Skipped {skipped_other_errors} items due to other errors (wrong type, shape, etc.).")
    logging.info(f"Found {len(rest_features_flattened_unscaled)} 'Rest' samples suitable for scaling.")

    # Combine features and labels - start with the pre-scaled Pong data
    combined_features_list = list(pong_features_list) # Use list() for shallow copy
    combined_labels_list = list(pong_labels_list)

    # Scale and add the 'Rest' features
    num_rest_to_sample = min(NUM_REST_SAMPLES_TO_ADD, len(rest_features_flattened_unscaled))
    if num_rest_to_sample > 0:
        logging.info(f"Randomly sampling {num_rest_to_sample} 'Rest' samples...")
        sampled_indices = random.sample(range(len(rest_features_flattened_unscaled)), num_rest_to_sample)
        sampled_rest_features_unscaled = [rest_features_flattened_unscaled[i] for i in sampled_indices]

        logging.info("Scaling sampled 'Rest' features...")
        try:
            sampled_rest_features_unscaled_np = np.array(sampled_rest_features_unscaled)
            if sampled_rest_features_unscaled_np.ndim != 2 or (sampled_rest_features_unscaled_np.shape[1] != 64 and sampled_rest_features_unscaled_np.size > 0) : # check shape[1] only if not empty
                 logging.error(f"Shape mismatch for unscaled 'Rest' features before scaling. Expected (n, 64), got {sampled_rest_features_unscaled_np.shape}")
                 sys.exit(1)

            if sampled_rest_features_unscaled_np.size > 0: # Only scale if there's data
                sampled_rest_features_scaled = scaler.transform(sampled_rest_features_unscaled_np)
                logging.info(f"Scaled {sampled_rest_features_scaled.shape[0]} 'Rest' samples.")

                combined_features_list.extend(sampled_rest_features_scaled.tolist())
                # Use REST_MODEL_LABEL_INDEX (e.g., 2) for the labels of these 'Rest' samples
                combined_labels_list.extend([REST_MODEL_LABEL_INDEX] * len(sampled_rest_features_scaled))
            else:
                logging.info("No 'Rest' features were sampled to scale (empty array).")

        except ValueError as ve:
             logging.error(f"ValueError during scaling 'Rest' features: {ve}. Check feature dimensions.")
             if 'sampled_rest_features_unscaled_np' in locals(): # Check if var exists
                  logging.error(f"Shape of unscaled 'Rest' data passed to scaler: {sampled_rest_features_unscaled_np.shape}")
             sys.exit(1)
        except Exception as e:
             logging.error(f"Error scaling 'Rest' features: {e}")
             sys.exit(1)
    else:
        logging.info("No 'Rest' data to sample (either none found/extracted or NUM_REST_SAMPLES_TO_ADD is 0).")

    # 6. Prepare final DMatrix for retraining
    if not combined_features_list:
        logging.error("No data available for retraining after processing success data and sampling 'Rest' data. Exiting.")
        sys.exit(1)

    logging.info(f"Total samples for retraining: {len(combined_features_list)}")
    X_retrain = np.array(combined_features_list)
    y_retrain = np.array(combined_labels_list)

    if X_retrain.ndim != 2 or X_retrain.shape[0] != len(y_retrain):
         logging.error(f"Final data shape mismatch before creating DMatrix. Features: {X_retrain.shape}, Labels: {y_retrain.shape}")
         sys.exit(1)
    if X_retrain.shape[0] > 0 and X_retrain.shape[1] != 64: # Check num features only if samples exist
         logging.warning(f"Final feature matrix has {X_retrain.shape[1]} columns, expected 64.")


    logging.info("Creating DMatrix for retraining...")
    try:
        dtrain_retrain = xgb.DMatrix(X_retrain, label=y_retrain)
    except Exception as e:
        logging.error(f"Failed to create DMatrix: {e}")
        logging.error(f"Feature matrix shape: {X_retrain.shape}, Label array shape: {y_retrain.shape}")
        sys.exit(1)

    # 7. Retrain the model
    logging.info(f"Retraining the model for {NUM_BOOST_ROUND} rounds...")
    try:
        retrained_bst = xgb.train(
            PARAMS,
            dtrain_retrain,
            num_boost_round=NUM_BOOST_ROUND,
            xgb_model=bst # Continue training the loaded model
            # Consider adding evals for monitoring if you have a separate validation set
            # evals=[(dtrain_retrain, 'retrain_eval_set')], # Example
            # verbose_eval=10 # Print evaluation metric every 10 rounds
        )
        logging.info("Model retraining completed.")
    except Exception as e:
        logging.error(f"Error during model retraining: {e}")
        sys.exit(1)

    # 8. Save the retrained model
    logging.info(f"Saving retrained model to: {RETRAINED_MODEL_PATH}")
    try:
        retrained_bst.save_model(RETRAINED_MODEL_PATH)
        logging.info("Retrained model saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save retrained model: {e}")
        sys.exit(1)

    logging.info("--- Retraining Script Finished ---")
