# Filename: process_xdf_for_training.py
# Description: Loads XDF files, synchronizes LSL streams (Markers, Raw EEG),
#              extracts epochs based on markers, generates features (covariance from Raw EEG),
#              applies margin-based filtering, optionally augments data, and saves results.

import os
import glob
import json
import pickle
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import pyxdf
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler # For optional scaling
from scipy.signal import butter, lfilter, detrend # For filtering

# --- Configuration ---

# --- Input/Output Paths ---
XDF_DATA_DIRECTORY = r'C:\Users\silve\OneDrive\Desktop\upload_to_github\training\training_data\success_data' # User-provided path
BASE_OUTPUT_DIR = 'processed_xdf_data'  # Main directory for all outputs
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Subdirectories for outputs
PROCESSED_DATA_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"run_{TIMESTAMP}")
os.makedirs(PROCESSED_DATA_OUTPUT_DIR, exist_ok=True)
FIGURES_OUTPUT_DIR = os.path.join(PROCESSED_DATA_OUTPUT_DIR, 'figures')
os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(PROCESSED_DATA_OUTPUT_DIR, f"xdf_processing_log_{TIMESTAMP}.txt")

# Output Filenames (will be saved in PROCESSED_DATA_OUTPUT_DIR)
RAW_EXTRACTED_JSONL_FILENAME = f'xdf_raw_extracted_features_{TIMESTAMP}.jsonl'
SCALED_FEATURES_JSONL_FILENAME = f'xdf_scaled_features_{TIMESTAMP}.jsonl' # If scaling is applied
CLEANED_JSONL_FILENAME = f'xdf_cleaned_features_{TIMESTAMP}.jsonl'
AUGMENTED_JSONL_FILENAME = f'xdf_augmented_features_{TIMESTAMP}.jsonl'

# --- LSL Stream Identification ---
MARKER_STREAM_QUERY = {'name': 'NeuroCollectorEvents'}
RAW_EEG_STREAM_QUERY = {'type': 'EEG'} # Query for the Raw EEG stream

# --- Epoching Parameters ---
EPOCH_START_OFFSET_SECS = 0.5  # Start epoch 0.5s after marker onset
EPOCH_DURATION_SECS = 2.0    # Extract 2.0s long epochs

# --- Feature Extraction Strategy ---
FEATURE_SOURCE = 'RawEEG' # Changed to focus on Raw EEG for these files

# Raw EEG Processing Parameters (if FEATURE_SOURCE is 'RawEEG')
# These might be overridden by values from XDF stream info if available
RAW_EEG_DEFAULT_SFREQ = 256.0  # Default sampling frequency
RAW_EEG_DEFAULT_NB_CHAN = 8    # Default number of channels (e.g., Neurosity Crown)
RAW_EEG_FILTER_LOW_HZ = 7.0
RAW_EEG_FILTER_HIGH_HZ = 30.0
RAW_EEG_FILTER_ORDER = 5

# --- Label Mapping ---
MARKER_TO_LABEL_MAP = {
    # Update these to match your exact marker strings from prompted sessions
    "COMMAND_Left_Fist": "Left_Fist",
    "COMMAND_Right_Fist": "Right_Fist",
    "COMMAND_Rest": "Rest",
    "Prompt_Left_Fist": "Left_Fist", # Example if you used "Prompt_"
    "Prompt_Right_Fist": "Right_Fist",
    "Prompt_Rest": "Rest"
}
EXPECTED_LABELS_FOR_CLEANING = ["Left_Fist", "Right_Fist", "Rest"]

# --- Data Scaling (Applied AFTER feature extraction) ---
APPLY_SCALING = True
SCALER_PATH = None # Path to a PRE-FITTED StandardScaler (.pkl) or None to fit a new one
NEW_SCALER_SAVE_PATH = os.path.join(PROCESSED_DATA_OUTPUT_DIR, f'xdf_fitted_scaler_{TIMESTAMP}.pkl')

# --- Margin-Based Filtering (Applied AFTER scaling if APPLY_SCALING is True) ---
APPLY_MARGIN_FILTERING = True
MARGIN_THRESHOLDS_XDF = {"Left_Fist": 0.05, "Right_Fist": 0.05, "Rest": 0.02} # PLACEHOLDERS

# --- Data Augmentation (Applied to cleaned, scaled data) ---
CREATE_SYNTHETIC_DATA_XDF = True
SYNTHETIC_SAMPLES_PER_CLEANED_REAL_XDF = 5
NOISE_STD_DEV_RATIO_XDF = 0.04

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE_PATH),
                        logging.StreamHandler()
                    ])

# --- Helper Functions ---

def find_stream(streams, query):
    """Finds a stream in the XDF data based on a query dictionary."""
    for stream_idx, stream in enumerate(streams): # Keep track of index for more informative logging
        info = stream.get('info', {})
        name_val = info.get('name', ['N/A'])[0] if isinstance(info.get('name'), list) else info.get('name', 'N/A')
        type_val = info.get('type', ['N/A'])[0] if isinstance(info.get('type'), list) else info.get('type', 'N/A')
        
        match = True
        for key, value in query.items():
            stream_value = None
            if key == 'name':
                stream_value = name_val
            elif key == 'type':
                stream_value = type_val
            else: # For other keys like 'channel_count' etc. if added to query
                stream_value = info.get(key)

            if stream_value != value:
                match = False
                break
        if match:
            logging.info(f"Found stream matching query {query}: Index={stream_idx}, Name='{name_val}', Type='{type_val}', Channels='{info.get('channel_count',['N/A'])[0]}', SRATE='{info.get('nominal_srate',['N/A'])[0]}'")
            return stream
    logging.warning(f"Could not find stream matching query: {query}")
    return None


def get_samples_in_epoch(stream_data, stream_timestamps, epoch_start_time, epoch_end_time):
    """Extracts samples from a stream that fall within an epoch window."""
    if stream_data is None or stream_timestamps is None:
        return None, None
    if stream_timestamps.ndim > 1: stream_timestamps = stream_timestamps.flatten()
    indices = np.where((stream_timestamps >= epoch_start_time) & (stream_timestamps <= epoch_end_time))[0]
    if len(indices) == 0: return None, None
    return stream_data[indices], stream_timestamps[indices]

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    # Ensure low and high are valid and in order
    if low <= 0: low = 1e-5 # Avoid zero or negative lowcut
    if high >= 1: high = 1 - 1e-5 # Avoid highcut at or above Nyquist
    if low >= high:
        logging.warning(f"Lowcut ({lowcut}Hz) is >= highcut ({highcut}Hz) or invalid relative to Nyquist. Skipping filter.")
        return data
        
    b, a = butter(order, [low, high], btype='band')
    # Apply filter channel by channel if data is (samples, channels) or (channels, samples)
    if data.ndim == 1:
        return lfilter(b, a, data)
    elif data.ndim == 2:
        # Assuming data is (samples, channels), apply along axis 0
        # If (channels, samples), then axis=1
        # Let's assume (samples, channels) from typical LSL EEG streams
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]): # Iterate over channels
            filtered_data[:, i] = lfilter(b, a, data[:, i])
        return filtered_data
    else:
        logging.warning("Data for filtering has unexpected dimensions. Skipping filter.")
        return data

def calculate_covariance_features(epoch_data_filtered):
    """Calculates and flattens the covariance matrix from epoch data (channels x samples)."""
    if epoch_data_filtered is None or epoch_data_filtered.ndim != 2 or epoch_data_filtered.shape[0] < 2: # Need at least 2 samples for cov
        logging.debug("Not enough data or wrong dimensions for covariance calculation.")
        return None
    # Transpose if necessary: np.cov expects variables (channels) as rows
    # If epoch_data_filtered is (samples, channels), transpose it.
    if epoch_data_filtered.shape[0] < epoch_data_filtered.shape[1]: # Heuristic: fewer rows than columns suggests (channels, samples)
        pass # Already (channels, samples)
    else: # Assume (samples, channels)
        epoch_data_filtered = epoch_data_filtered.T

    if epoch_data_filtered.shape[1] < 2: # Need at least 2 observations (time points) per channel
        logging.debug("Not enough samples per channel for covariance calculation after potential transpose.")
        return None

    cov_matrix = np.cov(epoch_data_filtered)
    return cov_matrix.flatten()


def augment_data_with_noise(X_original, y_original_numerical, num_synthetic_per_real, noise_std_ratio, int_to_label_map):
    """Augments data by adding Gaussian noise. Returns list of dicts for JSONL."""
    if X_original.size == 0 or num_synthetic_per_real == 0:
        logging.info("No augmentation performed (original data empty or num_synthetic_per_real is 0).")
        return []
    logging.info(f"Starting data augmentation: {num_synthetic_per_real} synthetic samples per real sample.")
    n_samples_orig, n_features = X_original.shape
    feature_std_devs = np.std(X_original, axis=0) + 1e-9
    augmented_records = []
    for i in range(n_samples_orig):
        original_sample_features = X_original[i]
        original_numerical_label = y_original_numerical[i]
        string_label = int_to_label_map.get(original_numerical_label, "Unknown_Label_Synthetic")
        for _ in range(num_synthetic_per_real):
            noise = np.random.normal(loc=0.0, scale=feature_std_devs * noise_std_ratio, size=n_features)
            synthetic_features = (original_sample_features + noise).tolist()
            augmented_records.append({"features": synthetic_features, "label": string_label, "type": "synthetic"})
    logging.info(f"Generated {len(augmented_records)} synthetic samples.")
    return augmented_records

# --- Main Processing Logic ---
def process_all_xdf_files():
    all_extracted_epochs = []
    xdf_files = glob.glob(os.path.join(XDF_DATA_DIRECTORY, '*.xdf'))

    if not xdf_files:
        logging.error(f"No XDF files found in directory: {XDF_DATA_DIRECTORY}")
        return

    logging.info(f"Found {len(xdf_files)} XDF files to process.")

    for xdf_filepath in xdf_files:
        logging.info(f"--- Processing XDF file: {xdf_filepath} ---")
        try:
            streams, header = pyxdf.load_xdf(xdf_filepath, synchronize_clocks=True)
        except Exception as e:
            logging.error(f"Failed to load XDF file {xdf_filepath}: {e}", exc_info=True)
            continue

        marker_stream = find_stream(streams, MARKER_STREAM_QUERY)
        
        raw_eeg_stream = None
        if FEATURE_SOURCE == 'RawEEG':
            raw_eeg_stream = find_stream(streams, RAW_EEG_STREAM_QUERY)
        # Add logic for PSD/PowerByBand if those become options again for other datasets

        if not marker_stream:
            logging.warning(f"Marker stream not found in {xdf_filepath}. Skipping this file for epoching.")
            continue
        if FEATURE_SOURCE == 'RawEEG' and not raw_eeg_stream:
            logging.warning(f"Raw EEG stream not found in {xdf_filepath} (FEATURE_SOURCE='RawEEG'). Skipping this file.")
            continue
        
        marker_timestamps = marker_stream['time_stamps']
        marker_values = marker_stream['time_series']

        # For RawEEG processing
        current_sfreq = RAW_EEG_DEFAULT_SFREQ
        current_nb_chan = RAW_EEG_DEFAULT_NB_CHAN
        if FEATURE_SOURCE == 'RawEEG' and raw_eeg_stream:
            raw_eeg_info = raw_eeg_stream.get('info', {})
            try: # nominal_srate is often a list containing a string
                srate_val = raw_eeg_info.get('nominal_srate', [str(RAW_EEG_DEFAULT_SFREQ)])[0]
                current_sfreq = float(srate_val) if srate_val else RAW_EEG_DEFAULT_SFREQ
            except (IndexError, ValueError, TypeError) as e:
                logging.warning(f"Could not parse nominal_srate from EEG stream info in {xdf_filepath}. Using default {RAW_EEG_DEFAULT_SFREQ}Hz. Error: {e}")
            try: # channel_count is often a list containing a string
                chan_val = raw_eeg_info.get('channel_count', [str(RAW_EEG_DEFAULT_NB_CHAN)])[0]
                current_nb_chan = int(chan_val) if chan_val else RAW_EEG_DEFAULT_NB_CHAN
            except (IndexError, ValueError, TypeError) as e:
                logging.warning(f"Could not parse channel_count from EEG stream info in {xdf_filepath}. Using default {RAW_EEG_DEFAULT_NB_CHAN} channels. Error: {e}")

            raw_eeg_data = raw_eeg_stream['time_series']
            raw_eeg_timestamps = raw_eeg_stream['time_stamps']
            
            # Ensure raw_eeg_data has the expected number of channels
            if raw_eeg_data.shape[1] != current_nb_chan:
                logging.warning(f"EEG stream in {xdf_filepath} has {raw_eeg_data.shape[1]} channels, expected {current_nb_chan} based on info/default. Adjusting expectations or check data.")
                # current_nb_chan = raw_eeg_data.shape[1] # Or adjust to actual data
                # For now, we'll proceed but features might be inconsistent if this varies.

        logging.info(f"Found {len(marker_timestamps)} markers in {os.path.basename(xdf_filepath)}.")

        for i, marker_ts in enumerate(marker_timestamps):
            marker_str = marker_values[i][0] if isinstance(marker_values[i], list) and marker_values[i] else str(marker_values[i])

            if marker_str in MARKER_TO_LABEL_MAP:
                label = MARKER_TO_LABEL_MAP[marker_str]
                epoch_start = marker_ts + EPOCH_START_OFFSET_SECS
                epoch_end = epoch_start + EPOCH_DURATION_SECS
                
                extracted_feature_vector = None

                if FEATURE_SOURCE == 'RawEEG' and raw_eeg_stream:
                    epoch_raw_eeg_samples, _ = get_samples_in_epoch(raw_eeg_data, raw_eeg_timestamps, epoch_start, epoch_end)
                    if epoch_raw_eeg_samples is not None and len(epoch_raw_eeg_samples) > 0:
                        # Detrend before filtering
                        epoch_raw_eeg_samples_detrended = detrend(epoch_raw_eeg_samples, axis=0, type='linear')
                        epoch_filtered_eeg = bandpass_filter(epoch_raw_eeg_samples_detrended, RAW_EEG_FILTER_LOW_HZ, RAW_EEG_FILTER_HIGH_HZ, current_sfreq, order=RAW_EEG_FILTER_ORDER)
                        # Ensure data is (channels, samples) for np.cov if it's (samples, channels)
                        # Our bandpass_filter assumes (samples, channels) and returns that.
                        # np.cov expects variables (channels) as rows.
                        if epoch_filtered_eeg.shape[0] < epoch_filtered_eeg.shape[1]: # If (channels, samples)
                            transposed_for_cov = epoch_filtered_eeg
                        else: # If (samples, channels)
                            transposed_for_cov = epoch_filtered_eeg.T 
                        
                        # Ensure we have enough channels and samples for covariance
                        if transposed_for_cov.shape[0] == current_nb_chan and transposed_for_cov.shape[1] >= 2 : # Need at least 2 samples
                            extracted_feature_vector = calculate_covariance_features(transposed_for_cov)
                        else:
                            logging.debug(f"Not enough data for covariance for epoch around marker '{marker_str}' at {marker_ts:.2f}s. Shape after transpose: {transposed_for_cov.shape}, Expected channels: {current_nb_chan}")
                    else:
                        logging.debug(f"No RawEEG samples found for epoch around marker '{marker_str}' at {marker_ts:.2f}s.")
                
                # Add logic for PSD/PBB here if re-enabled

                if extracted_feature_vector is not None:
                    all_extracted_epochs.append({
                        "features": extracted_feature_vector.tolist(),
                        "label": label,
                        "source_xdf": os.path.basename(xdf_filepath),
                        "marker_timestamp": marker_ts
                    })
                # else:
                    # logging.debug(f"Could not extract/calculate features for epoch around marker '{marker_str}' at {marker_ts:.2f}s.")
            # else:
                # logging.debug(f"Marker '{marker_str}' ('{type(marker_str)}') not in MARKER_TO_LABEL_MAP. Keys: {list(MARKER_TO_LABEL_MAP.keys())}")


    # --- Post-Extraction Processing ---
    if not all_extracted_epochs:
        logging.info("No epochs were extracted from any XDF files. Exiting.")
        return

    df_epochs = pd.DataFrame(all_extracted_epochs)
    logging.info(f"Successfully extracted {len(df_epochs)} epochs in total.")
    logging.info("Initial extracted class distribution:")
    logging.info(df_epochs['label'].value_counts())

    raw_extracted_output_path = os.path.join(PROCESSED_DATA_OUTPUT_DIR, RAW_EXTRACTED_JSONL_FILENAME)
    with open(raw_extracted_output_path, 'w') as f:
        for record_idx, record_data in df_epochs.iterrows(): # Use iterrows to access as dict-like
            # Convert numpy arrays in features to list if they are not already
            # This should already be handled by .tolist() during append
            f.write(json.dumps(record_data.to_dict()) + '\n')
    logging.info(f"Saved raw extracted features to: {raw_extracted_output_path}")

    try:
        X_extracted = np.array(df_epochs['features'].tolist())
        if X_extracted.ndim == 1: # Happens if all feature vectors are same length but only 1 epoch
            expected_feature_len = RAW_EEG_DEFAULT_NB_CHAN * RAW_EEG_DEFAULT_NB_CHAN
            if X_extracted.shape[0] == expected_feature_len:
                 X_extracted = X_extracted.reshape(1, -1)
            else:
                raise ValueError(f"Single extracted epoch has unexpected feature length {X_extracted.shape[0]}, expected {expected_feature_len}")

    except ValueError as e:
        logging.error(f"Could not convert features to NumPy array. Possible inconsistent feature lengths: {e}")
        feature_lengths = df_epochs['features'].apply(len)
        logging.error(f"Feature length stats: min={feature_lengths.min()}, max={feature_lengths.max()}, mean={feature_lengths.mean()}")
        if feature_lengths.min() != feature_lengths.max():
            logging.error("Inconsistent feature lengths found. Example lengths:")
            logging.error(feature_lengths.value_counts())
        return
    
    if X_extracted.shape[1] != (RAW_EEG_DEFAULT_NB_CHAN * RAW_EEG_DEFAULT_NB_CHAN):
        logging.warning(f"Warning: Extracted features have {X_extracted.shape[1]} dimensions, but expected {RAW_EEG_DEFAULT_NB_CHAN**2} for {RAW_EEG_DEFAULT_NB_CHAN}-channel covariance.")
        # This could happen if current_nb_chan varied between files and RAW_EEG_DEFAULT_NB_CHAN was not updated,
        # or if covariance calculation failed for some epochs returning None/shorter features that were then filtered out
        # before this check. For now, proceed but be aware.


    y_extracted_str = df_epochs['label']
    unique_labels_str = sorted(y_extracted_str.unique())
    label_to_int_map = {label: i for i, label in enumerate(unique_labels_str)}
    int_to_label_map = {i: label for label, i in label_to_int_map.items()}
    y_extracted_numerical = y_extracted_str.map(label_to_int_map).values

    scaler = None
    if APPLY_SCALING:
        if SCALER_PATH and os.path.exists(SCALER_PATH):
            try:
                with open(SCALER_PATH, 'rb') as f_scaler: scaler = pickle.load(f_scaler)
                logging.info(f"Loaded pre-fitted scaler from: {SCALER_PATH}")
                X_scaled = scaler.transform(X_extracted)
            except Exception as e:
                logging.error(f"Failed to load or use scaler from {SCALER_PATH}: {e}. Fitting a new one.")
                scaler = None
        if scaler is None:
            logging.info("Fitting a new StandardScaler on the extracted XDF features.")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_extracted)
            try:
                with open(NEW_SCALER_SAVE_PATH, 'wb') as f_scaler_save: pickle.dump(scaler, f_scaler_save)
                logging.info(f"Saved newly fitted scaler to: {NEW_SCALER_SAVE_PATH}")
            except Exception as e: logging.error(f"Could not save newly fitted scaler: {e}")
        
        # Save scaled features to JSONL
        scaled_records_to_save = []
        for i in range(X_scaled.shape[0]):
            original_record = df_epochs.iloc[i]
            scaled_records_to_save.append({
                "features": X_scaled[i].tolist(),
                "label": original_record['label'],
                "source_xdf": original_record['source_xdf'],
                "marker_timestamp": original_record['marker_timestamp']
            })
        scaled_output_path = os.path.join(PROCESSED_DATA_OUTPUT_DIR, SCALED_FEATURES_JSONL_FILENAME)
        with open(scaled_output_path, 'w') as f:
            for record in scaled_records_to_save:
                f.write(json.dumps(record) + '\n')
        logging.info(f"Saved {len(scaled_records_to_save)} scaled records to: {scaled_output_path}")
        X_current_features = X_scaled
    else:
        logging.info("Skipping feature scaling.")
        X_current_features = X_extracted

    df_processed = pd.DataFrame() # Rebuild df_processed with the features used for margin calculation
    df_processed['features_for_margin'] = [feat.tolist() for feat in X_current_features] # Store as list
    df_processed['label'] = y_extracted_str.values # Ensure index alignment
    df_processed['source_xdf'] = df_epochs['source_xdf'].values
    df_processed['marker_timestamp'] = df_epochs['marker_timestamp'].values


    if APPLY_MARGIN_FILTERING:
        logging.info("Applying margin-based filtering...")
        if X_current_features.shape[0] == 0: logging.warning("No features to filter.")
        else:
            centroids = {}
            # Use y_extracted_numerical which is aligned with X_current_features
            for label_str_filt, label_int_filt in label_to_int_map.items():
                if label_str_filt not in EXPECTED_LABELS_FOR_CLEANING: continue
                class_mask = (y_extracted_numerical == label_int_filt)
                if np.sum(class_mask) > 0:
                    centroids[label_int_filt] = np.mean(X_current_features[class_mask], axis=0)
                else: logging.warning(f"No samples for class '{label_str_filt}' for centroid calculation.")

            if len(centroids) < 2: logging.warning("Less than 2 centroids; margin filtering may be ineffective.")
            
            margins = []
            valid_indices_for_margin_calc = df_processed.index.tolist() # All rows initially
            
            for idx_val in valid_indices_for_margin_calc: # Iterate using index from df_processed
                # Get numerical label corresponding to this row in X_current_features
                current_label_int = y_extracted_numerical[idx_val] 
                current_label_str = int_to_label_map[current_label_int]

                if current_label_str not in EXPECTED_LABELS_FOR_CLEANING or current_label_int not in centroids:
                    margins.append(np.nan); continue
                
                d_own = euclidean(X_current_features[idx_val], centroids[current_label_int])
                d_others = [euclidean(X_current_features[idx_val], oc) for oli_c, oc in centroids.items() if oli_c != current_label_int]
                
                if not d_others: margins.append(0) 
                else: margins.append(np.min(d_others) - d_own)
            
            df_processed['margin'] = margins # Assign directly as lengths should match
            df_processed.dropna(subset=['margin'], inplace=True)


            plt.figure(figsize=(12, 7)) # Plot margins before filtering
            for label_str_plot in EXPECTED_LABELS_FOR_CLEANING:
                if label_str_plot in df_processed['label'].unique():
                    subset_margins = df_processed[df_processed['label'] == label_str_plot]['margin'].dropna()
                    if not subset_margins.empty: plt.hist(subset_margins, bins=30, alpha=0.7, label=f'Margins for {label_str_plot}')
            plt.title('XDF Extracted: Margin Distributions Before Filtering'); plt.xlabel('Margin'); plt.ylabel('Frequency'); plt.legend(); plt.grid(True);
            plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, 'xdf_margins_before_filtering.png')); plt.close()

            kept_indices_filt = df_processed[df_processed.apply(lambda row: row['label'] in MARGIN_THRESHOLDS_XDF and row.get('margin', np.nan) >= MARGIN_THRESHOLDS_XDF[row['label']], axis=1)].index
            df_cleaned = df_processed.loc[kept_indices_filt].copy()
            logging.info(f"Kept {len(df_cleaned)} records after margin filtering.")
            logging.info("Cleaned XDF class distribution:"); logging.info(df_cleaned['label'].value_counts())

            plt.figure(figsize=(12, 7)) # Plot margins after filtering
            for label_str_plot in EXPECTED_LABELS_FOR_CLEANING:
                 if label_str_plot in df_cleaned['label'].unique():
                    subset_margins_kept = df_cleaned[df_cleaned['label'] == label_str_plot]['margin'].dropna()
                    if not subset_margins_kept.empty: 
                        plt.hist(subset_margins_kept, bins=30, alpha=0.7, label=f'Kept Margins for {label_str_plot}')
                        if label_str_plot in MARGIN_THRESHOLDS_XDF: plt.axvline(MARGIN_THRESHOLDS_XDF[label_str_plot], linestyle='--', color='red', label=f'{label_str_plot} Thresh ({MARGIN_THRESHOLDS_XDF[label_str_plot]:.2f})')
            plt.title('XDF Extracted: Margin Distributions After Filtering'); plt.xlabel('Margin'); plt.ylabel('Frequency'); plt.legend(); plt.grid(True);
            plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, 'xdf_margins_after_filtering.png')); plt.close()
            df_processed = df_cleaned
    else:
        logging.info("Skipping margin-based filtering.")

    cleaned_output_path = os.path.join(PROCESSED_DATA_OUTPUT_DIR, CLEANED_JSONL_FILENAME)
    records_to_save_cleaned = []
    for _, row in df_processed.iterrows():
        records_to_save_cleaned.append({
            "features": row['features_for_margin'], "label": row['label'],
            "source_xdf": row['source_xdf'], "marker_timestamp": row['marker_timestamp']
        })
    with open(cleaned_output_path, 'w') as f:
        for record in records_to_save_cleaned: f.write(json.dumps(record) + '\n')
    logging.info(f"Saved {len(records_to_save_cleaned)} cleaned/processed records to: {cleaned_output_path}")

    if CREATE_SYNTHETIC_DATA_XDF and not df_processed.empty:
        logging.info("Performing data augmentation on XDF cleaned/processed data...")
        X_for_aug = np.array([list(row) for row in df_processed['features_for_margin']]) # Ensure it's a 2D numpy array
        y_numerical_for_aug = df_processed['label'].map(label_to_int_map).values
        
        synthetic_records = augment_data_with_noise(X_for_aug, y_numerical_for_aug, SYNTHETIC_SAMPLES_PER_CLEANED_REAL_XDF, NOISE_STD_DEV_RATIO_XDF, int_to_label_map)
        final_augmented_records = records_to_save_cleaned + synthetic_records
        
        augmented_output_path = os.path.join(PROCESSED_DATA_OUTPUT_DIR, AUGMENTED_JSONL_FILENAME)
        with open(augmented_output_path, 'w') as f:
            for record in final_augmented_records: f.write(json.dumps(record) + '\n')
        logging.info(f"Saved {len(final_augmented_records)} records (cleaned + synthetic) to: {augmented_output_path}")
        logging.info(f"  - Cleaned/processed real samples: {len(records_to_save_cleaned)}")
        logging.info(f"  - Generated synthetic samples: {len(synthetic_records)}")
    elif not df_processed.empty:
        logging.info("XDF Data augmentation is disabled. Copying cleaned file to augmented path.")
        if os.path.exists(cleaned_output_path):
            import shutil
            shutil.copy(cleaned_output_path, os.path.join(PROCESSED_DATA_OUTPUT_DIR, AUGMENTED_JSONL_FILENAME))
    else: logging.info("No data to augment or save as augmented (df_processed is empty).")

    if not df_processed.empty and len(df_processed['label'].unique()) > 1:
        try:
            X_for_pca_final = np.array([list(row) for row in df_processed['features_for_margin']])
            if X_for_pca_final.ndim == 1: X_for_pca_final = X_for_pca_final.reshape(1, -1) # Handle single sample case

            if X_for_pca_final.shape[0] > 1 : # PCA needs more than 1 sample
                pca_final = PCA(n_components=min(2, X_for_pca_final.shape[0], X_for_pca_final.shape[1])) # Adjust n_components
                X_pca_final_transformed = pca_final.fit_transform(X_for_pca_final)
                
                plt.figure(figsize=(12, 7))
                for label_str_pca in df_processed['label'].unique():
                    if label_str_pca not in EXPECTED_LABELS_FOR_CLEANING: continue
                    idx_pca = (df_processed['label'] == label_str_pca)
                    # Ensure X_pca_final_transformed has data for this label
                    if np.sum(idx_pca) > 0 and X_pca_final_transformed[idx_pca].shape[1] > 0:
                        plt.scatter(X_pca_final_transformed[idx_pca, 0], 
                                    X_pca_final_transformed[idx_pca, 1] if X_pca_final_transformed.shape[1] > 1 else np.zeros(np.sum(idx_pca)), 
                                    label=label_str_pca, alpha=0.7)
                plt.title('PCA of Processed XDF Data (Cleaned/Scaled)')
                plt.xlabel(f'PC1 ({pca_final.explained_variance_ratio_[0]*100:.2f}%)' if pca_final.explained_variance_ratio_.size > 0 else 'PC1')
                if pca_final.n_components_ > 1 and pca_final.explained_variance_ratio_.size > 1:
                    plt.ylabel(f'PC2 ({pca_final.explained_variance_ratio_[1]*100:.2f}%)')
                else:
                    plt.ylabel('PC2 (Not available or 0%)')
                plt.legend(); plt.grid(True);
                plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, 'xdf_pca_processed_data.png')); plt.close()
            else:
                logging.info("Not enough samples or components for PCA plot after processing.")

        except ImportError: logging.warning("sklearn.decomposition.PCA not available. Skipping final PCA plot.")
        except Exception as e: logging.error(f"Error during final PCA plotting: {e}", exc_info=True)

    logging.info("--- XDF Data Processing Pipeline Finished ---")

if __name__ == '__main__':
    process_all_xdf_files()
