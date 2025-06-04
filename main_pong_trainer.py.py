# Filename: neuro_collector_trainer_psd_pbb.py
# Description: Flask app using WebSockets for low-latency Neuro-Collector BCI control,
#              with game-specific audio feedback and success-based data logging.
#              Includes PSD and PowerByBand LSL streaming.
#              Includes detailed timing logs and a padding workaround for filterer.py.

import os
import sys
import time
import numpy as np
import xgboost as xgb
import pickle
import json
from dotenv import load_dotenv
from neurosity import NeurositySDK
import logging
from threading import Thread, Lock, Event, Timer 
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from collections import deque
from datetime import datetime

# --- Audio Imports ---
from pysinewave import SineWave 

# --- Filterer Import ---
try:
    from filterer import Filterer, RingBufferSignal 
except ImportError:
    print("ERROR: Could not import Filterer or RingBufferSignal.")
    print("Please ensure 'filterer.py' is accessible (e.g., in the same directory).")
    sys.exit(1)

# --- LSL Imports ---
try:
    from pylsl import StreamInfo, StreamOutlet, cf_float32, cf_string
except ImportError:
    print("WARNING: pylsl not found. LSL streaming will be disabled.")
    StreamInfo = None
    StreamOutlet = None


# --- Configuration ---
ENV_PATH = r".\.env" 

MODEL_PATH = r".\models\3_class_eeg_xgb_model_20250525_104029.json"
SCALER_PATH = r".\models\3_class_eeg_scaler_20250525_104029.pkl"

COLLECTED_DATA_FOLDER = r".\training\training_data\success_data\round_two_success" 
SUCCESS_DATA_FILE = r'.\training\training_data\success_data\round_two_success\pong_successful_hits_new_3_class_model.jsonl'

# --- LSL Stream Configuration ---
LSL_MARKER_STREAM_NAME = 'NeuroCollectorEvents'
LSL_PSD_STREAM_NAME = 'NeurosityPSDEvents'
LSL_POWERBYBAND_STREAM_NAME = 'NeurosityPowerByBandEvents'
LSL_PSD_POWERBYBAND_SAMPLING_RATE = 4.0 # Hz, as per Neurosity SDK documentation for these metrics

# --- Label Mapping for the NEW 3-CLASS MODEL ---
COMMAND_NAME_TO_LABEL_INDEX = {
    "Rest": 0,
    "Left_Fist": 1,
    "Right_Fist": 2
}
REVERSE_LABEL_MAPPING = {v: k for k, v in COMMAND_NAME_TO_LABEL_INDEX.items()}
NUM_CLASSES = len(COMMAND_NAME_TO_LABEL_INDEX) 

REST_LABEL_INDEX = COMMAND_NAME_TO_LABEL_INDEX.get("Rest")
LEFT_FIST_CMD_LABEL_INDEX = COMMAND_NAME_TO_LABEL_INDEX.get("Left_Fist")
RIGHT_FIST_CMD_LABEL_INDEX = COMMAND_NAME_TO_LABEL_INDEX.get("Right_Fist")

if REST_LABEL_INDEX is None: print("ERROR: 'Rest' command missing from new 3-class mapping."); sys.exit(1)
if LEFT_FIST_CMD_LABEL_INDEX is None: print("ERROR: 'Left_Fist' command missing from new 3-class mapping."); sys.exit(1)
if RIGHT_FIST_CMD_LABEL_INDEX is None: print("ERROR: 'Right_Fist' command missing from new 3-class mapping."); sys.exit(1)

# --- Initialize LSL marker outlet ---
marker_outlet = None
if StreamInfo and StreamOutlet:
    try:
        marker_info_obj = StreamInfo(LSL_MARKER_STREAM_NAME, 'Markers', 1, 0, cf_string, f'{LSL_MARKER_STREAM_NAME}_id_1')
        marker_outlet = StreamOutlet(marker_info_obj)
        logging.info(f"LSL Marker Stream '{LSL_MARKER_STREAM_NAME}' initialized.")
    except Exception as lsl_e:
        logging.error(f"Failed to initialize LSL marker stream '{LSL_MARKER_STREAM_NAME}': {lsl_e}")
        marker_outlet = None
else:
    logging.warning("LSL marker streaming is disabled (pylsl not found or failed to init).")


# Session Timings
COMMAND_DURATION = 8
REST_DURATION = 8
TRANSITION_DURATION = 2

# EEG Settings
NB_CHAN = 8 
SFREQ = 256.0
SIGNAL_BUFFER_LENGTH_SECS = 8
SIGNAL_BUFFER_LENGTH = int(SFREQ * SIGNAL_BUFFER_LENGTH_SECS)
FILTER_LOW_HZ = 7.0
FILTER_HIGH_HZ = 30.0
NEW_COV_RATE = 2

# --- Logging Setup ---
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(threadName)s %(message)s")
logger = logging.getLogger(__name__) 
if not logger.handlers: 
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
else: 
    logger.setLevel(logging.DEBUG)


# --- Flask & SocketIO App Setup ---
app = Flask(__name__, template_folder= r".\html_templates")
app.config['SECRET_KEY'] = 'your_very_secret_key_here_neuro_collector'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# --- Global Variables & Locks ---
neurosity = None
model = None
scaler = None
filterer = None

# Neurosity Unsubscribe Handles
raw_unsubscribe = None
focus_unsubscribe = None
accelerometer_unsubscribe = None
psd_unsubscribe = None # New for PSD
powerbyband_unsubscribe = None # New for PowerByBand

# LSL Outlets and Initialization Flags
psd_outlet = None
psd_outlet_initialized = False
powerbyband_outlet = None
powerbyband_outlet_initialized = False
global_neurosity_channel_names = None # Will be populated from the first raw EEG packet
global_neurosity_device_id_short = "unknown_device" # Will be populated after login

data_processing_lock = Lock()
cov_counter = 0
FEATURE_BUFFER_MAXLEN = 30 
feature_buffer = deque(maxlen=FEATURE_BUFFER_MAXLEN) 
session_lock = Lock()
session_active = False
session_primary_command_name = None
session_primary_label_index = None
session_duration_total = 0
session_start_time = 0
session_data_buffer = []
session_thread = None
stop_session_flag = Event()
current_prompt = "Idle"
prompt_start_time = 0

# --- Game-Specific Audio Configuration (Chord-based) ---
FIXED_DURATION = 0.20  
AUDIO_DECIBELS = -9    
MIDI_D4 = 62
MIDI_FSHARP4 = 66 
MIDI_A4 = 69      
SINEWAVE_D4 = None
SINEWAVE_FSHARP4 = None
SINEWAVE_A4 = None

# --- Audio Helper Functions ---
def initialize_chord_audio():
    global SINEWAVE_D4, SINEWAVE_FSHARP4, SINEWAVE_A4
    try:
        SINEWAVE_D4 = SineWave(pitch=26, decibels=AUDIO_DECIBELS) # D4 (MIDI 62 maps to pitch value 26 in pysinewave if root is A4=440Hz=pitch 0)
        SINEWAVE_FSHARP4 = SineWave(pitch=30, decibels=AUDIO_DECIBELS) # F#4 (MIDI 66 maps to pitch 30)
        SINEWAVE_A4 = SineWave(pitch=33, decibels=AUDIO_DECIBELS) # A4 (MIDI 69 maps to pitch 33)
        logging.info(f"Initialized SineWave objects for notes: D4, F#4, A4 at {AUDIO_DECIBELS}dB")
    except Exception as e:
        logging.error(f"Failed to initialize pysinewave objects for chords: {e}", exc_info=True)
        SINEWAVE_D4, SINEWAVE_FSHARP4, SINEWAVE_A4 = None, None, None
        
def play_chord_feedback(chord_type: str):
    global SINEWAVE_D4, SINEWAVE_FSHARP4, SINEWAVE_A4, FIXED_DURATION
    notes_to_play = []
    active_chord_log_name = "Unknown Chord"
    if chord_type == "major_third":  
        active_chord_log_name = "Major Third (D+F#)"
        if SINEWAVE_D4 and SINEWAVE_FSHARP4: notes_to_play = [SINEWAVE_D4, SINEWAVE_FSHARP4]
        else: logging.warning("SineWave objects for Major Third chord not initialized."); return
    elif chord_type == "perfect_fifth": 
        active_chord_log_name = "Perfect Fifth (D+A)"
        if SINEWAVE_D4 and SINEWAVE_A4: notes_to_play = [SINEWAVE_D4, SINEWAVE_A4]
        else: logging.warning("SineWave objects for Perfect Fifth chord not initialized."); return
    else: logging.warning(f"Unknown chord_type requested: {chord_type}"); return
    if not notes_to_play: return
    try:
        for note_wave_object in notes_to_play: note_wave_object.play()
        def stop_chord_notes():
            for note_wave_object in notes_to_play: note_wave_object.stop()
        stop_timer = Timer(FIXED_DURATION, stop_chord_notes); stop_timer.start()
    except Exception as e: logging.error(f"Error playing {active_chord_log_name} chord: {e}", exc_info=True)

# --- Core Helper Functions ---
def load_dotenv_config():
    logging.info(f"Loading .env from: {ENV_PATH}")
    if not os.path.exists(ENV_PATH): logging.error(f".env file not found at {ENV_PATH}."); sys.exit(1)
    load_dotenv(dotenv_path=ENV_PATH)
    logging.info(".env file loaded.")

def load_model_and_scaler():
    global model, scaler
    model_loaded, scaler_loaded = False, False
    logging.info(f"Loading 3-class model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH): logging.warning(f"Model file NOT FOUND: {MODEL_PATH}.")
    else:
        try: model = xgb.Booster(); model.load_model(MODEL_PATH); logging.info("3-class XGBoost model loaded."); model_loaded = True
        except Exception as e: logging.error(f"FAILED to load 3-class model: {e}")
    logging.info(f"Loading scaler from: {SCALER_PATH}")
    if not os.path.exists(SCALER_PATH): logging.warning(f"Scaler file NOT FOUND: {SCALER_PATH}.")
    else:
        try:
            with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
            logging.info("Scaler loaded successfully."); scaler_loaded = True
        except Exception as e: logging.error(f"FAILED to load scaler: {e}")
    if not model_loaded or not scaler_loaded: logging.warning("Prediction will be DISABLED.")
    else: logging.info("Prediction ENABLED.")

def initialize_filterer():
    global filterer
    logging.info("Initializing Filterer...")
    try:
        logging.info(f"Initializing Filterer with nb_chan={NB_CHAN} to aim for an 8x8 covariance matrix (64 features)...")
        filterer = Filterer(filter_high=FILTER_HIGH_HZ, filter_low=FILTER_LOW_HZ,
                            nb_chan=NB_CHAN, 
                            sample_rate=SFREQ,
                            signal_buffer_length=SIGNAL_BUFFER_LENGTH)
        logging.info(f"Filterer initialized for {NB_CHAN} channels. Settings: Low {FILTER_LOW_HZ}Hz, High {FILTER_HIGH_HZ}Hz, SR {SFREQ}Hz, {SIGNAL_BUFFER_LENGTH_SECS}s buffer.")
    except Exception as e:
        logging.error(f"Filterer initialization FAILED: {e}", exc_info=True)
        sys.exit(1)

def connect_and_login():
    global neurosity, global_neurosity_device_id_short
    logging.info("--- Connecting to Neurosity Device ---")
    device_id = os.getenv("NEUROSITY_DEVICE_ID"); email = os.getenv("NEUROSITY_EMAIL"); password = os.getenv("NEUROSITY_PASSWORD")
    if not all([device_id, email, password]): logging.error("Neurosity credentials not found."); return False
    try:
        neurosity = NeurositySDK({"device_id": device_id}); logging.info(f"SDK initialized for device: {device_id}")
        global_neurosity_device_id_short = device_id.split('-')[-1] if device_id else "unknown_device" # Get last part of ID
        neurosity.login({"email": email, "password": password}); logging.info("Login request sent.")
        time.sleep(2); logging.info(">>> Neurosity LOGIN SUCCEEDED (assumed). <<<"); return True
    except Exception as e: logging.error(f">>> Neurosity LOGIN FAILED: {e} <<<"); neurosity = None; return False

# --- Saving Functions ---
def save_prompted_session_data(primary_command_name_str, data_buffer):
    if not data_buffer: logging.warning("Prompted session data save skipped: No data."); return False
    filepath = None
    try:
        os.makedirs(COLLECTED_DATA_FOLDER, exist_ok=True)
        sanitized_label = "".join(c if c.isalnum() else "_" for c in primary_command_name_str)
        filename = f"Session_{sanitized_label}_NeuroCollector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        filepath = os.path.join(COLLECTED_DATA_FOLDER, filename)
        logging.info(f"SAVING Prompted data ({len(data_buffer)} samples for '{primary_command_name_str}') to {filepath}...")
        with open(filepath, 'w') as f:
            for item in data_buffer:
                if isinstance(item.get("features"), np.ndarray): item["features"] = item["features"].tolist()
                f.write(json.dumps(item) + '\n')
        time.sleep(0.1) 
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0: logging.info(f"SAVE SUCCESSFUL: {filepath}"); return True
        else: logging.error(f"SAVE FAILED or empty file: {filepath}"); return False
    except Exception as e: logging.error(f"SAVE FAILED for {filepath}: {e}"); return False

def save_successful_hit_data(feature_vector, success_command_name_str, lsl_timestamp_ms=None):
    global marker_outlet
    if feature_vector is None or success_command_name_str is None: 
        logging.warning("Skipping save_successful_hit: Invalid input (feature_vector or command_name is None).")
        return
    
    success_label_to_save_in_file = f"Success_{success_command_name_str}"
    try:
        entry = {
            "features": feature_vector.tolist() if isinstance(feature_vector, np.ndarray) else list(feature_vector),
            "label": success_label_to_save_in_file
        }
        if lsl_timestamp_ms is not None: entry["lsl_timestamp_ms"] = lsl_timestamp_ms
        success_data_dir = os.path.dirname(SUCCESS_DATA_FILE)
        if success_data_dir: os.makedirs(success_data_dir, exist_ok=True)
        with open(SUCCESS_DATA_FILE, 'a') as f: f.write(json.dumps(entry) + '\n')
        logging.info(f"Appended successful game event: Label='{success_label_to_save_in_file}', LSL ts={lsl_timestamp_ms}ms to {SUCCESS_DATA_FILE}")
        
        if marker_outlet and lsl_timestamp_ms is not None:
            try: marker_outlet.push_sample([str(success_label_to_save_in_file)], lsl_timestamp_ms / 1000.0) 
            except Exception as me: logging.error(f"Failed to push LSL marker for '{success_label_to_save_in_file}': {me}")
    except Exception as e: logging.error(f"Failed to save successful game event data to {SUCCESS_DATA_FILE}: {e}", exc_info=True)

# --- EEG Data Parsing and Prediction (Raw EEG for BCI Control) ---
def parse_data_and_predict(brainwave_data):
    global cov_counter, filterer, model, scaler, feature_buffer, session_active, current_prompt
    global session_primary_label_index, session_data_buffer, global_neurosity_channel_names # Added global_neurosity_channel_names

    func_start_time = time.perf_counter()
    if filterer is None: return

    with data_processing_lock:
        processing_start_time = time.perf_counter()
        try:
            raw_data = brainwave_data.get('data')
            info = brainwave_data.get('info', {})
            start_time_ms_neurosity = info.get('startTime', time.time() * 1000)

            # --- Capture global channel names from first raw packet ---
            if global_neurosity_channel_names is None and 'channelNames' in info:
                global_neurosity_channel_names = info.get('channelNames')
                if global_neurosity_channel_names and len(global_neurosity_channel_names) == NB_CHAN:
                    logging.info(f"Captured global Neurosity channel names: {global_neurosity_channel_names}")
                else:
                    logging.warning(f"Could not capture valid global channel names from raw EEG. Expected {NB_CHAN}, got {global_neurosity_channel_names}")
                    global_neurosity_channel_names = [f"CH{i+1}" for i in range(NB_CHAN)] # Fallback
            # --- End capture ---

            if not raw_data or not isinstance(raw_data, list) or len(raw_data) == 0: return
            num_samples = len(raw_data[0]) if len(raw_data) > 0 and isinstance(raw_data[0], list) else 0
            if num_samples == 0: return

            parse_eeg_start_time = time.perf_counter()
            eeg_data_from_neurosity = np.zeros((NB_CHAN, num_samples))
            for i in range(NB_CHAN):
                if i < len(raw_data) and len(raw_data[i]) == num_samples: eeg_data_from_neurosity[i, :] = raw_data[i]
                else: eeg_data_from_neurosity[i, :] = 0
            
            timestamps_ms_array = np.linspace(start_time_ms_neurosity, start_time_ms_neurosity + (num_samples - 1) * (1000.0 / SFREQ), num_samples)
            parse_eeg_duration_ms = (time.perf_counter() - parse_eeg_start_time) * 1000
            data_to_send_to_filterer = eeg_data_from_neurosity
            if True: 
                num_dummy_channels = 2; dummy_padding = np.zeros((num_dummy_channels, num_samples))
                data_to_send_to_filterer = np.vstack((eeg_data_from_neurosity, dummy_padding))
            
            filter_start_time = time.perf_counter()
            filterer.partial_transform(data_to_send_to_filterer)
            filter_duration_ms = (time.perf_counter() - filter_start_time) * 1000
            cov_counter += 1
            if cov_counter >= NEW_COV_RATE:
                feature_block_start_time = time.perf_counter()
                current_server_processing_timestamp_ms = int(time.time() * 1000)
                cov_counter = 0; cov_start_time = time.perf_counter(); cov_matrix = filterer.get_cov()
                cov_duration_ms = (time.perf_counter() - cov_start_time) * 1000
                if cov_matrix is None or cov_matrix.size == 0: logging.debug("Cov matrix empty."); return
                features_flat = cov_matrix.flatten().reshape(1, -1); features_scaled = None; scale_duration_ms = 0
                expected_feature_len = NB_CHAN * NB_CHAN
                if scaler:
                    if features_flat.shape == (1, expected_feature_len):
                        scale_start_time = time.perf_counter()
                        try: features_scaled = scaler.transform(features_flat)
                        except Exception as scale_e: logging.error(f"Scaler transform error: {scale_e}", exc_info=True); return
                        scale_duration_ms = (time.perf_counter() - scale_start_time) * 1000
                    else: logging.warning(f"Feature shape mismatch for scaler: Exp (1,{expected_feature_len}), Got {features_flat.shape}."); return
                else: features_scaled = features_flat; logging.debug("Scaler not loaded, using unscaled features.")
                session_log_start_time = time.perf_counter()
                if features_scaled is not None:
                    with session_lock:
                        if session_active:
                            label_to_save_for_session = None
                            current_command_name_for_session = REVERSE_LABEL_MAPPING.get(session_primary_label_index)
                            if current_prompt == "COMMAND" and current_command_name_for_session in COMMAND_NAME_TO_LABEL_INDEX:
                                label_to_save_for_session = session_primary_label_index
                            elif current_prompt == "REST" and current_command_name_for_session == "Rest":
                                label_to_save_for_session = COMMAND_NAME_TO_LABEL_INDEX.get("Rest", REST_LABEL_INDEX) 
                            if label_to_save_for_session is not None:
                                session_data_buffer.append({"features": features_scaled[0].tolist(), "label": label_to_save_for_session})
                    last_sample_lsl_timestamp_ms = timestamps_ms_array[-1] if len(timestamps_ms_array) > 0 else current_server_processing_timestamp_ms
                    feature_buffer.append((last_sample_lsl_timestamp_ms, features_scaled[0])) 
                session_log_duration_ms = (time.perf_counter() - session_log_start_time) * 1000
                predict_duration_ms = 0; emit_duration_ms = 0; audio_duration_ms = 0
                if model and features_scaled is not None:
                    probabilities_dict = None; top_predicted_label_str = "Pred Error"; probs_array_for_audio = None
                    try:
                        predict_model_start_time = time.perf_counter(); dmatrix = xgb.DMatrix(features_scaled)
                        probabilities_array = model.predict(dmatrix)
                        predict_duration_ms = (time.perf_counter() - predict_model_start_time) * 1000
                        if probabilities_array is not None and probabilities_array.ndim == 2 and probabilities_array.shape[0] > 0 and probabilities_array.shape[1] == NUM_CLASSES:
                            probs_for_dict = probabilities_array[0]; probs_array_for_audio = probs_for_dict
                            probabilities_dict = {REVERSE_LABEL_MAPPING.get(i, f"Unk_{i}"): round(float(p), 3) for i, p in enumerate(probs_for_dict)}
                            top_predicted_label_str = REVERSE_LABEL_MAPPING.get(np.argmax(probs_for_dict), f"UnkIdx_{np.argmax(probs_for_dict)}")
                        else: top_predicted_label_str = "Pred Invalid Shape"; logging.warning(f"Prediction output shape error. Got {probabilities_array.shape if probabilities_array is not None else 'None'}")
                    except Exception as pred_e: logging.error(f"Prediction error: {pred_e}", exc_info=True); top_predicted_label_str = "Pred Exception"
                    chord_to_trigger = None
                    if probs_array_for_audio is not None:
                        prob_left = probs_array_for_audio[LEFT_FIST_CMD_LABEL_INDEX]; prob_right = probs_array_for_audio[RIGHT_FIST_CMD_LABEL_INDEX]
                        audio_thresh = 0.05
                        if prob_left > prob_right + audio_thresh: chord_to_trigger = "perfect_fifth"
                        elif prob_right > prob_left + audio_thresh: chord_to_trigger = "major_third"
                    emit_start_time = time.perf_counter()
                    socketio.emit('prediction_update', {"probabilities": probabilities_dict, "predicted_label": top_predicted_label_str, "executionTime": round(predict_duration_ms,2), "timestamp": current_server_processing_timestamp_ms})
                    emit_duration_ms = (time.perf_counter() - emit_start_time) * 1000
                    audio_proc_start_time = time.perf_counter()
                    if chord_to_trigger: play_chord_feedback(chord_to_trigger)
                    audio_duration_ms = (time.perf_counter() - audio_proc_start_time) * 1000
                feature_block_duration_ms = (time.perf_counter() - feature_block_start_time) * 1000
                logging.debug(f"TIMINGS (ms) | ParseEEG: {parse_eeg_duration_ms:.2f} | Filter: {filter_duration_ms:.2f} | Cov: {cov_duration_ms:.2f} | Scale: {scale_duration_ms:.2f} | SessLog: {session_log_duration_ms:.2f} | Predict: {predict_duration_ms:.2f} | Emit: {emit_duration_ms:.2f} | Audio: {audio_duration_ms:.2f} | TOTAL_FEAT_BLOCK: {feature_block_duration_ms:.2f}")
        except Exception as e: logging.error(f"Error in parse_data_and_predict main try: {e}", exc_info=True)
        finally:
            processing_duration_ms = (time.perf_counter() - processing_start_time) * 1000; func_duration_ms = (time.perf_counter() - func_start_time) * 1000
            logging.debug(f"parse_data_and_predict call (ms): TotalFunc={func_duration_ms:.2f}, LockHeld={processing_duration_ms:.2f}")

# --- NEW: PSD Data Handling and LSL Streaming ---
def handle_psd_data(psd_data_packet):
    global psd_outlet, psd_outlet_initialized, global_neurosity_channel_names, NB_CHAN, global_neurosity_device_id_short
    
    if not StreamInfo or not StreamOutlet: return # LSL not available

    try:
        if not psd_outlet_initialized:
            if global_neurosity_channel_names is None:
                logging.warning("PSD LSL: Global channel names not yet available. Skipping initialization.")
                return

            info = psd_data_packet.get('info', {})
            sdk_sampling_rate = info.get('samplingRate', SFREQ) # Default to raw SFREQ if not in PSD info
            sdk_notch_freq = info.get('notchFrequency', "Unknown")
            
            freq_bins = psd_data_packet.get('freqs')
            if not freq_bins or not isinstance(freq_bins, list):
                logging.error("PSD LSL: 'freqs' field missing or invalid in PSD data. Cannot initialize stream.")
                return
            num_freq_bins = len(freq_bins)

            lsl_channel_count = NB_CHAN * num_freq_bins
            lsl_channel_labels = [f"{ch_name}_Freq_{freq_val:.0f}Hz" 
                                  for ch_name in global_neurosity_channel_names 
                                  for freq_val in freq_bins]

            source_id_str = f"{LSL_PSD_STREAM_NAME}_{global_neurosity_device_id_short}"
            
            psd_stream_info = StreamInfo(
                name=LSL_PSD_STREAM_NAME,
                type='EEG_PSD',
                channel_count=lsl_channel_count,
                nominal_srate=LSL_PSD_POWERBYBAND_SAMPLING_RATE,
                channel_format=cf_float32,
                source_id=source_id_str
            )
            
            desc = psd_stream_info.desc()
            desc.append_child_value("neurosity_sdk_sampling_rate", str(sdk_sampling_rate))
            desc.append_child_value("neurosity_sdk_notch", str(sdk_notch_freq))
            # You can add more metadata from psd_data_packet.info if needed
            channels_node = desc.append_child("channels")
            for label_idx, label_val in enumerate(lsl_channel_labels):
                ch = channels_node.append_child("channel")
                ch.append_child_value("label", label_val)
                ch.append_child_value("unit", "microVolts^2/Hz") # Or appropriate unit for PSD
                ch.append_child_value("type", "EEG_PSD_Component")
                ch.append_child_value("original_channel_index_neurosity", str(label_idx // num_freq_bins)) # 0-indexed channel
                ch.append_child_value("frequency_bin_hz", str(freq_bins[label_idx % num_freq_bins]))


            psd_outlet = StreamOutlet(psd_stream_info)
            psd_outlet_initialized = True
            logging.info(f"LSL Stream '{LSL_PSD_STREAM_NAME}' initialized with {lsl_channel_count} channels. Labels example: {lsl_channel_labels[0]}")

        # Process and push data
        if psd_outlet:
            psd_matrix = psd_data_packet.get('psd') # List of lists: [channel][psd_value_for_freq_bin]
            if not psd_matrix or len(psd_matrix) != NB_CHAN or (NB_CHAN > 0 and len(psd_matrix[0]) != len(psd_data_packet.get('freqs',[]))):
                logging.warning(f"PSD LSL: PSD matrix data is missing, malformed, or channel/freq count mismatch. Expected {NB_CHAN} channels and {len(psd_data_packet.get('freqs',[]))} freqs.")
                return

            flat_psd_data = [value for channel_data in psd_matrix for value in channel_data]
            
            timestamp_ms = psd_data_packet.get('info', {}).get('startTime', time.time() * 1000)
            lsl_timestamp_sec = timestamp_ms / 1000.0
            
            psd_outlet.push_sample(flat_psd_data, timestamp=lsl_timestamp_sec)
            # logging.debug(f"Pushed PSD sample to LSL at {lsl_timestamp_sec:.3f}")

    except Exception as e:
        logging.error(f"Error in handle_psd_data: {e}", exc_info=True)
        if psd_outlet_initialized: # If error after init, try to reset to allow re-init
            logging.warning("Resetting PSD LSL stream due to error.")
            psd_outlet_initialized = False 
            psd_outlet = None


# --- NEW: PowerByBand Data Handling and LSL Streaming ---
def handle_powerbyband_data(pbb_data_packet):
    global powerbyband_outlet, powerbyband_outlet_initialized, global_neurosity_channel_names, NB_CHAN, global_neurosity_device_id_short

    if not StreamInfo or not StreamOutlet: return # LSL not available

    try:
        if not powerbyband_outlet_initialized:
            if global_neurosity_channel_names is None:
                logging.warning("PowerByBand LSL: Global channel names not yet available. Skipping PBB initialization.")
                return

            # Attempt to get info from standard location, or use fallback for timestamp if missing
            pbb_info = pbb_data_packet.get('info', {}) # Assume info might be present
            sdk_sampling_rate = pbb_info.get('samplingRate', SFREQ) 
            sdk_notch_freq = pbb_info.get('notchFrequency', "Unknown")

            bands_data_dict = pbb_data_packet.get('data') # e.g. {'alpha': [ch1, ch2,...], 'beta': [...]}
            if not bands_data_dict or not isinstance(bands_data_dict, dict):
                logging.error("PowerByBand LSL: 'data' field missing or invalid. Cannot initialize stream.")
                return
            
            # Ensure consistent band order for LSL stream definition
            # Standard bands: delta, theta, alpha, beta, gamma. Others might exist.
            standard_bands_ordered = ["delta", "theta", "alpha", "beta", "gamma"]
            available_band_names = sorted(list(bands_data_dict.keys()))
            
            # Use standard order if bands are present, otherwise alphabetical of available
            ordered_band_names_for_stream = [b for b in standard_bands_ordered if b in available_band_names]
            ordered_band_names_for_stream += [b for b in available_band_names if b not in standard_bands_ordered]


            if not ordered_band_names_for_stream:
                 logging.error("PowerByBand LSL: No band data found. Cannot initialize stream.")
                 return
            num_bands = len(ordered_band_names_for_stream)

            lsl_channel_count = NB_CHAN * num_bands
            lsl_channel_labels = [f"{ch_name}_{band_name}_power" 
                                  for ch_name in global_neurosity_channel_names 
                                  for band_name in ordered_band_names_for_stream]
            
            source_id_str = f"{LSL_POWERBYBAND_STREAM_NAME}_{global_neurosity_device_id_short}"

            pbb_stream_info = StreamInfo(
                name=LSL_POWERBYBAND_STREAM_NAME,
                type='EEG_PowerByBand',
                channel_count=lsl_channel_count,
                nominal_srate=LSL_PSD_POWERBYBAND_SAMPLING_RATE,
                channel_format=cf_float32,
                source_id=source_id_str
            )
            
            desc = pbb_stream_info.desc()
            desc.append_child_value("neurosity_sdk_sampling_rate", str(sdk_sampling_rate)) # May be from fallback
            desc.append_child_value("neurosity_sdk_notch", str(sdk_notch_freq)) # May be from fallback

            channels_node = desc.append_child("channels")
            for label_idx, label_val in enumerate(lsl_channel_labels):
                ch = channels_node.append_child("channel")
                ch.append_child_value("label", label_val)
                ch.append_child_value("unit", "microVolts^2_avg") # Or appropriate unit for power
                ch.append_child_value("type", "EEG_PowerBand_Component")
                ch.append_child_value("original_channel_index_neurosity", str(label_idx // num_bands))
                ch.append_child_value("band_name", str(ordered_band_names_for_stream[label_idx % num_bands]))


            powerbyband_outlet = StreamOutlet(pbb_stream_info)
            powerbyband_outlet_initialized = True
            logging.info(f"LSL Stream '{LSL_POWERBYBAND_STREAM_NAME}' initialized with {lsl_channel_count} channels. Bands: {ordered_band_names_for_stream}. Labels example: {lsl_channel_labels[0]}")

        # Process and push data
        if powerbyband_outlet:
            bands_data_dict = pbb_data_packet.get('data')
            if not bands_data_dict: logging.warning("PowerByBand LSL: No data in packet."); return

            # Re-fetch ordered band names for data extraction consistency
            standard_bands_ordered = ["delta", "theta", "alpha", "beta", "gamma"]
            available_band_names = sorted(list(bands_data_dict.keys()))
            ordered_band_names_for_data = [b for b in standard_bands_ordered if b in available_band_names]
            ordered_band_names_for_data += [b for b in available_band_names if b not in standard_bands_ordered]

            flat_pbb_data = []
            for band_name in ordered_band_names_for_data:
                channel_powers = bands_data_dict.get(band_name)
                if channel_powers and len(channel_powers) == NB_CHAN:
                    flat_pbb_data.extend(channel_powers)
                else:
                    logging.warning(f"PowerByBand LSL: Data for band '{band_name}' missing or channel count mismatch. Expected {NB_CHAN}. Filling with zeros.")
                    flat_pbb_data.extend([0.0] * NB_CHAN) # Fill with zeros to maintain stream structure

            # Timestamp handling: Prioritize info.startTime, fallback to time.time()
            pbb_info = pbb_data_packet.get('info', {})
            timestamp_ms = pbb_info.get('startTime')
            if timestamp_ms is None:
                timestamp_ms = time.time() * 1000
                # logging.debug("PowerByBand LSL: Using server time for timestamp as 'startTime' not found in PBB packet info.")
            
            lsl_timestamp_sec = timestamp_ms / 1000.0
            
            powerbyband_outlet.push_sample(flat_pbb_data, timestamp=lsl_timestamp_sec)
            # logging.debug(f"Pushed PowerByBand sample to LSL at {lsl_timestamp_sec:.3f}")

    except Exception as e:
        logging.error(f"Error in handle_powerbyband_data: {e}", exc_info=True)
        if powerbyband_outlet_initialized: # If error after init, try to reset
            logging.warning("Resetting PowerByBand LSL stream due to error.")
            powerbyband_outlet_initialized = False
            powerbyband_outlet = None

# --- Session Management Thread ---
def session_manager():
    global session_active, current_prompt, prompt_start_time, session_data_buffer, session_primary_command_name, session_primary_label_index, session_start_time, session_duration_total
    logging.info(f"Session Manager started for '{session_primary_command_name}' (Idx: {session_primary_label_index}). Duration: {session_duration_total}s")
    session_error = None; next_prompt = "COMMAND"; prompt_duration = TRANSITION_DURATION
    try:
        with session_lock: current_prompt = "STARTING_TRANSITION"; prompt_start_time = time.time()
        while True:
            if stop_session_flag.is_set(): logging.info("SessionMan: Stop signal."); break
            now = time.time(); time_in_prompt = now - prompt_start_time; total_elapsed_time = now - session_start_time
            if total_elapsed_time >= session_duration_total: logging.info("SessionMan: Duration reached."); break
            if time_in_prompt >= prompt_duration:
                with session_lock:
                    if not session_active or stop_session_flag.is_set(): logging.info("SessionMan: Inactive/stopped."); break
                    current_prompt = next_prompt; prompt_start_time = now
                    logging.info(f"SessionMan: New Prompt: {current_prompt} for '{session_primary_command_name}'")
                    if current_prompt == "COMMAND": next_prompt = "TRANSITION_TO_REST"; prompt_duration = COMMAND_DURATION
                    elif current_prompt == "TRANSITION_TO_REST": next_prompt = "REST"; prompt_duration = TRANSITION_DURATION
                    elif current_prompt == "REST": next_prompt = "TRANSITION_TO_COMMAND"; prompt_duration = REST_DURATION
                    elif current_prompt == "TRANSITION_TO_COMMAND": next_prompt = "COMMAND"; prompt_duration = TRANSITION_DURATION
            time.sleep(0.05) 
    except Exception as e: logging.error(f"Session manager error: {e}", exc_info=True); session_error = e
    finally:
        logging.info("SessionMan: Ending..."); saved_successfully = False; cmd_saved_name = None; buf_size = 0
        with session_lock:
            if session_error: current_prompt = "ERROR_IN_SESSION"
            elif stop_session_flag.is_set(): current_prompt = "STOPPED_BY_USER"
            else: current_prompt = "SAVING_SESSION_DATA"
            session_active = False; cmd_saved_name = session_primary_command_name
            buffer_to_save = list(session_data_buffer); buf_size = len(buffer_to_save); session_data_buffer.clear()
        if cmd_saved_name and buf_size > 0: saved_successfully = save_prompted_session_data(cmd_saved_name, buffer_to_save)
        elif cmd_saved_name: logging.warning(f"Session data for '{cmd_saved_name}' empty, not saved.")
        with session_lock:
            if current_prompt not in ["ERROR_IN_SESSION", "STOPPED_BY_USER"]: current_prompt = "Idle" if saved_successfully else "Save_Failed"
            session_primary_command_name = None; session_primary_label_index = None; session_start_time = 0; session_duration_total = 0
        stop_session_flag.clear()
        logging.info(f"SessionMan finished. Saved: {saved_successfully}, Cmd: {cmd_saved_name}, Samples: {buf_size}")

# --- Focus & Motion Callbacks ---
def update_focus(focus_data):
    try:
        score = focus_data.get("probability", 0.0); timestamp = focus_data.get("timestamp", time.time()*1000)
        socketio.emit('focus_update', {"score": round(score,3), "timestamp": int(timestamp)})
    except Exception as e: logging.error(f"Error processing focus: {e}")

def update_motion_or_accel(data): 
    try:
        if "accelX" in data and "accelY" in data and "accelZ" in data: 
            accel = {"x":round(data["accelX"],3),"y":round(data["accelY"],3),"z":round(data["accelZ"],3)}
            socketio.emit('motion_update', {"accel":accel, "timestamp":data.get("timestamp",time.time()*1000)})
    except Exception as e: logging.error(f"Error processing motion/accel: {e}")

# --- Flask Routes ---
@app.route('/')
def index():
    try: return render_template('index.html', available_commands=list(COMMAND_NAME_TO_LABEL_INDEX.keys()))
    except Exception as e: logging.error(f"Error rendering index.html: {e}", exc_info=True); return "Error rendering dashboard.", 500
@app.route('/pong')
def pong_game():
    try: return render_template('neuro_pong.html')
    except Exception as e: logging.error(f"Error rendering alternate_neuro_pong.html: {e}", exc_info=True); return "Error rendering Pong.", 500
#@app.route('/neuro_collector')
#def neuro_collector_game():
#    try: return render_template('neuro_collector.html')
#    except Exception as e: logging.error(f"Error rendering neuro_collector.html: {e}", exc_info=True); return "Error rendering Neuro-Collector.", 500

# --- WebSocket Event Handlers ---
@socketio.on('connect')
def handle_connect(): 
    logging.info(f"Client connected: {request.sid}")
    emit('connection_ack', {'message': 'Backend connected successfully!'}) 

@socketio.on('disconnect')
def handle_disconnect(): 
    logging.info(f"Client disconnected: {request.sid}")

@socketio.on('start_session')
def handle_start_session(data):
    global session_active, session_thread, session_primary_command_name, session_primary_label_index, session_duration_total, session_start_time, current_prompt, stop_session_flag, session_data_buffer
    with session_lock:
        if session_active: 
            logging.warning("Session start denied: another session is already active.")
            emit('session_status',{'active':True,'message':'Session already active.'}) 
            return
        cmd = data.get('command'); dur_str = data.get('duration')
        if not cmd or cmd not in COMMAND_NAME_TO_LABEL_INDEX: 
            logging.error(f"Invalid command received for session start: {cmd}")
            emit('session_status',{'active':False,'message':f'Invalid command: {cmd}'}); return
        try:
            dur_sec = int(dur_str)
            if not (10 <= dur_sec <= 600): raise ValueError("Duration out of range 10-600s")
        except (TypeError, ValueError) as e: 
            logging.error(f"Invalid duration for session: {dur_str}. Error: {e}")
            emit('session_status',{'active':False,'message':f'Invalid duration: {dur_str}.'}); return
        
        session_active = True; session_primary_command_name = cmd
        session_primary_label_index = COMMAND_NAME_TO_LABEL_INDEX[cmd]
        session_duration_total = dur_sec; session_start_time = time.time()
        current_prompt = "Initializing..."; stop_session_flag.clear(); session_data_buffer.clear()
        session_thread = Thread(target=session_manager, name="NeuroCollectorSessionThread", daemon=True); session_thread.start()
        logging.info(f"Started session: '{cmd}' (Idx:{session_primary_label_index}), Duration:{dur_sec}s.")
        emit('session_status', {'active':True,'message':f'Session for {cmd} ({dur_sec}s) started. Prompt: {current_prompt}'})

@socketio.on('stop_session_request')
def handle_stop_session_request():
    global session_active, stop_session_flag, current_prompt
    logging.info("Stop session request received from client.")
    with session_lock:
        if session_active: 
            stop_session_flag.set(); current_prompt = "STOPPING_SESSION" 
            logging.info("Stop flag set for active session.")
            emit('session_status',{'active':True,'message':'Session stopping...'})
        else: 
            logging.info("No active session to stop.")
            emit('session_status',{'active':False,'message':'No active session.'})

@socketio.on('success_signal') 
def handle_pong_success_signal(data):
    global feature_buffer, data_processing_lock 
    success_command_name = data.get('command'); client_hit_timestamp = data.get('hit_timestamp') 
    if not success_command_name:
        logging.warning("Received 'success_signal' without 'command'. Cannot log.")
        emit('game_event_response', {'status': 'error', 'message': 'Missing command in success_signal data'}); return
    logging.info(f"Received 'success_signal'. Command: {success_command_name}, Client Hit TS: {client_hit_timestamp}")
    latest_feature_vector = None; latest_lsl_timestamp_ms = None
    with data_processing_lock: 
        if feature_buffer: latest_lsl_timestamp_ms, latest_feature_vector_data = feature_buffer[-1]; latest_feature_vector = latest_feature_vector_data
        else: logging.warning("Feature buffer empty. Cannot save Pong hit data."); emit('game_event_response', {'status': 'error', 'message': 'Feature buffer empty'}); return
    if latest_feature_vector is not None:
        try:
            save_successful_hit_data(feature_vector=latest_feature_vector, success_command_name_str=success_command_name, lsl_timestamp_ms=latest_lsl_timestamp_ms)
            emit('game_event_response', {'status': 'success', 'message': f'Pong success for {success_command_name} logged.'})
        except Exception as e:
            logging.error(f"Error processing 'success_signal' for '{success_command_name}': {e}", exc_info=True)
            emit('game_event_response', {'status': 'error', 'message': f'Server error logging Pong success: {str(e)}'})
    else:
        logging.warning(f"Could not log Pong success. Features available: {latest_feature_vector is not None}, Command: {success_command_name}")
        emit('game_event_response', {'status': 'error', 'message': 'Feature vector not available for logging'})

# --- Neurosity Streaming Function ---
def neurosity_stream_runner():
    global raw_unsubscribe, focus_unsubscribe, accelerometer_unsubscribe, neurosity
    global psd_unsubscribe, powerbyband_unsubscribe # Added new unsubscribe handles

    if not neurosity: logging.error("Neurosity SDK not initialized. Cannot start streaming thread."); return
    
    logging.info("Starting Neurosity stream runner thread...")
    while True: 
        local_raw_unsub, local_focus_unsub, local_accel_unsub = None, None, None
        local_psd_unsub, local_pbb_unsub = None, None # Added new local handles
        all_subs_active = False
        try:
            logging.info("Attempting to subscribe to Neurosity streams...")
            try: 
                local_raw_unsub = neurosity.brainwaves_raw(parse_data_and_predict)
                raw_unsubscribe = local_raw_unsub 
                logging.info("Subscribed to raw EEG brainwaves.")
            except Exception as e: logging.error(f"Failed to subscribe to raw EEG: {e}", exc_info=True)

            #try: 
            #    local_focus_unsub = neurosity.focus(update_focus)
            #    focus_unsubscribe = local_focus_unsub 
            #    logging.info("Subscribed to focus.")
            #except Exception as e: logging.error(f"Failed to subscribe to focus: {e}", exc_info=True)
            
            #try:
            #    if hasattr(neurosity,'accelerometer') and callable(neurosity.accelerometer):
            #        local_accel_unsub = neurosity.accelerometer(update_motion_or_accel)
            #        accelerometer_unsubscribe = local_accel_unsub 
            #        logging.info("Subscribed to accelerometer.")
            #    else: logging.warning("Neurosity SDK no '.accelerometer()'. Skipping.")
            #except Exception as e: logging.error(f"Failed to subscribe to accelerometer: {e}", exc_info=True)

            # --- NEW: Subscribe to PSD ---
            if StreamInfo and StreamOutlet: # Only if LSL is available
                try:
                    local_psd_unsub = neurosity.brainwaves_psd(handle_psd_data)
                    psd_unsubscribe = local_psd_unsub
                    logging.info("Subscribed to PSD brainwaves for LSL streaming.")
                except Exception as e: logging.error(f"Failed to subscribe to PSD brainwaves: {e}", exc_info=True)
            else: logging.warning("LSL not available, skipping PSD subscription.")
            # --- END NEW PSD ---

            # --- NEW: Subscribe to PowerByBand ---
            if StreamInfo and StreamOutlet: # Only if LSL is available
                try:
                    local_pbb_unsub = neurosity.brainwaves_power_by_band(handle_powerbyband_data)
                    powerbyband_unsubscribe = local_pbb_unsub
                    logging.info("Subscribed to PowerByBand brainwaves for LSL streaming.")
                except Exception as e: logging.error(f"Failed to subscribe to PowerByBand brainwaves: {e}", exc_info=True)
            else: logging.warning("LSL not available, skipping PowerByBand subscription.")
            # --- END NEW PowerByBand ---

            if not local_raw_unsub: 
                logging.error("Essential raw EEG subscription failed. Will retry connection loop.")
                time.sleep(15); continue 

            logging.info(">>> Neurosity streams subscribed (raw EEG essential, others optional). Waiting for data... <<<")
            all_subs_active = True 
            while all_subs_active: time.sleep(30) 
        except Exception as e: 
            logging.error(f"Major error in Neurosity stream runner loop: {e}. Retrying connection...", exc_info=True)
            all_subs_active = False 
        finally:
            logging.info("Neurosity runner: Cleaning up current subscriptions before retry or exit...")
            if local_raw_unsub:
                try:
                    local_raw_unsub()
                    logging.info("Raw EEG unsubscribed.")
                except Exception as ue:
                    logging.error(f"Error unsubscribing raw EEG: {ue}")
            #if local_focus_unsub:
            #    try:
            #        local_focus_unsub()
            #        logging.info("Focus unsubscribed.")
            #    except Exception as ue:
            #        logging.error(f"Error unsubscribing focus: {ue}")
            #if local_accel_unsub:
            #    try:
            #        local_accel_unsub()
            #        logging.info("Accelerometer unsubscribed.")
            #    except Exception as ue:
            #        logging.error(f"Error unsubscribing accelerometer: {ue}")
            
            # --- NEW: Unsubscribe PSD and PowerByBand ---
            if local_psd_unsub:
                try:
                    local_psd_unsub()
                    logging.info("PSD brainwaves unsubscribed.")
                except Exception as ue:
                    logging.error(f"Error unsubscribing PSD: {ue}")
            if local_pbb_unsub:
                try:
                    local_pbb_unsub()
                    logging.info("PowerByBand brainwaves unsubscribed.")
                except Exception as ue:
                    logging.error(f"Error unsubscribing PowerByBand: {ue}")
            # --- END NEW Unsubscribe ---
            
            raw_unsubscribe, focus_unsubscribe, accelerometer_unsubscribe = None, None, None
            psd_unsubscribe, powerbyband_unsubscribe = None, None # Clear global new ones too
            
            if not all_subs_active: 
                logging.info("Waiting 15 seconds before re-attempting Neurosity stream subscriptions...")
                time.sleep(15)

# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- Starting Neuro-Collector Trainer (Log Level: {logging.getLevelName(logger.getEffectiveLevel())}) ---")
    print(f">>> Model: {MODEL_PATH} for {NUM_CLASSES} classes: {list(COMMAND_NAME_TO_LABEL_INDEX.keys())} <<<")
    print(f">>> Audio: Chords based on D4 (MIDI {MIDI_D4}), F#4 (MIDI {MIDI_FSHARP4}), A4 (MIDI {MIDI_A4}) <<<")
    print(f">>> Success Hits Log: {SUCCESS_DATA_FILE} <<<")
    print(f">>> LSL Streams: Markers='{LSL_MARKER_STREAM_NAME}', PSD='{LSL_PSD_STREAM_NAME}', PowerByBand='{LSL_POWERBYBAND_STREAM_NAME}' at {LSL_PSD_POWERBYBAND_SAMPLING_RATE} Hz <<<")


    load_dotenv_config()
    load_model_and_scaler()
    initialize_filterer()
    initialize_chord_audio() 
    
    if not connect_and_login(): 
        print("\nERROR: Exiting: Neurosity connection failed.")
        sys.exit(1)
    
    Thread(target=neurosity_stream_runner, name="NeurosityStreamThread", daemon=True).start()
    
    logging.info("Starting Flask-SocketIO server for Neuro-Collector...")
    print("-----------------------------------------------------")
    print(">>> Dashboard UI at: http://127.0.0.1:5000/ <<<")
    print(">>> Neuro-Collector Game at: http://127.0.0.1:5000/neuro_collector <<<")
    if '/pong' in [r.rule for r in app.url_map.iter_rules()]: print(">>> Pong Game at: http://127.0.0.1:5000/pong <<<")
    print("-----------------------------------------------------")
    
    try:
        socketio.run(app, host='127.0.0.1', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt: logging.info("Server stopped by user (KeyboardInterrupt).")
    except OSError as e:
        if "address already in use" in str(e).lower(): logging.error(f"FATAL: Port 5000 already in use. (OSError: {e})")
        else: logging.error(f"FATAL: Server OSError: {e}")
    except Exception as e: logging.error(f"FATAL: Server Exception: {e}", exc_info=True)
    finally:
        logging.info("Shutdown sequence initiated..."); print("\nShutting down... please wait.")
        stop_session_flag.set() 
        if session_thread and session_thread.is_alive():
            logging.info("Waiting for session manager thread to join...")
            session_thread.join(timeout=5)
            if session_thread.is_alive(): logging.warning("Session manager thread did not join in time.")
        
        if raw_unsubscribe:
            try:
                raw_unsubscribe()
                logging.info("Final raw EEG unsubscribed.")
            except Exception as e:
                logging.error(f"Error during final raw EEG unsubscribe: {e}")
        if focus_unsubscribe:
            try:
                focus_unsubscribe()
                logging.info("Final focus unsubscribed.")
            except Exception as e:
                logging.error(f"Error during final focus unsubscribe: {e}")
        if accelerometer_unsubscribe:
            try:
                accelerometer_unsubscribe()
                logging.info("Final accelerometer unsubscribed.")
            except Exception as e:
                logging.error(f"Error during final accelerometer unsubscribe: {e}")
        
        # --- NEW: Final Unsubscribe for PSD and PowerByBand ---
        if psd_unsubscribe:
            try:
                psd_unsubscribe()
                logging.info("Final PSD brainwaves unsubscribed.")
            except Exception as e:
                logging.error(f"Error during final PSD unsubscribe: {e}")
        if powerbyband_unsubscribe:
            try:
                powerbyband_unsubscribe()
                logging.info("Final PowerByBand brainwaves unsubscribed.")
            except Exception as e:
                logging.error(f"Error during final PowerByBand unsubscribe: {e}")
        # --- END NEW Final Unsubscribe ---

        if neurosity and hasattr(neurosity,'disconnect') and callable(neurosity.disconnect):
            try: logging.info("Disconnecting Neurosity SDK..."); neurosity.disconnect(); logging.info("Neurosity SDK Disconnected.")
            except Exception as de: logging.error(f"Error during Neurosity SDK disconnect: {de}")
        
        logging.info("Shutdown complete."); print("Shutdown complete.")
