import os
import librosa
import numpy as np
import h5py
from tqdm import tqdm
import soundfile as sf
import traceback

# --------- CONFIG --------
stego_path = r"/input/clean-mp3-files/"
sr = 22050
n_mfcc = 13
TARGET_VALID_FILES = 10400
INVALID_LOG = "invalid_stego_files.txt"
OUTPUT_H5 = "normal_features_MFCC.h5"
# --------------------------

def is_valid_audio(file_path, log_file=INVALID_LOG):
    try:
        with sf.SoundFile(file_path) as f:
            return True
    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"{file_path} – {e}\n")
        return False

def MFCC_Feature_Extractor(file_path):
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=5.0)  # load first 5 sec
        y_harmonic, _ = librosa.effects.hpss(y)
        mfcc = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=n_mfcc)
        return mfcc.T
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        traceback.print_exc()
        return None

def extract_valid_features(folder_path, label_desc, target_valid_files=TARGET_VALID_FILES):
    features = []
    valid_count = 0
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mp3')])
    
    print(f"Trying to collect {target_valid_files} valid features from {len(all_files)} files...")

    for file in tqdm(all_files, desc=label_desc):
        if valid_count >= target_valid_files:
            break

        path = os.path.join(folder_path, file)

        if not is_valid_audio(path):
            continue  # skip invalid audio

        feat = MFCC_Feature_Extractor(path)
        if feat is not None:
            features.append(feat)
            valid_count += 1

    print(f"Collected {valid_count} valid feature sets.")
    return features

def pad_and_save(features, filename):
    if not features:
        print("No features to save!")
        return

    print(f"Padding and saving {filename}...")
    max_len = max(f.shape[0] for f in features)
    feature_dim = features[0].shape[1]
    num_samples = len(features)

    with h5py.File(filename, "w") as f:
        X_dset = f.create_dataset("X", (num_samples, max_len, feature_dim), dtype='float32')
        f.attrs["max_len"] = max_len

        for i, feat in enumerate(features):
            padded_feat = np.pad(feat, ((0, max_len - feat.shape[0]), (0, 0)), mode='constant')
            X_dset[i] = padded_feat

    print(f"Saved: {filename}")

# ------------------ Run Extraction and Save ------------------
stego_features = extract_valid_features(clean_path, "Clean Files", target_valid_files=TARGET_VALID_FILES)
pad_and_save(clean_features, OUTPUT_H5)
