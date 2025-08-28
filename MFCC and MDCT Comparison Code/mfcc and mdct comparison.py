import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# ------------------------------
# Utility to create data generator
# ------------------------------
class H5DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, normal_file, stego_file, batch_size=32, test_size=0.2, mode='train', scaler=None):
        self.batch_size = batch_size
        self.normal_file = normal_file
        self.stego_file = stego_file
        self.normal_dataset_name = self._find_dataset(normal_file)
        self.stego_dataset_name = self._find_dataset(stego_file)

        with h5py.File(normal_file, "r") as f:
            self.n_normal = len(f[self.normal_dataset_name])
            self.n_features = f[self.normal_dataset_name].shape[-1]
            self.normal_max_len = f[self.normal_dataset_name].shape[1] if len(f[self.normal_dataset_name].shape) == 3 else 1

        with h5py.File(stego_file, "r") as f:
            self.n_stego = len(f[self.stego_dataset_name])
            self.stego_max_len = f[self.stego_dataset_name].shape[1] if len(f[self.stego_dataset_name].shape) == 3 else 1

        self.max_len = max(self.normal_max_len, self.stego_max_len)
        if self.max_len < 2: self.max_len = 2
        self.total_samples = self.n_normal + self.n_stego
        indices = np.arange(self.total_samples)
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
        self.indices = train_idx if mode == 'train' else test_idx

        if scaler is None:
            self.scaler = StandardScaler()
            self._fit_scaler()
        else:
            self.scaler = scaler

    def _find_dataset(self, filepath):
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    return key
        raise ValueError("No dataset found.")

    def _fit_scaler(self):
        samples = []
        with h5py.File(self.normal_file, 'r') as f:
            samples.append(f[self.normal_dataset_name][:1000].reshape(-1, self.n_features))
        with h5py.File(self.stego_file, 'r') as f:
            samples.append(f[self.stego_dataset_name][:1000].reshape(-1, self.n_features))
        self.scaler.fit(np.vstack(samples))

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        X = np.zeros((len(batch_idx), self.max_len, self.n_features), dtype=np.float32)
        y = np.zeros(len(batch_idx), dtype=np.int32)

        for i, index in enumerate(batch_idx):
            if index < self.n_normal:
                with h5py.File(self.normal_file, 'r') as f:
                    data = f[self.normal_dataset_name][index]
                label = 0
            else:
                with h5py.File(self.stego_file, 'r') as f:
                    data = f[self.stego_dataset_name][index - self.n_normal]
                label = 1

            if len(data.shape) == 1:
                data = data.reshape(1, -1)

            padded = np.zeros((self.max_len, self.n_features), dtype=np.float32)
            padded[:data.shape[0], :] = data
            X[i] = self.scaler.transform(padded)
            y[i] = label

        return X, y

# ------------------------------
# SE-ResNet Architecture
# ------------------------------
def squeeze_excite_block(x, ratio=16):
    filters = x.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling1D()(x)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, filters))(se)
    return tf.keras.layers.Multiply()([x, se])

def residual_se_block(x, filters, stride=1):
    shortcut = x
    x = tf.keras.layers.Conv1D(filters, 3, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv1D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = squeeze_excite_block(x)

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = tf.keras.layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.ReLU()(x)

def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(3, strides=2, padding='same')(x)

    x = residual_se_block(x, 64)
    x = residual_se_block(x, 64)
    x = residual_se_block(x, 128, stride=2)
    x = residual_se_block(x, 128)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ------------------------------
# Evaluate and collect metrics
# ------------------------------
def evaluate_model(model, test_gen, label):
    y_true, y_pred = [], []
    for i in range(len(test_gen)):
        X_batch, y_batch = test_gen[i]
        y_true.extend(y_batch)
        y_pred.extend((model.predict(X_batch) > 0.5).astype(int).flatten())

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, target_names=['Normal', 'Stego'])

    metrics = {
        "Model": label,
        "Accuracy": report['accuracy'],
        "Precision": report['Stego']['precision'],
        "Recall": report['Stego']['recall'],
        "F1-Score": report['Stego']['f1-score'],
        "False Positives": cm[0][1],
        "False Negatives": cm[1][0]
    }
    return metrics, cm

# ------------------------------
# Run Ablation for MFCC and MDCT
# ------------------------------
results = []
feature_sets = {
    "MFCC_256": {
        "normal": "/kaggle/input/mfcc-mdct-windows-features/normal_features_MFCC_win256.h5",
        "stego": "/kaggle/input/mfcc-mdct-windows-features/stego_features_MFCC_win256.h5"
    },
    "MDCT_256": {
        "normal": "/kaggle/input/mfcc-mdct-windows-features/normal_features_MDCT_win256.h5",
        "stego": "/kaggle/input/mfcc-mdct-windows-features/stego_features_MDCT_win256.h5"
    },
     "MFCC_512": {
        "normal": "/kaggle/input/mfcc-mdct-windows-features/normal_features_MFCC_win512.h5",
        "stego": "/kaggle/input/mfcc-mdct-windows-features/stego_features_MFCC_win512.h5"
    },
    "MDCT_512": {
        "normal": "/kaggle/input/mfcc-mdct-windows-features/normal_features_MDCT_win512.h5",
        "stego": "/kaggle/input/mfcc-mdct-windows-features/stego_features_MDCT_win512.h5"
    },
     "MFCC_1024": {
        "normal": "/kaggle/input/mfcc-mdct-windows-features/normal_features_MFCC_win1024.h5",
        "stego": "/kaggle/input/mfcc-mdct-windows-features/stego_features_MFCC_win1024.h5"
    },
    "MDCT_1024": {
        "normal": "/kaggle/input/mfcc-mdct-windows-features/normal_features_MDCT_win1024.h5",
        "stego": "/kaggle/input/mfcc-mdct-windows-features/stego_features_MDCT_win1024.h5"
    }
}

for label, paths in feature_sets.items():
    print(f"\nTraining on {label} features...")
    train_gen = H5DataGenerator(paths['normal'], paths['stego'], mode='train')
    test_gen = H5DataGenerator(paths['normal'], paths['stego'], mode='test', scaler=train_gen.scaler)

    model = build_model((train_gen.max_len, train_gen.n_features))

    model.fit(train_gen,
              validation_data=test_gen,
              epochs=20,
              callbacks=[
                  EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
              ],
              verbose=1)

    metrics, cm = evaluate_model(model, test_gen, label)
    results.append(metrics)

    # Confusion Matrix Plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Stego'], yticklabels=['Normal', 'Stego'])
    plt.title(f"{label} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# ------------------------------
# Summary Results Table and Plot
# ------------------------------
df = pd.DataFrame(results)
print("\nAblation Study Results:")
print(df)

# Bar plot for metrics
df.set_index("Model")[["Precision", "Recall", "F1-Score"]].plot(kind='bar', figsize=(8, 5))
plt.title("MFCC vs MDCT - Precision, Recall, F1-Score")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.show()
