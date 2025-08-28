import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

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
# Model with Attention for Feature Analysis
# ------------------------------
def build_model_with_attention(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial convolution & pooling
    x = tf.keras.layers.Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(3, strides=2, padding='same')(x)

    # Residual SE blocks
    x = residual_se_block(x, 64)
    x = residual_se_block(x, 64)
    x = residual_se_block(x, 128, stride=2)
    x = residual_se_block(x, 128)

    # --- Attention mechanism ---
    attention = tf.keras.layers.Dense(1, activation='tanh', name='attention_dense')(x)
    attention = tf.keras.layers.Flatten(name='attention_flatten')(attention)
    attention = tf.keras.layers.Activation('softmax', name='attention_weights')(attention)
    attention_expanded = tf.keras.layers.RepeatVector(x.shape[-1])(attention)
    attention_expanded = tf.keras.layers.Permute([2, 1])(attention_expanded)
    
    # Apply attention
    x = tf.keras.layers.Multiply(name='attention_multiply')([x, attention_expanded])
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Fully connected layers
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_attention_extractor(attention_model):
    """Create a model to extract attention weights"""
    return tf.keras.Model(
        inputs=attention_model.input,
        outputs=attention_model.get_layer('attention_weights').output
    )

def train_attention_model(model, train_gen, test_gen, epochs=15):
    """Custom training function for attention model"""
    def adapt_generator(gen):
        for X, y in gen:
            yield X, {'output': y, 'attention_weights': np.zeros(y.shape[0])}
    
    history = model.fit(adapt_generator(train_gen),
                      validation_data=adapt_generator(test_gen),
                      epochs=epochs,
                      steps_per_epoch=len(train_gen),
                      validation_steps=len(test_gen),
                      callbacks=[
                          EarlyStopping(monitor='val_output_loss', patience=3, restore_best_weights=True)
                      ],
                      verbose=1)
    return history

# ------------------------------
# Evaluation and Feature Importance
# ------------------------------
def evaluate_model(model, test_gen, label):
    y_true, y_pred = [], []
    for i in range(len(test_gen)):
        X_batch, y_batch = test_gen[i]
        if isinstance(model.output, list):  # For attention model
            preds, _ = model.predict(X_batch, verbose=0)
        else:
            preds = model.predict(X_batch, verbose=0)
        y_true.extend(y_batch)
        y_pred.extend((preds > 0.5).astype(int).flatten())

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

def analyze_feature_importance(model, test_gen, n_features, n_runs=3):
    """Analyze importance of each bin by masking individual features"""
    # Get baseline accuracy
    y_true, y_pred = [], []
    for i in range(len(test_gen)):
        X_batch, y_batch = test_gen[i]
        preds = model.predict(X_batch, verbose=0)
        y_true.extend(y_batch)
        y_pred.extend((preds > 0.5).astype(int).flatten())
    baseline_acc = accuracy_score(y_true, y_pred)
    
    # Test masking each feature
    feature_importance = np.zeros(n_features)

   
    for feature_idx in tqdm(range(n_features), desc="Analyzing features"):
        total_drop = 0
        
        for _ in range(n_runs):
            y_true_masked, y_pred_masked = [], []
            for i in range(len(test_gen)):
                X_batch, y_batch = test_gen[i]
                X_masked = X_batch.copy()
                X_masked[:, :, feature_idx] = 0  # Mask this feature
                preds = model.predict(X_masked, verbose=0)
                y_true_masked.extend(y_batch)
                y_pred_masked.extend((preds > 0.5).astype(int).flatten())
            
            masked_acc = accuracy_score(y_true_masked, y_pred_masked)
            total_drop += baseline_acc - masked_acc
        
        feature_importance[feature_idx] = total_drop / n_runs
    
    return feature_importance

def compute_gradient_importance(model, X_sample):
    """Compute gradient-based importance scores"""
    input_tensor = model.input
    output_tensor = model.output[0] if isinstance(model.output, list) else model.output
    gradients = tf.keras.backend.gradients(output_tensor, input_tensor)[0]
    get_gradients = tf.keras.backend.function([input_tensor], [gradients])
    
    grad_values = get_gradients([X_sample])[0]
    return np.mean(np.abs(grad_values), axis=(0, 1))

def plot_feature_importance(importance_scores, feature_name="MFCC", method="Masking"):
    """Plot importance scores for each feature bin"""
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importance_scores)), importance_scores)
    plt.xlabel(f"{feature_name} Bin Index")
    plt.ylabel("Importance Score")
    plt.title(f"Feature Importance by {method} ({feature_name})")
    plt.grid(True)
    plt.show()
    
    # Print top bins
    top_bins = np.argsort(importance_scores)[-5:][::-1]
    print(f"\nTop 5 most important {feature_name} bins ({method}):")
    for i, bin_idx in enumerate(top_bins):
        print(f"{i+1}. Bin {bin_idx}: {importance_scores[bin_idx]:.4f}")

def plot_cross_window_comparison(importance_results, feature_type='MFCC'):
    """Compare importance patterns across window sizes"""
    window_sizes = ['256', '512', '1024']
    methods = ['masking', 'attention', 'gradient']
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    
    for i, method in enumerate(methods):
        ax = axes[i]
        for ws in window_sizes:
            key = f"{feature_type}_{ws}"
            if key in importance_results:
                scores = importance_results[key][method]
                ax.plot(scores, label=f'{ws} samples')
        
        ax.set_title(f'{method.capitalize()} Importance - {feature_type}')
        ax.set_xlabel('Bin Index')
        ax.set_ylabel('Importance Score')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# ------------------------------
# Main Analysis Pipeline
# ------------------------------
# ------------------------------
# Main Analysis Pipeline
# ------------------------------
def run_full_analysis(feature_sets):
    results = []
    importance_results = {}
    
    for label, paths in feature_sets.items():
        print(f"\n{'='*50}")
        print(f"Analyzing {label} features...")
        print(f"{'='*50}")
        
        # Prepare data generators
        train_gen = H5DataGenerator(paths['normal'], paths['stego'], mode='train')
        test_gen = H5DataGenerator(paths['normal'], paths['stego'], mode='test', scaler=train_gen.scaler)
        
        # Standard model training and evaluation
        model = build_model((train_gen.max_len, train_gen.n_features))
        print("\nTraining standard model...")
        model.fit(train_gen,
                validation_data=test_gen,
                epochs=20,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
                ],
                verbose=1)
        
        # Evaluate standard model
        metrics, cm = evaluate_model(model, test_gen, label)
        results.append(metrics)
        
        # Confusion Matrix Plot
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Stego'], yticklabels=['Normal', 'Stego'])
        plt.title(f"{label} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()
        
        # Only perform feature importance analysis for MFCC features
        if label.startswith("MFCC"):
            print(f"\nPerforming detailed feature analysis for {label}...")
            current_results = {}
            
            # 1. Feature importance by masking
            print("\n1. Computing feature importance by masking...")
            importance_scores = analyze_feature_importance(model, test_gen, train_gen.n_features)
            plot_feature_importance(importance_scores, label, "Masking")
            current_results['masking'] = importance_scores
            importance_results[label] = current_results
    
    # Summary Results
    df = pd.DataFrame(results)
    print("\nPerformance Results:")
    print(df)
    
    # Comparative plots
    plt.figure(figsize=(12, 6))
    df.set_index('Model')[['Accuracy', 'F1-Score']].plot(kind='bar')
    plt.title("Model Performance Across Feature Sets")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot comparison of MFCC feature importance across window sizes
    if any(k.startswith('MFCC') for k in importance_results):
        plot_cross_window_comparison(importance_results, feature_type='MFCC')
    
    return df, importance_results

# ------------------------------
# Run the Analysis
# ------------------------------
if __name__ == "__main__":
     feature_sets = {
        "MFCC_256": {
            "normal": "/input/mfcc-mdct-windows-features/normal_features_MFCC_win256.h5",
            "stego": "/input/mfcc-mdct-windows-features/stego_features_MFCC_win256.h5"
        },
        "MFCC_256_96bitrate": {
            "normal": "/input/mfcc-mdct-windows-features/normal_features_MFCC_96bit256.h5",
            "stego": "/input/96-bit-stego-features/stego_features_MFCC_96bit256.h5"
        },
        "MFCC_512": {
            "normal": "/input/mfcc-mdct-windows-features/normal_features_MFCC_win512.h5",
            "stego": "/input/mfcc-mdct-windows-features/stego_features_MFCC_win512.h5"
        },
        "MFCC_512_96bitrate": {
            "normal": "/input/mfcc-mdct-windows-features/normal_features_MFCC_96bit512.h5",
            "stego": "/input/96-bit-stego-features/stego_features_MFCC_96bit512.h5"
        },
        "MFCC_1024": {
            "normal": "/input/mfcc-mdct-windows-features/normal_features_MFCC_win1024.h5",
            "stego": "/input/mfcc-mdct-windows-features/stego_features_MFCC_win1024.h5"
        },
        "MFCC_1024_96bitrate": {
            "normal": "/input/mfcc-mdct-windows-features/normal_features_MFCC_96bit1024.h5",
            "stego": "/input/96-bit-stego-features/stego_features_MFCC_96bit1024.h5"
        }
    }
    
performance_df, importance_results = run_full_analysis(feature_sets)