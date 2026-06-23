import os
import sys
# Ensure the project root is in sys.path for local imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
META_CSV = os.path.join(BASE_DIR, 'release_in_the_wild', 'meta.csv')
AUDIO_ROOT = os.path.join(BASE_DIR, 'release_in_the_wild')
MODEL_OUT = os.path.join(BASE_DIR, 'models', 'cnn_lstm.keras')
IMG_HEIGHT = 128          # number of mel bins
MAX_FRAMES = 300          # time‑axis length used during training
# ----------------------------------------------------------------------

def load_meta():
    """Load meta.csv – must contain 'file' and 'label' columns."""
    df = pd.read_csv(META_CSV)
    df['filename'] = df['file']
    df['label'] = df['label'].map({'bona-fide': 0, 'spoof': 1})
    df = df.dropna(subset=['label'])
    df_bona = df[df['label'] == 0].sample(n=min(sum(df['label'] == 0), 500), random_state=42)
    df_spoof = df[df['label'] == 1].sample(n=min(sum(df['label'] == 1), 500), random_state=42)
    df = pd.concat([df_bona, df_spoof]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def extract_features(df):
    """Extract mel‑spectrograms and labels for the whole dataset.

    Returns
    -------
    X : np.ndarray, shape (N, IMG_HEIGHT, MAX_FRAMES, 1)
    y : np.ndarray, shape (N,)
    """
    from utils import preprocess
    specs = []
    labels = []
    for _, row in df.iterrows():
        wav_path = os.path.join(AUDIO_ROOT, row['filename'])
        if not os.path.isfile(wav_path):
            print(f'⚠️  Missing audio file: {wav_path}')
            continue
        spec = preprocess.extract_spectrogram(wav_path)
        # Pad / truncate to MAX_FRAMES
        if spec.shape[1] < MAX_FRAMES:
            pad_width = MAX_FRAMES - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            spec = spec[:, :MAX_FRAMES]
        # Add channel dimension
        spec = spec[np.newaxis, ..., np.newaxis]   # (1, H, W, 1)
        specs.append(spec)
        labels.append(row['label'])
    X = np.concatenate(specs, axis=0)  # (N, H, W, 1)
    y = np.array(labels)
    return X, y

def build_model(input_shape):
    """Create an enhanced CNN‑LSTM architecture.

    input_shape = (IMG_HEIGHT, MAX_FRAMES, 1)
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Convolutional block – deeper with dropout and L2 regularisation
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Reshape for LSTM (collapse frequency, keep time)
    x = tf.keras.layers.Permute((2, 1, 3))(x)  # (batch, time, freq, channel)
    # After three (2x2) poolings, dimensions are reduced by factor 8
    time_steps = MAX_FRAMES // 8  # 300 -> 37
    freq = IMG_HEIGHT // 8      # 128 -> 16
    channels = 128               # last Conv2D filters
    x = tf.keras.layers.Reshape((time_steps, freq * channels))(x)

    # LSTM block – bidirectional for richer temporal modeling
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Classification head
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print('Loading meta...')
    meta = load_meta()
    print('Extracting spectrogram features...')
    X, y = extract_features(meta)

    print('Train / test split...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print('Building model...')
    model = build_model(input_shape=X_train.shape[1:])

    # Train the model if no saved checkpoint exists
    if not os.path.exists(MODEL_OUT):
        print('Starting training...')
        # Compute class weights to handle imbalance
        from sklearn.utils import class_weight
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
            tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor='val_accuracy')
        ]
        model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=30,
            batch_size=32,
            class_weight=class_weights_dict,
            callbacks=callbacks,
            verbose=2
        )
    else:
        print('Loading saved model...')
        model = tf.keras.models.load_model(MODEL_OUT)

    print('Evaluating on test set...')
    preds = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    print(classification_report(y_test, preds))

    # Compute additional metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    # Save metrics as JSON
    import json
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    with open(os.path.join(BASE_DIR, 'models', 'metrics_cnn_lstm.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    # Plot confusion matrix
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['genuine', 'fake']
    tick_marks = [0,1]
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # Save figure to static folder
    static_path = os.path.join(BASE_DIR, 'static', 'confusion_cnn_lstm.png')
    os.makedirs(os.path.dirname(static_path), exist_ok=True)
    plt.savefig(static_path)
    plt.close()

    print(f'Model saved -> {MODEL_OUT}')

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # hide info logs
    main()
