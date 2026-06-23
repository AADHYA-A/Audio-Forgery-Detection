import os
import sys
# Ensure the local utils package is found before any installed 'utils' package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
META_CSV = os.path.join(BASE_DIR, 'release_in_the_wild', 'meta.csv')
AUDIO_ROOT = os.path.join(BASE_DIR, 'release_in_the_wild')
MODEL_OUT = os.path.join(BASE_DIR, 'models', 'xgboost_model.pkl')
# ----------------------------------------------------------------------

def load_meta():
    df = pd.read_csv(META_CSV)
    df['filename'] = df['file']
    df['label'] = df['label'].map({'bona-fide': 0, 'spoof': 1})
    df = df.dropna(subset=['label'])
    df_bona = df[df['label'] == 0].sample(n=min(sum(df['label'] == 0), 500), random_state=42)
    df_spoof = df[df['label'] == 1].sample(n=min(sum(df['label'] == 1), 500), random_state=42)
    df = pd.concat([df_bona, df_spoof]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def extract_features(df):
    from utils import preprocess
    X = []
    y = []
    for _, row in df.iterrows():
        wav_path = os.path.join(AUDIO_ROOT, row['filename'])
        if not os.path.isfile(wav_path):
            continue
        gfcc = preprocess.extract_gfcc(wav_path)  # n_coeffs
        X.append(gfcc)
        y.append(row['label'])
    return np.stack(X), np.array(y)

def main():
    meta = load_meta()
    X, y = extract_features(meta)

    print('Train / test split')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print('Training XGBoost model')
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=4,
        random_state=42,
    )
    model.fit(X_train, y_train)

    print('Evaluating')
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    # Compute additional metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    # Save metrics as JSON
    import json, os
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }
    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    with open(os.path.join(BASE_DIR, 'models', 'metrics_xgboost.json'), 'w') as f:
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
    static_path = os.path.join(BASE_DIR, 'static', 'confusion_xgboost.png')
    os.makedirs(os.path.dirname(static_path), exist_ok=True)
    plt.savefig(static_path)
    plt.close()

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f'Model saved to {MODEL_OUT}')

if __name__ == '__main__':
    main()
