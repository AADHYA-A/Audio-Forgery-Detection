# Audio Forgery Detection

![Project Banner](https://raw.githubusercontent.com/yourusername/Audio_Forgery_Detection/main/assets/banner.png)

## Overview

This repository provides a **state‑of‑the‑art audio forgery detection** pipeline. It includes data preprocessing, feature extraction, model training, and evaluation scripts that can identify manipulated or synthetic audio segments.

The system is built with:
- **Python 3.11**
- **PyTorch** for deep‑learning models
- **Librosa** for audio signal processing
- **scikit‑learn** for classic ML baselines

> **Why this project?**
> Detecting forged audio is becoming increasingly important for security, media forensics, and deep‑fake mitigation. This codebase offers a reproducible baseline that can be extended with newer architectures.

---

## 📦 Dataset

We use the **In‑The‑Wild** audio forgery dataset from Hugging Face:

- 👉 **Download:** [https://huggingface.co/datasets/mueller91/In-The-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild)
- License: Creative Commons Attribution 4.0 (CC‑BY‑4.0)
- Contains real and forged audio clips across various environments and recording devices.

> **Note:** The dataset is **not** included in this repository. Please download it manually using the link above or via the `datasets` library:
>
> ```bash
> pip install huggingface-hub datasets
> python -c "from datasets import load_dataset; load_dataset('mueller91/In-The-Wild')"
> ```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Audio_Forgery_Detection.git
cd Audio_Forgery_Detection

# Create a virtual environment (optional but recommended)
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/macOS
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- numpy
- scipy
- librosa
- torch
- torchaudio
- scikit-learn
- pandas
- tqdm
- huggingface-datasets

---

## ▶️ Usage

1. **Prepare the data**
   ```bash
   python scripts/prepare_data.py --dataset-path /path/to/In-The-Wild
   ```
2. **Train a model**
   ```bash
   python scripts/train.py --config configs/default.yaml
   ```
3. **Evaluate**
   ```bash
   python scripts/evaluate.py --model checkpoints/best.pt --test-dir /path/to/test
   ```

All scripts provide `--help` for detailed arguments.

---

## 📈 Results

| Feature | CNN-LSTM | GFCC-XGBoost |
|--------|----------|--------------|
| Model Type | Deep Learning | Traditional Machine Learning |
| Input Features | Mel-Spectrogram | GFCC Features |
| Training Time | Higher | Lower |
| Computational Cost | High | Moderate |
| Explainability | Low | High |
| Accuracy | 60.0% | 93.5% |
| Recall | 39.0% | 92.0% |
| F1-Score | 49.4% | 93.4% |
| MCC | 0.21 | 0.87 |

Feel free to contribute new models and report benchmark improvements!

---
