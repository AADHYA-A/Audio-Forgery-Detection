import numpy as np
import librosa


def extract_gfcc(audio_path, n_coeffs=13):
    """Extract GFCC-like features using librosa.

    Since a true GFCC implementation requires a gammatone filterbank which is not
    available in the default libs, we approximate it by computing MFCCs on the
    mel‑scaled spectrogram. The returned feature is the mean across time frames
    producing a fixed‑length 1‑D vector.
    """
    y, sr = librosa.load(audio_path, sr=None)
    # Compute MFCCs (as a proxy for GFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_coeffs)
    # Average over time axis
    return np.mean(mfcc, axis=1)


def extract_spectrogram(audio_path, n_mels=128, fmax=None):
    """Return a mel‑spectrogram suitable for CNN/LSTM input.

    The spectrogram is returned as a 2‑D numpy array (frequency x time) with
    values in decibels.
    """
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    # Convert to log scale (dB)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db
