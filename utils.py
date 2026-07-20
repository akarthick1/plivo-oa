import pandas as pd
import librosa

def load_labels(csv_path):
    """Loads labels.csv into a pandas DataFrame."""
    return pd.read_csv(csv_path)

def load_audio(wav_path, target_sr=16000):
    """Reads a wav file, converts to mono, resamples to 16kHz."""
    audio, sr = librosa.load(wav_path, sr=target_sr)
    return audio, sr