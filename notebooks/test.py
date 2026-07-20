from utils import load_labels, load_audio
import os

labels_path = 'data/english/labels.csv'
df = load_labels(labels_path)
print("--- Labels Head ---")
print(df.head())

first_turn = df.iloc[0]['turn_id']
wav_path = f"data/english/audio/{first_turn}.wav" 

if os.path.exists(wav_path):
    audio, sr = load_audio(wav_path)
    print("\n--- Audio Info ---")
    print(f"File: {wav_path}")
    print(f"Sample Rate: {sr} Hz")
    print(f"Audio array shape: {audio.shape}")
    print(f"Duration: {len(audio)/sr:.2f} seconds")
    print(f"First 5 samples: {audio[:5]}")
else:
    print(f"File not found: {wav_path}")