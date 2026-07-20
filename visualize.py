import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from features import load_wav, speech_before, frame_energy_db, f0_contour

os.makedirs('outputs', exist_ok=True)

# Load labels and grab two contrasting examples (one HOLD, one EOT)
df = pd.read_csv('data/english/labels.csv')
hold_row = df[df['label'] == 'hold'].iloc[0]
eot_row = df[df['label'] == 'eot'].iloc[0]

def plot_audio_features(row, ax_wave, ax_energy, ax_pitch):
    wav_path = os.path.join('data/english', row['audio_file'])
    x, sr = load_wav(wav_path)
    
    # Strictly isolate the 1.5 seconds BEFORE the pause
    pause_start = float(row['pause_start'])
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    
    # Time axis for waveform
    time_wave = np.linspace(-1.5, 0, len(seg))
    
    # Extract Features
    energy = frame_energy_db(seg, sr)
    pitch = f0_contour(seg, sr)
    
    # Time axes for frames
    time_frames = np.linspace(-1.5, 0, len(energy))
    time_pitch = np.linspace(-1.5, 0, len(pitch))
    
    # Plot Waveform
    ax_wave.plot(time_wave, seg, color='gray')
    ax_wave.set_title(f"Waveform - {row['label'].upper()}")
    ax_wave.set_ylabel("Amplitude")
    
    # Plot Energy
    ax_energy.plot(time_frames, energy, color='orange')
    ax_energy.set_title("Energy (dB)")
    ax_energy.set_ylabel("dB")
    
    # Plot Pitch (masking 0 values which mean unvoiced/silence)
    pitch_masked = np.where(pitch > 0, pitch, np.nan)
    ax_pitch.plot(time_pitch, pitch_masked, color='blue', marker='.', linestyle='-')
    ax_pitch.set_title("Pitch (Hz)")
    ax_pitch.set_ylabel("Hz")
    ax_pitch.set_xlabel("Time until pause (seconds)")

# Create figure
fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)

# Plot Hold on left, EOT on right
plot_audio_features(hold_row, axes[0, 0], axes[1, 0], axes[2, 0])
plot_audio_features(eot_row, axes[0, 1], axes[1, 1], axes[2, 1])

plt.tight_layout()
plt.savefig('outputs/audio_features.png')
print("Saved visualization to outputs/audio_features.png")
