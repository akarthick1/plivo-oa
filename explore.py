import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
os.makedirs('outputs', exist_ok=True)

# Load data
df = pd.read_csv('data/english/labels.csv')

# Calculate pause durations purely for analysis
df['pause_dur'] = df['pause_end'] - df['pause_start']

# 1. Basic Counts
print("--- Dataset Statistics ---")
print(f"Total Pauses (Rows): {len(df)}")
print(f"Total Unique Turns: {df['turn_id'].nunique()}")

# 2. Class Balance
print("\n--- Label Distribution ---")
counts = df['label'].value_counts()
ratio = df['label'].value_counts(normalize=True) * 100
for idx in counts.index:
    print(f"{idx.upper()}: {counts[idx]} ({ratio[idx]:.1f}%)")

# 3. Pause Statistics by Class
print("\n--- Pause Duration Stats (Seconds) ---")
print(df.groupby('label')['pause_dur'].describe()[['mean', 'std', 'min', 'max']])

# 4. Visualization
plt.figure(figsize=(12, 5))

# Plot A: Class Balance
plt.subplot(1, 2, 1)
counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
plt.title('Hold vs EOT Count')
plt.ylabel('Number of Pauses')
plt.xticks(rotation=0)

# Plot B: Duration Distribution
plt.subplot(1, 2, 2)
df[df['label'] == 'hold']['pause_dur'].plot(kind='hist', alpha=0.6, label='Hold', bins=40, density=True)
df[df['label'] == 'eot']['pause_dur'].plot(kind='hist', alpha=0.6, label='EOT', bins=40, density=True)
plt.title('Pause Duration Density')
plt.xlabel('Seconds')
plt.legend()

plt.tight_layout()
plt.savefig('outputs/exploration_plots.png')
print("\nPlots saved to outputs/exploration_plots.png")