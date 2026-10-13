"""Demonstrate EMG preprocessing using a synthetic signal."""

import matplotlib.pyplot as plt
import numpy as np

from src.emg_processing import (
    bandpass_filter,
    emg_features,
    moving_rms,
    notch_filter,
    rectify_emg,
)

# 1. Generate a synthetic EMG-like signal.
# This is simulated data, not a real patient recording.
fs = 2000
duration = 5.0
rng = np.random.default_rng(42)
time = np.arange(int(fs * duration)) / fs

# Simulate muscle activity in three activation periods.
activation = np.zeros_like(time)
activation[(time >= 0.5) & (time < 1.5)] = 1.0
activation[(time >= 2.0) & (time < 3.2)] = 0.7
activation[(time >= 3.7) & (time < 4.6)] = 1.2

muscle_signal = rng.normal(0, 1, len(time)) * activation
muscle_signal = muscle_signal + 0.15 * rng.normal(0, 1, len(time))

# Add simulated 50 Hz electrical interference.
interference = 0.25 * np.sin(2 * np.pi * 50 * time)
raw_emg = muscle_signal + interference

# 2. Filter the signal.
filtered = bandpass_filter(raw_emg, fs, lowcut=20, highcut=450)
filtered = notch_filter(filtered, fs, line_frequency=50)

# 3. Rectify and calculate the moving RMS envelope.
rectified = rectify_emg(filtered)
rms_envelope = moving_rms(filtered, fs, window_seconds=0.05)

# 4. Print features.
features = emg_features(filtered)

print("Synthetic EMG demonstration")
print(f"Sampling rate: {fs} Hz")
print(f"Duration: {duration:.1f} seconds")
print(f"Samples: {len(raw_emg)}")
print("\nProcessed-signal features:")
for name, value in features.items():
    print(f"{name}: {value:.5f}")

# 5. Visualize the processing stages.
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(time, raw_emg, linewidth=0.6)
axes[0].set_title("Raw synthetic EMG")
axes[0].set_ylabel("Amplitude")

axes[1].plot(time, filtered, linewidth=0.6)
axes[1].set_title("Band-pass and notch filtered EMG")
axes[1].set_ylabel("Amplitude")

axes[2].plot(time, rectified, linewidth=0.6)
axes[2].set_title("Full-wave rectified EMG")
axes[2].set_ylabel("Amplitude")

axes[3].plot(time, rms_envelope, linewidth=1.0)
axes[3].set_title("Moving RMS envelope (50 ms)")
axes[3].set_ylabel("RMS")
axes[3].set_xlabel("Time (seconds)")

for axis in axes:
    axis.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
