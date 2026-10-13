import matplotlib.pyplot as plt
import numpy as np

from src.biomedical_utils import normalize_signal

# 1. Generate a synthetic biomedical signal
rng = np.random.default_rng(42)
time = np.linspace(0, 2, 1000)

signal = (
    0.7 * np.sin(2 * np.pi * 8 * time)
    + 0.2 * np.sin(2 * np.pi * 35 * time)
    + 0.15 * rng.normal(size=time.size)
)

# 2. Normalize the signal
normalized_signal = normalize_signal(signal)

# 3. Plot raw and normalized signals
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(time, signal, linewidth=0.8)
axes[0].set_title("Raw Biomedical Signal")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, alpha=0.3)

axes[1].plot(time, normalized_signal, linewidth=0.8)
axes[1].set_title("Min-Max Normalized Signal")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Normalized amplitude")
axes[1].set_ylim(-0.05, 1.05)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
