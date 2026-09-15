import numpy as np
import matplotlib.pyplot as plt
from simulate_neuron import simulate_neuron
from filter_signal import filter_signal

# generate a recording
df, spike_times = simulate_neuron(
    duration=2,
    sampling_rate=10000,
    firing_rate=5,
    noise_level=3,
    seed=42
)

sampling_rate = 10000
filtered = filter_signal(df["voltage"].values, sampling_rate, low_cutoff=300, high_cutoff=3000)

# get a window around spike 1
spike_t = spike_times[0]
window = 0.01  # 10 ms
mask = (df["time"] >= spike_t - window) & (df["time"] <= spike_t + window)

raw_window = df["voltage"].values[mask]
filtered_window = filtered[mask]
time_window = df["time"].values[mask]

# raw vs filtered amplitude
raw_amplitude = raw_window.max() - raw_window.min()
filtered_amplitude = filtered_window.max() - filtered_window.min()

print(f"Raw spike amplitude: {raw_amplitude:.2f} mV")
print(f"Filtered spike amplitude: {filtered_amplitude:.2f} mV")
print(f"Difference: {raw_amplitude - filtered_amplitude:.2f} mV "
      f"({100 * (raw_amplitude - filtered_amplitude) / raw_amplitude:.1f}% change)")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(time_window, raw_window, label="Raw", alpha=0.7)
plt.plot(time_window, filtered_window, label="Filtered", alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Single spike: raw vs filtered")
plt.legend()
plt.tight_layout()
plt.show()