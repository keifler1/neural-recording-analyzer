
import numpy as np
from scipy.signal import butter, filtfilt

def filter_signal(voltage, sampling_rate, low_cutoff=None, high_cutoff=None, order=4):

    #apply butterworth filter to the voltage signal

 #   - If only low_cutoff: high-pass filter (removes slow drift)
   # - If only high_cutoff: low-pass filter (removes high-freq. noise)
  #  - If both: band-pass filter

    nyquist = sampling_rate / 2  # Nyquist is highest measurable frequency captured by your sampling rate

    if low_cutoff and high_cutoff:
        b, a = butter(order, [low_cutoff / nyquist, high_cutoff / nyquist], btype="band")
    elif low_cutoff:
        b, a = butter(order, low_cutoff / nyquist, btype="high")
    elif high_cutoff:
        b, a = butter(order, high_cutoff / nyquist, btype="low")
    else:
        raise ValueError("specify at least one of low_cutoff or high_cutoff")

    filtered = filtfilt(b, a, voltage) #filter signal forward then backward to cancel phase distortion, detecting exact spike time
    return filtered

#test and compare raw to filtered

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from simulate_neuron import simulate_neuron

    df, spike_times = simulate_neuron(
        duration=2,
        sampling_rate=10000,
        firing_rate=5,
        noise_level=5,  #you can increase to show effect of filtering
        seed=42
    )

    filtered = filter_signal(
        df["voltage"].values,
        sampling_rate=10000,
        low_cutoff=300,
        high_cutoff=3000
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    axes[0].plot(df["time"], df["voltage"], linewidth=0.5)
    axes[0].set_title("Raw")
    axes[0].set_ylabel("Voltage (mV)")

    axes[1].plot(df["time"], filtered, linewidth=0.5, color="green")
    axes[1].set_title("Filtered (band-pass 300-3000 Hz)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Voltage (mV)")

    plt.tight_layout()
    plt.show()