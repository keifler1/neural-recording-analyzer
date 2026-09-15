import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def characterize_spike(voltage, time, peak_idx, sampling_rate, window_ms=5, baseline_ms=1):
    window_samples = int((window_ms / 1000) * sampling_rate)
    baseline_samples = int((baseline_ms / 1000) * sampling_rate)

    start = max(0, peak_idx - window_samples)
    end = min(len(voltage), peak_idx + window_samples)

    local_voltage = voltage[start:end]
    local_time = time[start:end]
    peak_voltage = voltage[peak_idx]
    peak_time = time[peak_idx]


    baseline_end = max(0, peak_idx - baseline_samples - start)
    baseline_segment = local_voltage[:max(baseline_end, 1)]
    baseline_voltage = np.mean(baseline_segment)

    amplitude = peak_voltage - baseline_voltage
    half_max = baseline_voltage + amplitude / 2


    local_peak_idx = peak_idx - start
    onset_idx = local_peak_idx
    while onset_idx > 0 and local_voltage[onset_idx] >= half_max:
        onset_idx -= 1
    time_to_peak = peak_time - local_time[onset_idx]


    offset_idx = local_peak_idx
    while offset_idx < len(local_voltage) - 1 and local_voltage[offset_idx] >= half_max:
        offset_idx += 1

    half_width = local_time[offset_idx] - local_time[onset_idx]

    return {
        "time": peak_time,
        "peak": peak_voltage,
        "baseline": baseline_voltage,
        "amplitude": amplitude,
        "time_to_peak": time_to_peak,
        "half_width": half_width,
    }

def characterize_all_spikes(voltage, time, detected_spikes, sampling_rate, window_ms=5):

    records = []
    for spike in detected_spikes:
        props = characterize_spike(voltage, time, spike["index"], sampling_rate, window_ms)
        records.append(props)

    return pd.DataFrame(records)

if __name__ == "__main__":
    from simulate_neuron import simulate_neuron
    from filter_signal import filter_signal
    from detect_spikes import compute_threshold, detect_spikes

    sampling_rate = 10000
    df, true_spike_times = simulate_neuron(
        duration=5, sampling_rate=sampling_rate, firing_rate=10,
        noise_level=3, seed=42
    )

    filtered = filter_signal(df["voltage"].values, sampling_rate, low_cutoff=300, high_cutoff=3000)
    threshold = compute_threshold(filtered, n_std=5)
    detected = detect_spikes(filtered, df["time"].values, sampling_rate, threshold=threshold)

    spike_table = characterize_all_spikes(filtered, df["time"].values, detected, sampling_rate)

    print(spike_table.head(10))
    print()
    print(spike_table.describe())