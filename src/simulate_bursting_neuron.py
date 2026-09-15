import numpy as np
from simulate_neuron import generate_spike_times, make_spike_waveform
import pandas as pd


def generate_bursting_spike_times(duration, background_rate, n_bursts, spikes_per_burst, burst_isi_ms=5):


    background_times = generate_spike_times(duration, background_rate)

    burst_onset_times = np.sort(np.random.uniform(0, duration - 0.5, size=n_bursts))

    burst_spike_times = []
    for onset in burst_onset_times:
        for i in range(spikes_per_burst):
            spike_t = onset + i * (burst_isi_ms / 1000)
            burst_spike_times.append(spike_t)

    burst_spike_times = np.array(burst_spike_times)

    all_times = np.sort(np.concatenate([background_times, burst_spike_times]))

    return all_times, burst_onset_times


def simulate_bursting_neuron(duration, sampling_rate, background_rate, n_bursts,
                               spikes_per_burst, burst_isi_ms=5, noise_level=2, seed=None):

    if seed is not None:
        np.random.seed(seed)

    n_samples = int(duration * sampling_rate)
    time = np.arange(n_samples) / sampling_rate
    voltage = np.full(n_samples, -65.0)

    spike_times, burst_onsets = generate_bursting_spike_times(
        duration, background_rate, n_bursts, spikes_per_burst, burst_isi_ms
    )

    spike_waveform = make_spike_waveform(sampling_rate)
    spike_len = len(spike_waveform)

    for spike_t in spike_times:
        start_idx = int(spike_t * sampling_rate)
        end_idx = start_idx + spike_len
        if end_idx < n_samples:
            voltage[start_idx:end_idx] = spike_waveform

    voltage += np.random.normal(0, noise_level, size=n_samples)

    df = pd.DataFrame({"time": time, "voltage": voltage})
    return df, spike_times, burst_onsets

#test

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from filter_signal import filter_signal
    from detect_spikes import compute_threshold, detect_spikes
    from burst_detection import detect_bursts

    sampling_rate = 10000
    duration = 10

    df, true_spike_times, true_burst_onsets = simulate_bursting_neuron(
        duration=duration,
        sampling_rate=sampling_rate,
        background_rate=5,
        n_bursts=5,
        spikes_per_burst=4,
        burst_isi_ms=5,
        noise_level=3,
        seed=42
    )

    print(f"True background+burst spikes: {len(true_spike_times)}")
    print(f"True burst onset times: {true_burst_onsets}")

    filtered = filter_signal(df["voltage"].values, sampling_rate, low_cutoff=300, high_cutoff=3000)
    threshold = compute_threshold(filtered, n_std=5)
    detected = detect_spikes(filtered, df["time"].values, sampling_rate, threshold=threshold)
    detected_spike_times = np.array([s["time"] for s in detected])

    bursts, isi_threshold = detect_bursts(detected_spike_times, isi_threshold=0.0181, min_spikes_per_burst=3)

    print(f"\nDetected {len(bursts)} bursts (using isi_threshold=18.1ms from tuning):")
    for b in bursts:
        print(b)