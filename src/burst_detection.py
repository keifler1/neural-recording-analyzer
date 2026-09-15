import numpy as np

def detect_bursts(spike_times, isi_threshold=None, min_spikes_per_burst=3):
    """
    detects bursts as runs of consecutive spikes with ISI below isi_threshold

    If isi_threshold is None, default is 1/2 the mean ISI of recording
    min_spikes_per_burst: minimum number of spikes required to classify a burst

    """
    spike_times = np.sort(spike_times)
    isis = np.diff(spike_times)

    if isi_threshold is None:
        isi_threshold = np.mean(isis) / 5


    is_short_isi = isis < isi_threshold

    bursts = []
    current_burst = [0]  # start with first spike

    for i, short in enumerate(is_short_isi):
        if short:
            current_burst.append(i + 1)  # extend burst to include next spike
        else:
            if len(current_burst) >= min_spikes_per_burst:
                bursts.append(current_burst)
            current_burst = [i + 1]  # start a new potential burst


    if len(current_burst) >= min_spikes_per_burst:
        bursts.append(current_burst)


    burst_info = []
    for burst_indices in bursts:
        burst_spike_times = spike_times[burst_indices]
        burst_info.append({
            "start_time": burst_spike_times[0],
            "end_time": burst_spike_times[-1],
            "n_spikes": len(burst_indices),
            "duration": burst_spike_times[-1] - burst_spike_times[0],
        })

    return burst_info, isi_threshold

#test on simulate_neuron first: we should not get bursts

if __name__ == "__main__":
    from simulate_neuron import simulate_neuron
    from filter_signal import filter_signal
    from detect_spikes import compute_threshold, detect_spikes

    sampling_rate = 10000
    duration = 10
    df, true_spike_times = simulate_neuron(
        duration=duration, sampling_rate=sampling_rate, firing_rate=10,
        noise_level=3, seed=42
    )

    filtered = filter_signal(df["voltage"].values, sampling_rate, low_cutoff=300, high_cutoff=3000)
    threshold = compute_threshold(filtered, n_std=5)
    detected = detect_spikes(filtered, df["time"].values, sampling_rate, threshold=threshold)
    spike_times = np.array([s["time"] for s in detected])

    bursts, isi_threshold = detect_bursts(spike_times, min_spikes_per_burst=3)

    print(f"ISI threshold used: {isi_threshold*1000:.1f} ms")
    print(f"Number of bursts detected: {len(bursts)}")
    for b in bursts:
        print(b)