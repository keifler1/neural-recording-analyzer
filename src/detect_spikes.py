import numpy as np

def compute_threshold(voltage, n_std=5):
    """
    estimate a spike threshold based on the signal's noise level,
    using median absolute deviation (MAD)
    """
    median = np.median(voltage)
    mad = np.median(np.abs(voltage - median))
    noise_std_estimate = mad / 0.6745  # converts MAD to an equivalent stdev
    threshold = median + n_std * noise_std_estimate
    return threshold


def detect_spikes(voltage, time, sampling_rate, threshold=-20, refractory_ms=2):

    #spike is detected by crossing a threshold
    # refractory_ms: minimum time between detected spikes, so one spike is not counted multiple times

    above = voltage > threshold

    # find where signal crosses from below to above threshold
    #i.e. diff of +1 indicates crossing upward
    crossings = np.where(np.diff(above.astype(int)) == 1)[0]

    refractory_samples = int((refractory_ms / 1000) * sampling_rate)

    spikes = []
    last_spike_idx = -refractory_samples  # allow first spike immediately

    for idx in crossings:
        if idx - last_spike_idx < refractory_samples:
            continue  # skip if too close to previous spike

        # search forward in a short window to find peak
        search_end = min(idx + refractory_samples, len(voltage))
        peak_offset = np.argmax(voltage[idx:search_end])
        peak_idx = idx + peak_offset

        spikes.append({
            "time": time[peak_idx],
            "peak": voltage[peak_idx],
            "index": peak_idx
        })

        last_spike_idx = peak_idx

    return spikes


#Let's test it with simulate_neuron

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from simulate_neuron import simulate_neuron
    from filter_signal import filter_signal

    sampling_rate = 10000
    df, true_spike_times = simulate_neuron(
        duration=5,
        sampling_rate=sampling_rate,
        firing_rate=10,
        noise_level=3,
        seed=42
    )

    filtered = filter_signal(df["voltage"].values, sampling_rate, low_cutoff=300, high_cutoff=3000)

    threshold = compute_threshold(filtered, n_std=5)
    detected = detect_spikes(filtered, df["time"].values, sampling_rate, threshold=threshold)

    print(f"Adaptive threshold: {threshold:.1f} mV")
    print(f"True spikes: {len(true_spike_times)}")
    print(f"Detected spikes: {len(detected)}")





    plt.figure(figsize=(12, 4))
    plt.plot(df["time"], filtered, linewidth=0.5, label="Filtered signal")
    plt.scatter([s["time"] for s in detected], [s["peak"] for s in detected],
                color="red", marker="x", label="Detected spikes", zorder=5)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.title(f"Detected {len(detected)} / {len(true_spike_times)} true spikes")
    plt.tight_layout()
    plt.show()

    #test with increasing noise and return a table showing accuracy by noise level
    noise_levels = [1, 3, 5, 10, 15, 20]

    print()
    print(f"{'Noise level':<12}{'True':<8}{'Detected':<10}{'Accuracy':<10}{'Threshold':<10}")
    for noise in noise_levels:
        df, true_spike_times = simulate_neuron(
            duration=5,
            sampling_rate=sampling_rate,
            firing_rate=10,
            noise_level=noise,
            seed=42
        )
        filtered = filter_signal(df["voltage"].values, sampling_rate, low_cutoff=300, high_cutoff=3000)

        threshold = compute_threshold(filtered, n_std=5)
        detected = detect_spikes(filtered, df["time"].values, sampling_rate, threshold=threshold)

        accuracy = 100 * len(detected) / len(true_spike_times)
        print(f"{noise:<12}{len(true_spike_times):<8}{len(detected):<10}{accuracy:<10.1f}{threshold:<10.1f}")


def score_detections(true_times, detected_times, tolerance_ms=2, sampling_rate=10000):
    """
    Compare detected spike times to ground-truth spike times.

    A detection counts as a match (true positive) if it falls within
    tolerance_ms of an unmatched true spike. Anything else is a false
    positive; any unmatched true spike is a false negative.
    """
    true_times = np.asarray(true_times)
    detected_times = np.asarray(detected_times)
    tolerance = tolerance_ms / 1000

    matched_true = np.zeros(len(true_times), dtype=bool)
    true_positives = 0
    false_positives = 0

    for dt in detected_times:
        if len(true_times) == 0:
            false_positives += 1
            continue

        diffs = np.abs(true_times - dt)
        nearest_idx = np.argmin(diffs)

        if diffs[nearest_idx] <= tolerance and not matched_true[nearest_idx]:
            true_positives += 1
            matched_true[nearest_idx] = True
        else:
            false_positives += 1

    false_negatives = len(true_times) - true_positives

    return true_positives, false_positives, false_negatives