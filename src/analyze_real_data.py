import numpy as np
from scipy.io import loadmat
from filter_signal import filter_signal
from detect_spikes import compute_threshold, detect_spikes

mat = loadmat("data/C_Easy2_noise015.mat")

sampling_interval_ms = mat["samplingInterval"][0, 0]
sampling_rate = round(1000 / sampling_interval_ms)  # convert ms -> Hz
print(f"Sampling rate: {sampling_rate} Hz")

voltage = mat["data"][0].astype(float)
n_samples = len(voltage)
time = np.arange(n_samples) / sampling_rate
duration = time[-1]
print(f"Duration: {duration:.2f} s, {n_samples} samples")


true_spike_indices = mat["spike_times"][0, 0].flatten()
true_spike_times = true_spike_indices / sampling_rate
print(f"True spikes: {len(true_spike_times)}")


filtered = filter_signal(voltage, sampling_rate, low_cutoff=300, high_cutoff=3000)
threshold = compute_threshold(filtered, n_std=5)
detected = detect_spikes(filtered, time, sampling_rate, threshold=threshold)
detected_spike_times = np.array([s["time"] for s in detected])

print(f"\nAdaptive threshold: {threshold:.3f}")
print(f"Detected spikes: {len(detected_spike_times)}")
print(f"Ratio (detected/true): {100 * len(detected_spike_times) / len(true_spike_times):.1f}%")

def score_detections(true_times, detected_times, tolerance_ms=2, sampling_rate=24000):
    tolerance = tolerance_ms / 1000
    matched_true = np.zeros(len(true_times), dtype=bool)
    true_positives = 0
    false_positives = 0

    for dt in detected_times:
        diffs = np.abs(true_times - dt)
        nearest_idx = np.argmin(diffs)
        if diffs[nearest_idx] <= tolerance and not matched_true[nearest_idx]:
            true_positives += 1
            matched_true[nearest_idx] = True
        else:
            false_positives += 1

    false_negatives = len(true_times) - true_positives
    return true_positives, false_positives, false_negatives

tp, fp, fn = score_detections(true_spike_times, detected_spike_times, tolerance_ms=2, sampling_rate=sampling_rate)
print(f"\nTrue positives: {tp}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"Precision: {100 * tp / (tp + fp):.1f}%")
print(f"Recall: {100 * tp / (tp + fn):.1f}%")

