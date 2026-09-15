
import numpy as np

from simulate_neuron import simulate_neuron
from filter_signal import filter_signal
from detect_spikes import compute_threshold, detect_spikes
from characterize_spikes import characterize_all_spikes
from firing_analysis import compute_firing_rate, compute_isis, isi_statistics
from burst_detection import detect_bursts


def run_pipeline(duration=10, sampling_rate=10000, firing_rate=10, noise_level=3, seed=42):
    df, true_spike_times = simulate_neuron(
        duration=duration, sampling_rate=sampling_rate,
        firing_rate=firing_rate, noise_level=noise_level, seed=seed
    )
    voltage = df["voltage"].values
    time = df["time"].values

    print("=" * 50)
    print("NEURAL RECORDING ANALYSIS: Simulated neuron")
    print("=" * 50)
    print(f"Duration:          {duration:.1f} s")
    print(f"Sampling rate:     {sampling_rate:,} Hz")

    # Filter
    filtered = filter_signal(voltage, sampling_rate, low_cutoff=300, high_cutoff=3000)

    # Detect
    threshold = compute_threshold(filtered, n_std=5)
    detected = detect_spikes(filtered, time, sampling_rate, threshold=threshold)
    spike_times = np.array([s["time"] for s in detected])
    print(f"\nSpikes detected:   {len(spike_times)}")



    # Characterize
    spike_table = characterize_all_spikes(filtered, time, detected, sampling_rate)
    print(f"\nMean amplitude:    {spike_table['amplitude'].mean():.2f}")
    print(f"Mean half-width:   {spike_table['half_width'].mean()*1000:.2f} ms")

    # Firing behavior
    rate = compute_firing_rate(spike_times, duration)
    isis = compute_isis(spike_times)
    isi_stats = isi_statistics(isis)
    print(f"\nFiring rate:       {rate:.2f} Hz")
    print(f"Mean ISI:          {isi_stats['mean_isi']*1000:.1f} ms")
    print(f"CV:                {isi_stats['cv_isi']:.2f}")

    # Bursts
    bursts, isi_threshold = detect_bursts(spike_times, min_spikes_per_burst=3)
    print(f"\nBursts detected:   {len(bursts)}")
    print("=" * 50)

    return {
        "filtered": filtered,
        "spike_times": spike_times,
        "spike_table": spike_table,
        "firing_rate": rate,
        "isi_stats": isi_stats,
        "bursts": bursts,
    }


if __name__ == "__main__":
    run_pipeline()


