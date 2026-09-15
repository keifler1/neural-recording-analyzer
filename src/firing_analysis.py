import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_firing_rate(spike_times, duration):
    #spikes/sec avg over recording
    return len(spike_times) / duration


def compute_isis(spike_times):
    #ISIs
    spike_times = np.sort(spike_times)
    return np.diff(spike_times)


def isi_statistics(isis):
    mean = np.mean(isis)
    median = np.median(isis)
    std = np.std(isis)
    cv = std / mean

    return {
        "mean_isi": mean,
        "median_isi": median,
        "std_isi": std,
        "cv_isi": cv,
    }


def plot_raster(spike_times, title="Raster plot"):
    plt.figure(figsize=(12, 2))
    plt.eventplot(spike_times, lineoffsets=0, linelengths=0.8, color="black")
    plt.xlabel("Time (s)")
    plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_isi_histogram(isis, title="ISI distribution"):
    plt.figure(figsize=(8, 4))
    plt.hist(isis * 1000, bins=30, color="steelblue", edgecolor="black")
    plt.xlabel("Inter-spike interval (ms)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_firing_rate_over_time(spike_times, duration, bin_size=0.5):
    bins = np.arange(0, duration + bin_size, bin_size)
    counts, edges = np.histogram(spike_times, bins=bins)
    rate = counts / bin_size

    bin_centers = (edges[:-1] + edges[1:]) / 2

    plt.figure(figsize=(12, 4))
    plt.plot(bin_centers, rate, marker="o")
    plt.xlabel("Time (s)")
    plt.ylabel("Firing rate (Hz)")
    plt.title("Firing rate over time")
    plt.tight_layout()
    plt.show()


#Let's test it with simulate_neuron

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

    rate = compute_firing_rate(spike_times, duration)
    isis = compute_isis(spike_times)
    stats = isi_statistics(isis)

    print(f"Firing rate: {rate:.2f} Hz")
    print(f"Mean ISI: {stats['mean_isi'] * 1000:.1f} ms")
    print(f"Median ISI: {stats['median_isi'] * 1000:.1f} ms")
    print(f"Std ISI: {stats['std_isi'] * 1000:.1f} ms")
    print(f"CV: {stats['cv_isi']:.2f}")

    plot_raster(spike_times)
    plot_isi_histogram(isis)
    plot_firing_rate_over_time(spike_times, duration)