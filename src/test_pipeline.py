import matplotlib.pyplot as plt

from run_pipeline import run_pipeline


# Run several simulated datasets
tests = [
    ("Low firing", 2, 3, 1),
    ("Normal firing", 10, 3, 2),
    ("High firing", 50, 3, 3),
    ("Low noise", 10, 0.5, 4),
    ("High noise", 10, 10, 5),
    ("Very high noise", 10, 20, 6),
]


# Store results
names = []
firing_rates = []
mean_amplitudes = []
mean_widths = []
mean_isis = []
cv_isis = []
burst_counts = []


# Run each test
for name, firing_rate, noise_level, seed in tests:

    result = run_pipeline(
        firing_rate=firing_rate,
        noise_level=noise_level,
        seed=seed
    )

    names.append(name)

    firing_rates.append(result["firing_rate"])

    mean_amplitudes.append(
        result["spike_table"]["amplitude"].mean()
    )

    mean_widths.append(
        result["spike_table"]["half_width"].mean() * 1000
    )

    mean_isis.append(
        result["isi_stats"]["mean_isi"] * 1000
    )

    cv_isis.append(
        result["isi_stats"]["cv_isi"]
    )

    burst_counts.append(
        len(result["bursts"])
    )


# Print a summary
print("\nSUMMARY OF ALL TESTS")
print("=" * 80)

for i, name in enumerate(names):

    print(
        f"{name:18s} | "
        f"Rate: {firing_rates[i]:6.2f} Hz | "
        f"Amplitude: {mean_amplitudes[i]:6.2f} | "
        f"Width: {mean_widths[i]:6.2f} ms | "
        f"ISI: {mean_isis[i]:6.2f} ms | "
        f"CV: {cv_isis[i]:5.2f} | "
        f"Bursts: {burst_counts[i]}"
    )


# ============================================================
# GRAPH
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Firing rate
axes[0, 0].bar(names, firing_rates)
axes[0, 0].set_title("Firing Rate")
axes[0, 0].set_ylabel("Hz")
axes[0, 0].tick_params(axis="x", rotation=45)

# Spike amplitude
axes[0, 1].bar(names, mean_amplitudes)
axes[0, 1].set_title("Mean Spike Amplitude")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].tick_params(axis="x", rotation=45)

# Spike width
axes[1, 0].bar(names, mean_widths)
axes[1, 0].set_title("Mean Spike Half-Width")
axes[1, 0].set_ylabel("Milliseconds")
axes[1, 0].tick_params(axis="x", rotation=45)

# Mean ISI
axes[1, 1].bar(names, mean_isis)
axes[1, 1].set_title("Mean Inter-Spike Interval")
axes[1, 1].set_ylabel("Milliseconds")
axes[1, 1].tick_params(axis="x", rotation=45)

# ISI CV
axes[2, 0].bar(names, cv_isis)
axes[2, 0].set_title("ISI Variability (CV)")
axes[2, 0].set_ylabel("CV")
axes[2, 0].tick_params(axis="x", rotation=45)

# Bursts
axes[2, 1].bar(names, burst_counts)
axes[2, 1].set_title("Bursts Detected")
axes[2, 1].set_ylabel("Number of bursts")
axes[2, 1].tick_params(axis="x", rotation=45)

plt.suptitle(
    "Neural Recording Pipeline — Simulated Dataset Comparison",
    fontsize=16
)

plt.tight_layout()
plt.show()