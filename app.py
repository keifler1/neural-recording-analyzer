import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from simulate_neuron import simulate_neuron
from filter_signal import filter_signal
from detect_spikes import compute_threshold, detect_spikes, score_detections
from firing_analysis import compute_firing_rate, compute_isis, isi_statistics
from burst_detection import detect_bursts

st.set_page_config(page_title="Neural Recording Analyzer", layout="wide")

st.title("Neural Recording Analyzer")
st.write("Visualizing the effects of recording parameters on spike detection and firing patterns through simulation")

with st.expander("New to this? Read this first"):
    st.markdown("""
    Neurons communicate using brief electrical events called **action potentials**
    (spikes), which are sharp jumps in voltage lasting about 1ms. A recording here is a
    simulated voltage trace over time including a flat resting level, occasional spikes,
    and background noise.

    This demo generates a recording and runs it through a spike detection pipeline.
    Adjust the parameters to change the quality of spike detection.
    """)

# --- Sidebar controls ---
st.sidebar.header("Simulation parameters")

duration = st.sidebar.slider(
    "Duration (s)", 1, 30, 10,
    help="How many seconds of recording to simulate."
)
firing_rate = st.sidebar.slider(
    "Firing rate (Hz)", 1, 50, 10,
    help="How often the neuron fires, on average, in spikes/second."
)
noise_level = st.sidebar.slider(
    "Noise level", 0.5, 20.0, 3.0, step=0.5,
    help="How much random electrical noise is present in the recording, such as "
         "from nearby cells. Higher values make spikes harder to tell apart from noise."
)
seed = st.sidebar.number_input(
    "Random seed", value=42, step=1,
    help="Controls the randomness used to generate the recording."
)

# --- Run pipeline ---
sampling_rate = 10000
df, true_spike_times = simulate_neuron(
    duration=duration, sampling_rate=sampling_rate,
    firing_rate=firing_rate, noise_level=noise_level, seed=seed
)
voltage = df["voltage"].values
time = df["time"].values

filtered = filter_signal(voltage, sampling_rate, low_cutoff=300, high_cutoff=3000)
threshold = compute_threshold(filtered, n_std=5)
detected = detect_spikes(filtered, time, sampling_rate, threshold=threshold)
spike_times = np.array([s["time"] for s in detected])

tp, fp, fn = score_detections(true_spike_times, spike_times, tolerance_ms=2, sampling_rate=sampling_rate)
precision = 100 * tp / (tp + fp) if (tp + fp) else 0
recall = 100 * tp / (tp + fn) if (tp + fn) else 0

rate = compute_firing_rate(spike_times, duration)
isis = compute_isis(spike_times)
isi_stats = isi_statistics(isis)
bursts, _ = detect_bursts(spike_times, min_spikes_per_burst=3)

# --- Metrics row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Spikes detected", len(spike_times),
            help="How many total spikes the detection algorithm found in this recording.")
col2.metric("Firing rate", f"{rate:.1f} Hz",
            help="Detected spikes divided by recording duration.")
col3.metric("Precision", f"{precision:.1f}%",
            help="Of all the spikes the algorithm detected, what percentage were "
                 "real spikes rather than noise?")
col4.metric("Recall", f"{recall:.1f}%",
            help="Of all the real spikes that occurred, what percentage did the "
                 "algorithm successfully detect?")

# --- Plot: raw vs filtered with detections ---
st.subheader("Signal: filtered trace with detected spikes")
st.caption("The blue line is the voltage recording after filtering (which removes very "
           "slow drift and very high-frequency noise, making spikes easier to isolate). "
           "Red X marks show where the algorithm detected a spike.")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(time, filtered, linewidth=0.5, label="Filtered signal")
if len(spike_times) > 0:
    ax.scatter(spike_times, [s["peak"] for s in detected], color="red", marker="x", label="Detected", zorder=5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage")
ax.legend()
st.pyplot(fig)

# --- Plot: ISI histogram ---
st.subheader("Inter-spike interval distribution")
st.caption("An inter-spike interval (ISI) is the gap in time between one spike and the "
           "next. This histogram shows how those gaps are distributed across the whole "
           "recording.")
fig2, ax2 = plt.subplots(figsize=(8, 3))
if len(isis) > 0:
    ax2.hist(isis * 1000, bins=30, color="steelblue", edgecolor="black")
ax2.set_xlabel("ISI (ms)")
ax2.set_ylabel("Count")
st.pyplot(fig2)

st.caption(
    f"**CV (coefficient of variation): {isi_stats['cv_isi']:.2f}** — a measure of how "
    f"irregular the firing is. A CV near 1.0 shows random firing; much lower means "
    f"very regular firing and much higher suggests bursting.  \n"
    f"**Bursts detected: {len(bursts)}** — a burst is a cluster of several spikes "
    f"firing much closer together than is typical.  \n"
    f"**Adaptive threshold: {threshold:.2f}** — the voltage level the algorithm used "
    f"to determine a spike. This is calculated from the recording's noise level and "
    f"adjusts automatically."
)