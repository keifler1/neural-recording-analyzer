import numpy as np
import pandas as pd

def make_spike_waveform(sampling_rate, spike_duration=0.002, peak=30, baseline=-65):
    #generates a spike waveform with steep rise/fall
    #returns array of voltage values

    n_points = int(spike_duration * sampling_rate)
    t = np.linspace(0, 1, n_points)

    #asymmetric, action potential rises quickly, falls slowly
    #formula of bell curve
    rise = np.exp(-((t - 0.3) ** 2) / (2 * 0.05 ** 2))
    waveform = baseline + (peak - baseline) * rise

    return waveform

#test plot
#if __name__ == "__main__":
    import matplotlib.pyplot as plt
    wf = make_spike_waveform(sampling_rate=10000)
    plt.plot(wf)
    plt.title("Single spike waveform")
    plt.show()

#random placement of spikes, noise
#firing rate: spikes/sec
#time intervals between spikes are random but cluster around average
#Poisson process: ISIs follow exponential distribution
#use random intervals from exponential distribution

def generate_spike_times(duration, firing_rate):

    spike_times = []
    t = 0
    while t < duration:
        interval = np.random.exponential(1 / firing_rate)
        t += interval
        if t < duration:
            spike_times.append(t)
    return np.array(spike_times)

#if __name__ == "__main__":
    times = generate_spike_times(duration=10, firing_rate=5)
    print(f"Generated {len(times)} spikes")
    print(times[:10])

#generate a simulated recording
#returns df with time and voltage columns, spike times

def simulate_neuron(duration, sampling_rate, firing_rate, noise_level=2, seed=None):

    if seed is not None:
        np.random.seed(seed)

    n_samples = int(duration * sampling_rate)
    time = np.arange(n_samples) / sampling_rate
    voltage = np.full(n_samples, -65.0)

    spike_times = generate_spike_times(duration, firing_rate)
    spike_waveform = make_spike_waveform(sampling_rate)
    spike_len = len(spike_waveform)

    for spike_t in spike_times:
        start_idx = int(spike_t * sampling_rate)
        end_idx = start_idx + spike_len
        if end_idx < n_samples:
            voltage[start_idx:end_idx] = spike_waveform

    #add noise
    voltage += np.random.normal(0, noise_level, size=n_samples)

    df = pd.DataFrame({"time": time, "voltage": voltage})
    return df, spike_times

#testing model

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df, spike_times = simulate_neuron(
        duration=5,
        sampling_rate=10000,
        firing_rate=5,
        noise_level=2,
        seed=42
    )

    print(f"Generated {len(spike_times)} spikes in 5 seconds")

    plt.figure(figsize=(12, 4))
    plt.plot(df["time"], df["voltage"], linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title(f"Simulated neuron — {len(spike_times)} spikes")
    plt.tight_layout()
    plt.show()
  