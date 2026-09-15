import pandas as pd
import matplotlib.pyplot as plt

#load recording csv into dataframe
def load_recording(filepath):
    df = pd.read_csv(filepath)
    return df

#estimate sampling rate (Hz) from time column
#takes time between samples to get 1/dt, samples/sec
def get_sampling_rate(df):
    dt = df["time"].iloc[1] - df["time"].iloc[0]
    return round(1 / dt)

def get_duration(df):
    return df["time"].iloc[-1] - df["time"].iloc[0]

def get_voltage_stats(df):
    v = df["voltage"]
    return {
        "min": v.min(),
        "max": v.max(),
        "mean": v.mean(),
        "std": v.std(),
    }
#plot
def plot_voltage(df, title="Voltage vs Time"):
    plt.figure(figsize=(12, 4))
    plt.plot(df["time"], df["voltage"], linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_recording("data/sample_recording.csv")

    fs = get_sampling_rate(df)
    duration = get_duration(df)
    stats = get_voltage_stats(df)

    print(f"Recording duration: {duration:.1f} seconds")
    print(f"Sampling rate: {fs:,} Hz")
    print(f"Minimum voltage: {stats['min']:.1f} mV")
    print(f"Maximum voltage: {stats['max']:.1f} mV")
    print(f"Mean voltage: {stats['mean']:.1f} mV")
    print(f"Standard deviation: {stats['std']:.1f} mV")

    plot_voltage(df)