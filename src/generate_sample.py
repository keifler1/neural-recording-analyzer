import numpy as np
import pandas as pd

duration = 30        # seconds
sampling_rate = 10000  # Hz

n_samples = duration * sampling_rate
time = np.arange(n_samples) / sampling_rate

# resting potential + noise
np.random.seed(42)
voltage = -65 + np.random.normal(0, 2, size=n_samples)

df = pd.DataFrame({"time": time, "voltage": voltage})
df.to_csv("data/sample_recording.csv", index=False)

print(f"Saved {len(df)} samples to data/sample_recording.csv")