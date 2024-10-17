import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('myo_emg_data.mat')

emg_data = np.array(data['emg'])  
timestamps = np.array(data['timestamps']).flatten()  

print(f"Shape of EMG data: {emg_data.shape}")
print(f"Shape of timestamps: {timestamps.shape}")

num_channels = emg_data.shape[1]

plt.figure(figsize=(12, 8))

for i in range(num_channels):
    plt.subplot(num_channels, 1, i+1) 
    plt.plot(timestamps, emg_data[:, i])  
    plt.title(f"EMG Channel {i+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

plt.tight_layout()
plt.show()
