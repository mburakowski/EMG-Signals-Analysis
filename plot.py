import matplotlib.pyplot as plt
import numpy as np
import scipy.io

def read_mat_file(file_path):
    try:
        mat_data = scipy.io.loadmat(file_path)
        return mat_data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def visualize_data(data, sample_idx=0):
    if 'osoba_4' in data:
        sample_data = data['osoba_4'][sample_idx]
        emg_signals = sample_data[0]  # Assuming first dimension is for EMG signals

        # Plot each EMG signal
        plt.figure(figsize=(15, 10))
        for i, signal in enumerate(emg_signals):
            plt.subplot(len(emg_signals), 1, i + 1)
            plt.plot(signal)
            plt.title(f'EMG Signal {i+1}')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
        
        plt.tight_layout()
        plt.show()
    else:
        print("Data not found in the provided dataset.")

# Example usage
file_path = 'C:/Users/MaciejBurakowski(258/Desktop/praca_inz/osoba_2.mat'
data = read_mat_file(file_path)
visualize_data(data)