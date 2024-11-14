import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.io import loadmat

class EMGAnalyzer:
    def __init__(self, sampling_rate=200):
        self.sampling_rate = sampling_rate
        plt.style.use('default')
        
    def prepare_continuous_signal(self, emg_windows, time_windows, channel):
        # Sortowanie okien według czasu
        start_times = [times[0] for times in time_windows]
        sorted_indices = np.argsort(start_times)
        
        continuous_data = []
        continuous_times = []
        
        for idx in sorted_indices:
            window_data = emg_windows[idx][:, channel]
            window_times = time_windows[idx]
            
            # Jeśli to pierwsze okno, po prostu dodaj dane
            if not continuous_times:
                continuous_data.extend(window_data)
                continuous_times.extend(window_times)
                continue
            
            # Znajdź punkt łączenia z poprzednim oknem
            overlap_start = np.where(window_times > continuous_times[-1])[0]
            if len(overlap_start) > 0:
                continuous_data.extend(window_data[overlap_start[0]:])
                continuous_times.extend(window_times[overlap_start[0]:])
        
        return np.array(continuous_data), np.array(continuous_times)

    def analyze_channel(self, emg_windows, time_windows, channel):
        # Przygotuj ciągły sygnał
        channel_data, times = self.prepare_continuous_signal(emg_windows, time_windows, channel)
        
        # Obliczanie STFT
        window_size = min(512, len(channel_data) // 8)
        noverlap = window_size * 3 // 4
        
        f, t, Zxx = signal.stft(channel_data,
                               fs=self.sampling_rate,
                               window='hann',
                               nperseg=window_size,
                               noverlap=noverlap)
        
        return f, t, Zxx, channel_data, times

    def plot_channel_analysis(self, channel_number, emg_data, timestamps):
        f, t, Zxx, channel_data, times = self.analyze_channel(emg_data, timestamps, channel_number)
        
        fig = plt.figure(figsize=(15, 10))
        
        # Wykres sygnału
        ax1 = plt.subplot(211)
        ax1.plot(times, channel_data, 'b-', linewidth=0.5, alpha=0.8)
        ax1.set_title(f'EMG Channel {channel_number + 1} Signal', pad=15, fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time [s]', fontsize=10)
        ax1.set_ylabel('Amplitude', fontsize=10)
        
        ax1.grid(True, which='major', linestyle='-', alpha=0.3)
        ax1.grid(True, which='minor', linestyle=':', alpha=0.2)
        ax1.minorticks_on()
        
        max_amp = np.max(np.abs(channel_data))
        ax1.set_ylim(-max_amp * 1.1, max_amp * 1.1)
        
        # Spektrogram
        ax2 = plt.subplot(212)
        Zxx_abs = np.abs(Zxx)
        vmin = max(Zxx_abs.max() / 1000, Zxx_abs.min())
        spec = ax2.pcolormesh(t, f, Zxx_abs, 
                            shading='gouraud', 
                            cmap='viridis',
                            norm=LogNorm(vmin=vmin, vmax=Zxx_abs.max()))
        
        cbar = plt.colorbar(spec, ax=ax2)
        cbar.set_label('Magnitude [dB]', fontsize=10)
        
        ax2.set_title(f'Channel {channel_number + 1} Spectrogram', pad=15, fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency [Hz]', fontsize=10)
        ax2.set_xlabel('Time [s]', fontsize=10)
        ax2.set_ylim(0, 100)
        
        ax2.grid(True, which='major', linestyle='-', alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        
        return fig

    def analyze_full_signal(self, data_file='myo_emg_data.mat'):
        print("Loading data...")
        mat_data = loadmat(data_file)
        emg_data = mat_data['emg']
        timestamps = mat_data['timestamps']
        
        num_channels = emg_data[0].shape[1]
        print(f"Analyzing {num_channels} channels...")
        
        for channel in range(num_channels):
            print(f"Processing channel {channel + 1}...")
            fig = self.plot_channel_analysis(channel, emg_data, timestamps)
            plt.show()
            plt.close(fig)

def main():
    analyzer = EMGAnalyzer()
    analyzer.analyze_full_signal()

if __name__ == '__main__':
    main()