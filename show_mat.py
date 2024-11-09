import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from scipy.io import loadmat
import numpy as np

class EMGVisualizer:
    def __init__(self):
        # Inicjalizacja buforów dla danych
        self.visualization_buffer = [[] for _ in range(8)]
        self.time_buffer = []
        self.setup_plot()
        
    def setup_plot(self):
        plt.style.use('default')
        self.fig, self.axes = plt.subplots(4, 2, figsize=(15, 10))
        self.lines = []
        self.axes = self.axes.flatten()
        
        for i in range(8):
            line, = self.axes[i].plot([], [], 'b-', linewidth=0.8)
            self.lines.append(line)
            self.axes[i].set_title(f'EMG Channel {i+1}', pad=10)
            
            # Konfiguracja siatki
            self.axes[i].grid(True, which='major', color='gray', linestyle='-', linewidth=0.8, alpha=0.3)
            self.axes[i].grid(True, which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.2)
            self.axes[i].minorticks_on()
            
            # Dostosowanie wyglądu
            self.axes[i].set_facecolor('#f8f8f8')
            self.axes[i].set_xlabel('Time (s)')
            self.axes[i].set_ylabel('EMG Value')
        
        self.fig.patch.set_facecolor('white')
        plt.tight_layout(pad=2.0)

    def process_data(self, emg_data, time_data):
        """Przetwarza dane do formatu odpowiedniego do wyświetlenia"""
        processed_data = [[] for _ in range(8)]
        processed_time = []
        
        # Sortowanie danych według czasu
        all_times = []
        all_values = [[] for _ in range(8)]
        
        # Zbieranie wszystkich punktów czasowych i wartości
        for window_idx in range(len(time_data)):
            window_times = time_data[window_idx]
            window_emg = emg_data[window_idx]
            
            for sample_idx in range(len(window_times)):
                time_point = window_times[sample_idx]
                all_times.append(time_point)
                for channel in range(8):
                    all_values[channel].append(window_emg[sample_idx][channel])
        
        # Sortowanie według czasu
        sorted_indices = np.argsort(all_times)
        processed_time = np.array(all_times)[sorted_indices]
        
        for channel in range(8):
            processed_data[channel] = np.array(all_values[channel])[sorted_indices]
            
        # Usuwanie duplikatów czasowych
        unique_times, unique_indices = np.unique(processed_time, return_index=True)
        processed_time = unique_times
        
        for channel in range(8):
            processed_data[channel] = np.array(processed_data[channel])[unique_indices]
        
        return processed_data, processed_time

    def update_plot(self, frame):
        try:
            data = loadmat('myo_emg_data.mat')
            if len(data['emg']) > 0:
                # Przetwarzanie danych
                processed_data, processed_time = self.process_data(data['emg'], data['timestamps'])
                
                # Aktualizacja wykresów
                for i in range(8):
                    self.lines[i].set_data(processed_time, processed_data[i])
                    
                    # Automatyczne dostosowanie zakresu osi Y
                    y_data = processed_data[i]
                    if len(y_data) > 0:
                        y_min, y_max = np.min(y_data), np.max(y_data)
                        margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
                        self.axes[i].set_ylim(y_min - margin, y_max + margin)
                    
                    # Dostosowanie zakresu osi X
                    if len(processed_time) > 0:
                        self.axes[i].set_xlim(np.min(processed_time), np.max(processed_time))
                
                self.fig.tight_layout()
                
        except Exception as e:
            print(f"Błąd podczas aktualizacji wykresu: {e}")
            
        return self.lines

    def start_visualization(self):
        self.ani = FuncAnimation(self.fig, self.update_plot, 
                               interval=50, blit=True)
        plt.show()

def main():
    print("Uruchamiam wizualizację danych EMG...")
    visualizer = EMGVisualizer()
    visualizer.start_visualization()

if __name__ == '__main__':
    main()