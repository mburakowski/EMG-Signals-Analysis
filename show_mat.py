import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import loadmat
import numpy as np
from pathlib import Path
import os

class EMGVisualizer:
    def __init__(self):
        self.current_measurement_index = 0
        self.measurements_files = []
        self.selected_folder = self.select_folder()
        if self.selected_folder:
            self.load_measurements_from_folder()
        
        self.setup_plot()
        
        
    def select_folder(self):
        folders = [f for f in os.listdir() if f.startswith('pomiary_emg_')]
        
        if not folders:
            print("Nie znaleziono żadnych folderów z pomiarami!")
            return None
            
        print("\nDostępne foldery z pomiarami:")
        for i, folder in enumerate(folders, 1):
            folder_path = Path(folder)
            measurement_count = len(list(folder_path.glob('pomiar_*.mat')))
            creation_time = folder[11:]
            print(f"{i}. {folder} ({measurement_count} pomiarów)")
            
        while True:
            try:
                choice = int(input("\nWybierz numer folderu do analizy: ")) - 1
                if 0 <= choice < len(folders):
                    return Path(folders[choice])
                else:
                    print("Nieprawidłowy numer folderu!")
            except ValueError:
                print("Proszę podać prawidłowy numer!")

    def load_measurements_from_folder(self):
        self.measurements_files = sorted(self.selected_folder.glob('pomiar_*.mat'))
        print(f"\nZnaleziono {len(self.measurements_files)} pomiarów w folderze {self.selected_folder}")
        
    def setup_plot(self):
        plt.style.use('default')
        self.fig = plt.figure(figsize=(15, 10))
        gs = self.fig.add_gridspec(4, 2)
        
        self.axes = []
        self.lines = []
        
        # Konfiguracja wykresów EMG
        for i in range(4):
            for j in range(2):
                ax = self.fig.add_subplot(gs[i, j])
                self.axes.append(ax)
                line, = ax.plot([], [], 'b-', linewidth=0.8)
                self.lines.append(line)
                
                ax.set_title(f'EMG Channel {i*2 + j + 1}', pad=10)
                ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.8, alpha=0.3)
                ax.grid(True, which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.2)
                ax.minorticks_on()
                ax.set_facecolor('#f8f8f8')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('EMG Value')
        
        # Przyciski nawigacji
        plt.subplots_adjust(bottom=0.1)
        self.btn_prev_ax = plt.axes([0.3, 0.02, 0.1, 0.04])
        self.btn_next_ax = plt.axes([0.6, 0.02, 0.1, 0.04])
        
        self.btn_prev = plt.Button(self.btn_prev_ax, 'Poprzedni')
        self.btn_next = plt.Button(self.btn_next_ax, 'Następny')
        
        self.btn_prev.on_clicked(self.previous_measurement)
        self.btn_next.on_clicked(self.next_measurement)
        
        self.fig.suptitle('', fontsize=12)
        self.fig.patch.set_facecolor('white')
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    def process_data(self, emg_data, time_data):
        processed_data = [[] for _ in range(8)]
        all_times = []
        all_values = [[] for _ in range(8)]
        
        for window_idx in range(len(time_data)):
            window_times = time_data[window_idx]
            window_emg = emg_data[window_idx]
            
            for sample_idx in range(len(window_times)):
                time_point = window_times[sample_idx]
                all_times.append(time_point)
                for channel in range(8):
                    all_values[channel].append(window_emg[sample_idx][channel])
        
        sorted_indices = np.argsort(all_times)
        processed_time = np.array(all_times)[sorted_indices]
        
        for channel in range(8):
            processed_data[channel] = np.array(all_values[channel])[sorted_indices]
            
        unique_times, unique_indices = np.unique(processed_time, return_index=True)
        processed_time = unique_times
        
        for channel in range(8):
            processed_data[channel] = np.array(processed_data[channel])[unique_indices]
        
        return processed_data, processed_time

    def update_plot(self, frame):
        if not self.measurements_files:
            return self.lines
            
        try:
            current_file = self.measurements_files[self.current_measurement_index]
            data = loadmat(current_file)
            
            if len(data['emg']) > 0:
                processed_data, processed_time = self.process_data(data['emg'], data['timestamps'])
                
                for i in range(8):
                    self.lines[i].set_data(processed_time, processed_data[i])
                    
                    y_data = processed_data[i]
                    if len(y_data) > 0:
                        y_min, y_max = np.min(y_data), np.max(y_data)
                        margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
                        self.axes[i].set_ylim(y_min - margin, y_max + margin)
                    if len(processed_time) > 0:
                        self.axes[i].set_xlim(np.min(processed_time), np.max(processed_time))
                
                self.fig.suptitle(
                    f'Pomiar {self.current_measurement_index + 1} z {len(self.measurements_files)}: {current_file.name}', 
                    fontsize=12
                )
                
        except Exception as e:
            print(f"Błąd podczas aktualizacji wykresu: {e}")
            
        return self.lines

    def next_measurement(self, event):
        if self.current_measurement_index < len(self.measurements_files) - 1:
            self.current_measurement_index += 1
            self.reset_axes()
            plt.draw()

    def previous_measurement(self, event):
        if self.current_measurement_index > 0:
            self.current_measurement_index -= 1
            self.reset_axes()
            plt.draw()
            
    def reset_axes(self):
        for i, ax in enumerate(self.axes):
            ax.clear()
            ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.8, alpha=0.3)
            ax.grid(True, which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.2)
            ax.minorticks_on()
            ax.set_facecolor('#f8f8f8')
            ax.set_title(f'EMG Channel {i+1}', pad=10)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('EMG Value')
            line, = ax.plot([], [], 'b-', linewidth=0.8)
            self.lines[i] = line

    def start_visualization(self):
        if not self.measurements_files:
            print("Brak pomiarów do wyświetlenia!")
            return
            
        self.ani = FuncAnimation(self.fig, self.update_plot, 
                               interval=50, blit=False)
        plt.show()

def main():
    visualizer = EMGVisualizer()
    visualizer.start_visualization()

if __name__ == '__main__':
    main()