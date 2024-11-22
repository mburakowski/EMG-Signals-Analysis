import myo
import datetime
import threading
from scipy.io import savemat
import numpy as np
from collections import deque
import os
from pathlib import Path

class MyoListener(myo.DeviceListener):
    def __init__(self, window_size, overlap, trigger_threshold):
            self.window_size = window_size
            self.overlap = overlap
            self.step_size = int(window_size * (1 - overlap))
            self.trigger_threshold = trigger_threshold
            self.raw_buffer = deque(maxlen=2000)
            self.timestamps_buffer = deque(maxlen=2000)
            self.triggered_data = {'emg': [], 'timestamps': []}
            self.is_recording = False
            self.start_time = None
            self.record_start_time = None
            self.sample_count = 0
            self.measurement_counter = 0
            
            self.trigger_check_buffer = deque(maxlen=20)
            self.stability_counter = 0
            self.min_stable_samples = 10
            
            self.recording_duration = 1.0  # Dokładnie jedna sekunda
            self.fs = 200  # Częstotliwość próbkowania
            self.required_samples = int(self.recording_duration * self.fs)  # Wymagana liczba próbek
            self.activity_threshold = 0.6  # Minimalna proporcja aktywnych próbek w nagraniu
            
            self.data_folder = Path('pomiary_emg_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            self.data_folder.mkdir(exist_ok=True)
            
            print(f"\nParametry konfiguracji:")
            print(f"- Rozmiar okna: {self.window_size} próbek")
            print(f"- Nakładanie się okien: {self.overlap * 100}%")
            print(f"- Przesunięcie między oknami: {self.step_size} próbek")
            print(f"- Próg wyzwalacza: {self.trigger_threshold}")
            print(f"- Czas nagrania: {self.recording_duration} s")
            print(f"- Wymagana liczba próbek: {self.required_samples}")
            print(f"- Folder zapisu: {self.data_folder}\n")

    def check_trigger(self):
        if len(self.trigger_check_buffer) < self.trigger_check_buffer.maxlen:
            return False
            
        samples_above_threshold = 0
        for sample in self.trigger_check_buffer:
            if any(abs(x) > self.trigger_threshold for x in sample):
                samples_above_threshold += 1
        
        if self.is_recording:
            threshold_ratio = 0.3
        else:
            threshold_ratio = 0.4
        
        return samples_above_threshold >= len(self.trigger_check_buffer) * threshold_ratio

    def on_emg(self, event):
        if self.start_time is None:
            self.start_time = datetime.datetime.now()
            
        current_time = datetime.datetime.now().timestamp()
        
        self.raw_buffer.append(event.emg)
        self.trigger_check_buffer.append(event.emg)
        
        is_triggered = self.check_trigger()
        max_value = max(abs(x) for x in event.emg)
        
        if self.is_recording:
            recording_duration = current_time - self.record_start_time
            relative_time = recording_duration
            self.timestamps_buffer.append(relative_time)
            
            samples_collected = len(self.triggered_data['emg'])
            status = f"NAGRYWANIE ({recording_duration:.2f}s, {samples_collected} okien)"
            
            if recording_duration >= self.recording_duration:
                active_samples = sum(1 for t in self.trigger_check_buffer if any(abs(x) > self.trigger_threshold for x in t))
                activity_ratio = active_samples / len(self.trigger_check_buffer) if self.trigger_check_buffer else 0
                
                if activity_ratio >= 0.6:  # % próbek aktywnych
                    print(f"\nKoniec pomiaru {self.measurement_counter + 1} (aktywność: {activity_ratio:.2f})")
                    self.save_measurement()
                    self.measurement_counter += 1
                else:
                    print(f"\nPomiar odrzucony - zbyt mała aktywność ({activity_ratio:.2f})")
                
                self.is_recording = False
                self.triggered_data = {'emg': [], 'timestamps': []}
                self.timestamps_buffer.clear()
                self.sample_count = 0
                return
        else:
            status = "OCZEKIWANIE"
        
        print(f"\rCzas: {(current_time - self.start_time.timestamp()):.2f}s | EMG: {' '.join(f'{x:4d}' for x in event.emg)} | "
              f"Max: {max_value:4d} | {status}", end='')
        
        if not self.is_recording and is_triggered:
            print("\nWyzwalacz aktywowany!")
            self.is_recording = True
            self.record_start_time = current_time
            self.timestamps_buffer.clear()
            self.triggered_data = {'emg': [], 'timestamps': []}
            self.sample_count = 0
            self.save_continuous_window()
            
        elif self.is_recording and recording_duration <= self.recording_duration:
            self.save_continuous_window()

    def save_continuous_window(self):
            if len(self.raw_buffer) >= self.window_size:
                window_data = list(self.raw_buffer)[-self.window_size:]
                current_time = list(self.timestamps_buffer)[-1] if self.timestamps_buffer else 0
                
                # Tworzymy tablicę czasów o stałym wymiarze
                window_times = [current_time] * len(window_data)
                
                self.triggered_data['emg'].append(window_data)
                self.triggered_data['timestamps'].append(window_times)
                self.sample_count += 1
                
                for _ in range(self.step_size):
                    if len(self.raw_buffer) > self.window_size:
                        self.raw_buffer.popleft()
                    if len(self.timestamps_buffer) > self.window_size:
                        self.timestamps_buffer.popleft()

    def save_measurement(self):
        if not self.triggered_data['emg']:
            print("\nBrak danych do zapisania")
            return
            
        try:
            # Konwersja danych do postaci numpy arrays
            emg_data = np.array(self.triggered_data['emg'])
            timestamps = np.array(self.triggered_data['timestamps'])
            
            # Sprawdzenie wymiarów
            print(f"Kształt danych EMG: {emg_data.shape}")
            print(f"Kształt znaczników czasu: {timestamps.shape}")
            
            timestamp = datetime.datetime.now().strftime('%H%M%S')
            filename = self.data_folder / f'pomiar_{self.measurement_counter + 1:03d}_{timestamp}.mat'
            
            print(f"\nZapisuję dane do: {filename}")
            segmented_data = {
                'emg': emg_data,
                'timestamps': timestamps
            }
            savemat(filename, segmented_data)
            print(f"Zapisano {self.sample_count} okien")
            
        except Exception as e:
            print(f"Błąd podczas zapisu danych: {str(e)}")
            print("Struktura danych:")
            print(f"EMG data length: {len(self.triggered_data['emg'])}")
            print(f"Timestamps length: {len(self.triggered_data['timestamps'])}")
            if self.triggered_data['emg']:
                print(f"Przykładowe okno EMG shape: {np.array(self.triggered_data['emg'][0]).shape}")
            if self.triggered_data['timestamps']:
                print(f"Przykładowe okno timestamps shape: {np.array(self.triggered_data['timestamps'][0]).shape}")
            raise

    def on_paired(self, event):
        print("Myo został sparowany!")
        event.device.stream_emg(myo.StreamEmg.enabled)
        self.start_time = datetime.datetime.now()

def main():
    myo.init()
    hub = myo.Hub()
    
    config = {
        'window_size': 40,
        'overlap': 0.4,
        'trigger_threshold': 20
    }
    
    listener = MyoListener(**config)

    try:
        print("Rozpoczynam nasłuchiwanie...")
        while True:
            hub.run(listener.on_event, 1)
    except KeyboardInterrupt:
        print("\nProgram przerwany przez użytkownika")
    finally:
        print("Zamykam Myo Hub")

if __name__ == '__main__':
    main()