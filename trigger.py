import myo
import datetime
import threading
from scipy.io import savemat
import numpy as np
from collections import deque

class MyoListener(myo.DeviceListener):
    def __init__(self, window_size, overlap, trigger_threshold):
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        self.trigger_threshold = trigger_threshold
        
        # Zwiększamy rozmiar buforów dla lepszej ciągłości
        self.raw_buffer = deque(maxlen=2000)
        self.timestamps_buffer = deque(maxlen=2000)
        
        # Dane wyzwolone
        self.triggered_data = {'emg': [], 'timestamps': []}
        self.is_recording = False
        self.start_time = None
        self.sample_count = 0
        
        # Bufor dla sprawdzania wyzwalacza
        self.trigger_check_buffer = deque(maxlen=20)  # Zwiększony bufor sprawdzania
        
        print(f"\nParametry konfiguracji:")
        print(f"- Rozmiar okna: {self.window_size} próbek")
        print(f"- Nakładanie się okien: {self.overlap * 100}%")
        print(f"- Przesunięcie między oknami: {self.step_size} próbek")
        print(f"- Próg wyzwalacza: {self.trigger_threshold}\n")

    def check_trigger(self):
        """Sprawdź warunek wyzwalania z histerezą"""
        if len(self.trigger_check_buffer) < self.trigger_check_buffer.maxlen:
            return False
            
        # Liczba próbek powyżej progu
        samples_above_threshold = 0
        for sample in self.trigger_check_buffer:
            if any(abs(x) > self.trigger_threshold for x in sample):
                samples_above_threshold += 1
        
        # Wymagamy, aby 50% próbek w buforze było powyżej progu
        return samples_above_threshold >= len(self.trigger_check_buffer) * 0.5

    def save_continuous_window(self):
        """Zapisz okno danych z zachowaniem ciągłości"""
        if len(self.raw_buffer) >= self.window_size:
            window_data = list(self.raw_buffer)[-self.window_size:]
            window_times = list(self.timestamps_buffer)[-self.window_size:]
            
            self.triggered_data['emg'].append(window_data)
            self.triggered_data['timestamps'].append(window_times)
            self.sample_count += 1
            
            # Przesuwamy bufor o step_size, zachowując nakładanie
            for _ in range(self.step_size):
                if len(self.raw_buffer) > self.window_size:  # Zabezpieczenie przed opróżnieniem bufora
                    self.raw_buffer.popleft()
                    self.timestamps_buffer.popleft()

    def on_paired(self, event):
        print("Myo został sparowany!")
        event.device.stream_emg(myo.StreamEmg.enabled)
        self.start_time = datetime.datetime.now()

    def on_emg(self, event):
        if self.start_time is None:
            self.start_time = datetime.datetime.now()
            
        current_time = datetime.datetime.now().timestamp()
        relative_time = current_time - self.start_time.timestamp()
        
        # Aktualizacja buforów
        self.raw_buffer.append(event.emg)
        self.timestamps_buffer.append(relative_time)
        self.trigger_check_buffer.append(event.emg)
        
        # Wyświetlanie danych
        print(f"\rCzas: {relative_time:.2f}s | EMG: {' '.join(f'{x:4d}' for x in event.emg)} | {'NAGRYWANIE' if self.is_recording else 'OCZEKIWANIE'}", end='')
        
        # Sprawdzenie wyzwalacza
        is_triggered = self.check_trigger()
        
        # Logika nagrywania
        if not self.is_recording and is_triggered:
            print("\nWyzwalacz aktywowany!")
            self.is_recording = True
            # Zapisz dane sprzed wyzwolenia (pre-trigger)
            self.save_continuous_window()
            
        elif self.is_recording:
            # Kontynuuj zapisywanie okien
            self.save_continuous_window()
            
            # Sprawdź czy należy zakończyć nagrywanie
            if not is_triggered:
                print("\nKoniec wyzwolenia")
                self.is_recording = False
                self.save_data()

    def save_data(self, filename='myo_emg_data.mat'):
        if not self.triggered_data['emg']:
            print("\nBrak danych do zapisania")
            return
            
        print(f"\nZapisuję dane...")
        segmented_data = {
            'emg': np.array(self.triggered_data['emg']),
            'timestamps': np.array(self.triggered_data['timestamps'])
        }
        savemat(filename, segmented_data)
        print(f"Zapisano {self.sample_count} okien")
        
def main():
    myo.init()
    hub = myo.Hub()
    
    # Konfiguracja
    config = {
        'window_size': 50,
        'overlap': 0.6,
        'trigger_threshold': 30
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
        listener.save_data()

if __name__ == '__main__':
    main()