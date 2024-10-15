import myo
import time
from scipy.io import savemat
import datetime
import numpy as np

class MyoListener(myo.DeviceListener):
    def __init__(self, energy_threshold=50):
        self.current_data = {
            'emg': [],
            'timestamps': []
        }
        self.all_data = {
            'emg': [],
            'timestamps': []
        }
        self.energy_threshold = energy_threshold  # Próg energii dla wyzwalacza
        self.active = False  # Flaga wyzwalacza, czy jesteśmy w trybie zapisu

    def on_paired(self, event):
        print("Myo is paired!")
        event.device.stream_emg(myo.StreamEmg.enabled)

    def on_unpaired(self, event):
        return False  # Stop the hub

    def on_emg(self, event):
        emg = event.emg
        if emg:
            # Wywołanie funkcji wyzwalacza
            self.check_trigger(emg)
        else:
            print("No EMG data received!")

    def check_trigger(self, emg):
        """
        Funkcja do wykrywania progu aktywacji na podstawie energii sygnału EMG.
        """
        energy = self.calculate_energy(emg)
        print(f"Calculated energy: {energy}")
        
        if energy > self.energy_threshold:
            if not self.active:
                print("Activation triggered - starting to record data")
                self.active = True  # Aktywacja, jeśli przekroczono próg
            self.current_data['emg'].append(emg)
            self.current_data['timestamps'].append(datetime.datetime.now().timestamp())
        elif self.active:
            # Zakończenie, gdy energia spada poniżej progu
            print("Deactivation triggered - stopping data recording and storing the segment")
            self.active = False
            self.store_segment()  # Przechowywanie segmentu danych
            self.reset_current_data()  # Reset danych z bieżącego segmentu

    def calculate_energy(self, emg):
        """
        Oblicza energię sygnału EMG jako sumę kwadratów wartości EMG.
        """
        return sum([val**2 for val in emg])

    def store_segment(self):
        """
        Przechowuje bieżący segment danych EMG do zbioru wszystkich danych.
        """
        if not self.current_data['emg']:
            print("No data to store")
            return
        
        self.all_data['emg'].extend(self.current_data['emg'])
        self.all_data['timestamps'].extend(self.current_data['timestamps'])
        print(f"Stored {len(self.current_data['emg'])} EMG samples in total data.")

    def reset_current_data(self):
        """
        Resetuje dane zebrane w trakcie aktywacji wyzwalacza.
        """
        self.current_data = {
            'emg': [],
            'timestamps': []
        }

    def save_data(self, filename='myo_emg_data.mat'):
        """
        Zapisuje wszystkie dane EMG w formacie mat po zakończeniu zbierania.
        """
        print(f"Saving all EMG data to file {filename}...")
        segmented_data = {
            'emg': np.array(self.all_data['emg']),
            'timestamps': np.array(self.all_data['timestamps'])
        }
        savemat(filename, segmented_data)
        print(f"All EMG data saved to {filename}")

def main():
    myo.init()
    hub = myo.Hub()
    listener = MyoListener()

    try:
        while True:
            hub.run(listener.on_event, 500)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Program interrupted")
    finally:
        print("Shutting down Myo Hub")
        listener.save_data()  # Zapisanie danych po zakończeniu

if __name__ == '__main__':
    main()
