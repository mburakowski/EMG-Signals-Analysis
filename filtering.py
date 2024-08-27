import myo
import time
import numpy as np
from scipy.io import savemat
from scipy.signal import butter, lfilter
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

class MyoListener(myo.DeviceListener):
    def __init__(self):
        self.emg_data = []
        self.sample_rate = 200  # Przykładowa wartość, zależy od specyfikacji Myo
        self.window_size = 100  # Okno czasowe (w próbkach)
        self.overlap = 50  # Przesunięcie okna (w próbkach)
        self.filtered_data = []
        self.model = joblib.load('trained_model.pkl')  # Załaduj wytrenowany model ML

    def on_paired(self, event):
        print("Myo is paired!")
        event.device.stream_emg(True)  # Włącza strumieniowanie EMG

    def on_unpaired(self, event):
        return False  # Stop the hub

    def on_emg(self, event):
        emg = np.array(event.emg)
        self.emg_data.append(emg)

        if len(self.emg_data) >= self.window_size:
            window_data = np.array(self.emg_data[-self.window_size:])
            processed_signal = self.process_emg(window_data)
            self.filtered_data.append(processed_signal)
            self.detect_intent(processed_signal)

            # Usuwanie nadmiarowych danych przy przesuwaniu okna
            self.emg_data = self.emg_data[-(self.window_size - self.overlap):]

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def process_emg(self, emg_window):
        # Filtracja pasmowa sygnału EMG
        lowcut = 20.0
        highcut = 450.0
        b, a = self.butter_bandpass(lowcut, highcut, self.sample_rate, order=3)
        filtered = lfilter(b, a, emg_window, axis=0)
        
        # Wyodrębnienie cech
        rms = np.sqrt(np.mean(filtered**2, axis=0))  # Root Mean Square
        mav = np.mean(np.abs(filtered), axis=0)      # Mean Absolute Value

        # Łączenie cech w jeden wektor
        features = np.concatenate([rms, mav])
        return features

    def detect_intent(self, features):
        prediction = self.model.predict(features.reshape(1, -1))
        print(f"Detected intent: {prediction}")

    def save_data(self, filename='emg_data.mat'):
        savemat(filename, {
            'emg_data': self.emg_data,
            'filtered_data': self.filtered_data
        })
        print(f"Data saved to {filename}")
        self.plot_emg_data()

    def plot_emg_data(self):
        plt.figure(figsize=(10, 6))
        plt.plot(np.array(self.emg_data).reshape(-1, 8))
        plt.title("Raw EMG data")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.show()

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
        listener.save_data()

if __name__ == '__main__':
    main()
