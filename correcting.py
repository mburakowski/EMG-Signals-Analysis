import myo
import time
from scipy.io import savemat
import datetime
import numpy as np
from collections import Counter

class MyoListener(myo.DeviceListener):
    def __init__(self, window_size_ms=1000):
        self.data = {
            'poses': [],
            'orientations': [],
            'emg': [],
            'timestamps': []
        }
        self.window_size_ms = window_size_ms
        self.segments = []  

    def on_paired(self, event):
        print("Myo is paired!")
        event.device.stream_emg(myo.StreamEmg.enabled)

    def on_unpaired(self, event):
        return False  # Stop the hub

    def on_pose(self, event):
        print(f"Pose detected: {event.pose}")
        self.data['poses'].append(event.pose.value)
        self.data['timestamps'].append(datetime.datetime.now().timestamp())

    def on_orientation(self, event):
        orientation = event.orientation
        print(f"Orientation: {orientation}")
        self.data['orientations'].append((orientation.x, orientation.y, orientation.z, orientation.w))
        self.data['timestamps'].append(datetime.datetime.now().timestamp())

    def on_emg(self, event):
        emg = event.emg
        if emg:
            print(f"EMG data: {emg}")
        else:
            print("No EMG data received!")
        self.data['emg'].append(emg)
        
    def segment_data(self):
        """
        Segmentuj dane w oknach czasowych o długości self.window_size_ms.
        """
        window_size_sec = self.window_size_ms / 1000.0
        start_time = self.data['timestamps'][0]
        end_time = start_time + window_size_sec
        window_data = {
            'poses': [],
            'orientations': [],
            'emg': []
        }

        print("Starting segmentation...")
        for i in range(len(self.data['timestamps'])):
            timestamp = self.data['timestamps'][i]

            if timestamp <= end_time:
                # Collecting data
                if i < len(self.data['poses']):
                    window_data['poses'].append(self.data['poses'][i])
                if i < len(self.data['orientations']):
                    window_data['orientations'].append(self.data['orientations'][i])
                if i < len(self.data['emg']):
                    window_data['emg'].append(self.data['emg'][i])
            else:
                # Closing window
                print(f"Segment created with {len(window_data['orientations'])} orientations and {len(window_data['emg'])} EMG samples")
                if window_data['poses']:
                    most_common_pose = Counter(window_data['poses']).most_common(1)[0][0]
                else:
                    most_common_pose = -1  # brak gestu

                # windows > segments
                self.segments.append({
                    'pose': most_common_pose,
                    'orientations': np.array(window_data['orientations']),
                    'emg': np.array(window_data['emg'])
                })

                # window restart
                window_data = {
                    'poses': [],
                    'orientations': [],
                    'emg': []
                }
                start_time = timestamp
                end_time = start_time + window_size_sec

        print("Segmentation finished.")

    def save_data(self, filename='myo_segmented_data.mat'):
        """
        Zapisuje dane w formacie mat po segmentacji.
        """
        print(f"Saving {len(self.segments)} segments to file...")
        segmented_data = {
            'segments': self.segments
        }
        savemat(filename, segmented_data)
        print(f"Segmented data saved to {filename}")

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
        listener.segment_data()  # Segmenting b4 saving
        listener.save_data()  

if __name__ == '__main__':
    main()
