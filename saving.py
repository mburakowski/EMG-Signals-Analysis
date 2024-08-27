import myo
import time
from scipy.io import savemat

class MyoListener(myo.DeviceListener):
    def __init__(self):
        self.data = {
            'poses': [],
            'orientations': [],
            'emg': []
        }

    def on_paired(self, event):
        print("Myo is paired!")

    def on_unpaired(self, event):
        return False  # Stop the hub

    def on_pose(self, event):
        print(f"Pose detected: {event.pose}")
        self.data['poses'].append(event.pose.value)

    def on_orientation(self, event):
        orientation = event.orientation
        print(f"Orientation: {orientation}")
        self.data['orientations'].append((orientation.x, orientation.y, orientation.z, orientation.w))

    def on_emg(self, event):
        emg = event.emg
        print(f"EMG data: {emg}")
        self.data['emg'].append(emg)

    def save_data(self, filename='myo_data.mat'):
        savemat(filename, self.data)
        print(f"Data saved to {filename}")

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
