import myo
import time

class MyoListener(myo.DeviceListener):
    def on_paired(self, event):
        print("Myo is paired!")

    def on_unpaired(self, event):
        return False  # Stop the hub

    def on_pose(self, event):
        print(f"Pose detected: {event.pose}")

    def on_orientation(self, event):
        orientation = event.orientation
        #print(f"Orientation: {orientation}")

    def on_emg(self, event):
        emg = event.emg
        print(f"EMG data: {emg}")

def main():
    myo.init()
    hub = myo.Hub()
    listener = MyoListener()

    try:
        while hub.run(listener.on_event, 500):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Program interrupted")
    finally:
        print("Shutting down Myo Hub")
        hub.shutdown()

if __name__ == '__main__':
    main()
