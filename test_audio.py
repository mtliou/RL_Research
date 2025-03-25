import pyaudio

p = pyaudio.PyAudio()

print("PyAudio version:", pyaudio.__version__)
print("\nAudio devices detected:")
print("-" * 70)

# List input devices
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
input_devices = 0

for i in range(0, numdevices):
    if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
        input_devices += 1
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        print(f"Input Device id {i} - {device_info.get('name')}")
        print(f"  Sample Rate: {int(device_info.get('defaultSampleRate'))}Hz")
        print(f"  Max Input Channels: {device_info.get('maxInputChannels')}")
        print()

if input_devices == 0:
    print("No input devices found. Make sure your microphone is connected and enabled.")
    print("Check your Windows sound settings to ensure microphones are enabled.")

# Clean up
p.terminate() 