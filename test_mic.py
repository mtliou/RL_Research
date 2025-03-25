import speech_recognition as sr

# List all available microphones
print("Testing microphone detection...")
try:
    mics = sr.Microphone.list_working_microphones()
    print(f"Detected {len(mics)} microphones: {mics}")
    
    if mics:
        # Try to use the first available microphone
        mic_index = list(mics)[0]
        print(f"Attempting to use microphone with index {mic_index}")
        
        r = sr.Recognizer()
        with sr.Microphone(device_index=mic_index) as source:
            print("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source)
            print("Please say something...")
            audio = r.listen(source, timeout=5)
            print("Audio captured successfully!")
    else:
        print("No microphones detected. Check your hardware connections.")
except Exception as e:
    print(f"Error testing microphone: {e}")
    import traceback
    traceback.print_exc() 