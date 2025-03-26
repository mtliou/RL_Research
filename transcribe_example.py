import io
import queue
import asyncio
import os
import time
import pyaudio
import argparse
from openai import OpenAI
import speech_recognition as sr
import numpy as np
import concurrent.futures
from threading import Thread

# Define supported languages
SUPPORTED_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "uk": "Ukrainian",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai"
}

def list_languages():
    """Print list of supported languages"""
    print("\nSupported Languages:")
    print("-" * 30)
    for code, name in SUPPORTED_LANGUAGES.items():
        print(f"{code:<6} : {name}")
    print("-" * 30)
    print("Use 'auto' for automatic language detection\n")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Real-time speech transcription using OpenAI GPT-4o Transcribe')
parser.add_argument('--language', '-l', type=str, default='en', 
                    help='Language code for transcription (default: en)')
parser.add_argument('--list-languages', action='store_true', 
                    help='List all supported language codes')
parser.add_argument('--energy-threshold', type=int, default=150,
                    help='Energy threshold for detecting speech (default: 150)')
args = parser.parse_args()

# If --list-languages flag is used, print languages and exit
if args.list_languages:
    list_languages()
    exit(0)

# Set language - None for auto-detection, otherwise use the code
LANGUAGE = None if args.language.lower() == 'auto' else args.language.lower()

# Check if selected language is supported
if LANGUAGE and LANGUAGE not in SUPPORTED_LANGUAGES:
    print(f"Error: Language code '{LANGUAGE}' is not supported.")
    list_languages()
    exit(1)

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    print("Please set it with: $env:OPENAI_API_KEY = 'your-key'")
    exit(1)
else:
    print("API key found")

# Configuration
CHUNK_SIZE = 1024                 # Process audio in larger chunks
PHRASE_TIME_LIMIT = 2             # Even shorter phrases for faster response
ENERGY_THRESHOLD = args.energy_threshold  # Use command line argument value
API_TIMEOUT = 2                   # Further reduce timeout
LOGPROB_THRESHOLD = -0.1          # More permissive threshold for faster acceptance
SAMPLE_RATE = 16000               # Optimized sample rate for speech
ENABLE_DEBUG = False              # Set to False for less console output
MAX_API_WORKERS = 3               # More workers for faster processing

# Print current configuration
print(f"Transcription language: {SUPPORTED_LANGUAGES.get(LANGUAGE, 'Auto-detect')}")
print(f"Energy threshold: {ENERGY_THRESHOLD}")
print(f"Workers: {MAX_API_WORKERS}")

# Initialize OpenAI client with optimized settings
client = OpenAI(
    api_key=api_key,
    timeout=API_TIMEOUT,          # Shorter timeout for API calls
    max_retries=1,                # Reduce retries to minimize latency
)

# Get microphone info using PyAudio directly
print("Detecting microphones using PyAudio...")
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
input_devices = []

for i in range(0, numdevices):
    if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        input_devices.append(i)
        print(f"Input Device id {i} - {device_info.get('name')}")

p.terminate()

if not input_devices:
    print("ERROR: No microphones detected. Please connect a microphone and try again.")
    exit(1)

# Use the first available microphone
MIC_INDEX = input_devices[0]
print(f"Using microphone with index {MIC_INDEX}")

# Create _temp directory if it doesn't exist
os.makedirs("_temp", exist_ok=True)


class VoiceTranscriber:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
        # Optimize for speed
        self.recognizer.dynamic_energy_threshold = False  # Use fixed threshold for speed
        self.recognizer.energy_threshold = ENERGY_THRESHOLD
        self.recognizer.pause_threshold = 0.7   # Faster response to pauses
        self.recognizer.non_speaking_duration = 0.5  # Faster detection of speech end
        
        self.phrase_time_limit = PHRASE_TIME_LIMIT
        self.audio_queue = queue.Queue()
        self.api_queue = queue.Queue(maxsize=10)  # Queue for API calls
        self.source = None
        
        # Stats tracking
        self.last_valid_transcript_time = 0
        self.valid_transcript_count = 0
        self.total_transcript_count = 0
        
        # Parallel processing
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.is_running = True

    def _record_callback(self, _, audio: sr.AudioData):
        """Callback to handle incoming audio asynchronously."""
        try:
            raw_audio = audio.get_raw_data()
            self.audio_queue.put_nowait(raw_audio)  # Non-blocking put
        except Exception as e:
            if ENABLE_DEBUG:
                print("[Record Error]:", e)

    def _debug_print(self, message):
        """Print debug messages only when enabled"""
        if ENABLE_DEBUG:
            print(message)
    
    def _api_worker(self):
        """Process API requests in a separate thread"""
        while self.is_running:
            try:
                data = self.api_queue.get(timeout=1)
                if data is None:
                    continue
                    
                audio_data, audio_energy = data
                self._process_single_audio(audio_data, audio_energy)
                self.api_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"API worker error: {e}")

    def _process_single_audio(self, audio_data, audio_energy):
        """Process a single audio chunk - runs in worker thread"""
        try:
            self.total_transcript_count += 1
            
            # Skip WAV conversion for faster processing
            wav_data = audio_data.get_wav_data()
            wav_stream = io.BytesIO(wav_data)
            wav_stream.name = "_temp/audio.wav"
            wav_stream.seek(0)
            
            # Prepare transcription parameters
            transcription_params = {
                "model": "gpt-4o-mini-transcribe",  # Faster model
                "file": wav_stream,
                "temperature": 0.0,  # Zero temperature for fastest, most deterministic response
                "response_format": "json",
                "include": ["logprobs"],
            }
            
            # Add language parameter if specified
            if LANGUAGE:
                transcription_params["language"] = LANGUAGE
                language_indicator = f"[{LANGUAGE}]"
            else:
                language_indicator = "[auto]"
            
            # Streamlined API call
            response = client.audio.transcriptions.create(**transcription_params)
            
            is_valid = self.is_valid_logprobs(response.logprobs)
            
            if is_valid:
                # Clean, simplified output for speed
                print(f"🎤 {language_indicator} {response.text}")
                self.last_valid_transcript_time = time.time()
                self.valid_transcript_count += 1
            
            # Only print stats every 20 transcripts to reduce console overhead
            if self.total_transcript_count % 20 == 0 and self.total_transcript_count > 0:
                valid_percent = (self.valid_transcript_count / self.total_transcript_count) * 100
                lang_display = SUPPORTED_LANGUAGES.get(LANGUAGE, "Auto-detect")
                print(f"Stats: {self.valid_transcript_count}/{self.total_transcript_count} valid ({valid_percent:.1f}%) - Language: {lang_display}")

        except Exception as e:
            # Simplified error handling for speed
            print(f"Error: {str(e)}")

    async def _record_loop(self):
        """Starts background listening and keeps the event loop alive."""
        mic = sr.Microphone(device_index=MIC_INDEX, sample_rate=SAMPLE_RATE)

        with mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Faster calibration
            self.source = source
            print("Microphone calibrated.")

        self.recognizer.listen_in_background(
            mic, self._record_callback, phrase_time_limit=self.phrase_time_limit
        )
        print(f"Listening started with {MAX_API_WORKERS} parallel processors")
        print("Speak clearly into the microphone...")
        print("-" * 50)
        
        # Start API worker threads
        api_threads = []
        for _ in range(MAX_API_WORKERS):  # Use multiple threads for API calls
            t = Thread(target=self._api_worker)
            t.daemon = True
            t.start()
            api_threads.append(t)

        # Keep the event loop alive
        while True:
            await asyncio.sleep(0.1)

    def is_valid_logprobs(self, logprobs, threshold=LOGPROB_THRESHOLD) -> bool:
        if not logprobs:
            return False
        avg_logprob = sum(lp.logprob for lp in logprobs) / len(logprobs)
        return avg_logprob > threshold

    async def _process_audio(self):
        while True:
            try:
                self._debug_print("Waiting for audio...")
                raw_audio = self.audio_queue.get()
                self._debug_print("Got audio chunk, processing...")
            except queue.Empty:
                await asyncio.sleep(0.01)  # Short sleep to avoid CPU spinning
                continue

            try:
                audio_data = sr.AudioData(
                    raw_audio, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH
                )
                
                # Fast energy check using numpy
                audio_array = np.frombuffer(raw_audio, dtype=np.int16)
                audio_energy = np.abs(audio_array).mean()
                
                if audio_energy < ENERGY_THRESHOLD:
                    self._debug_print(f"Audio energy too low ({audio_energy:.2f}), skipping")
                    continue
                
                # Add to API queue for parallel processing
                try:
                    self.api_queue.put_nowait((audio_data, audio_energy))
                except queue.Full:
                    self._debug_print("API queue full, dropping audio chunk")

            except Exception as e:
                print("[Audio Processing Error]:", e)
                if ENABLE_DEBUG:
                    import traceback
                    traceback.print_exc()

    async def run(self):
        """Run recording and transcription tasks concurrently."""
        self.is_running = True
        await asyncio.gather(self._record_loop(), self._process_audio())


if __name__ == "__main__":
    print("Starting fast real-time transcription...")
    transcriber = VoiceTranscriber()
    try:
        asyncio.run(transcriber.run())
    except KeyboardInterrupt:
        print("\nStopping transcription...")
        transcriber.is_running = False
    finally:
        print("Transcription ended.")
