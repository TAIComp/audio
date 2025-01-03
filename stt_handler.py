from google.cloud import speech
import queue
import pyaudio
import time
from datetime import datetime

class STTHandler:
    def __init__(self):
        # Initialize the Speech client
        self.client = speech.SpeechClient()
        
        # Initialize audio parameters
        self.RATE = 16000
        self.CHUNK = int(self.RATE / 10)  # 100ms chunks
        
        # Create a thread-safe queue for audio data
        self.audio_queue = queue.Queue()
        
        # Configure audio recording parameters
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.RATE,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )

        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True
        )

    def get_audio_input(self):
        """Initialize audio input with error handling and device selection."""
        try:
            audio = pyaudio.PyAudio()
            
            # List available input devices
            info = audio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            
            # Find default input device
            default_input = None
            for i in range(numdevices):
                device_info = audio.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:
                    if device_info.get('defaultSampleRate') == self.RATE:
                        default_input = i
            
            if default_input is None:
                default_input = audio.get_default_input_device_info()['index']
            
            # Open the audio stream with explicit device selection
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.RATE,
                input=True,
                input_device_index=default_input,
                frames_per_buffer=self.CHUNK,
                stream_callback=self._fill_queue
            )
            
            if not stream.is_active():
                stream.start_stream()
                
            return stream, audio
            
        except Exception as e:
            print(f"\nError initializing audio input: {e}")
            print("Please check if your microphone is properly connected and permissions are set correctly.")
            print("You may need to grant microphone access to the application.")
            raise

    def _fill_queue(self, in_data, frame_count, time_info, status_flags):
        """Callback to fill the audio queue with incoming audio data."""
        self.audio_queue.put(in_data)
        return None, pyaudio.paContinue

    def audio_input_stream(self):
        """Generator that yields audio data from the queue."""
        while True:
            data = self.audio_queue.get()
            if data is None:
                break
            yield data

    def get_streaming_requests(self):
        """Generate streaming requests from audio input."""
        return (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in self.audio_input_stream()
        )

    def start_recognition(self):
        """Start the streaming recognition process."""
        try:
            stream, audio = self.get_audio_input()
            
            requests = self.get_streaming_requests()
            responses = self.client.streaming_recognize(
                self.streaming_config,
                requests
            )
            
            return stream, audio, responses
            
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            raise

    def clear_audio_queue(self):
        """Clear the audio queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break 