import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import warnings
import time
from openai import OpenAI
import threading
from datetime import datetime
from google.cloud import texttospeech
import pygame
from pathlib import Path
import queue
import random
import sounddevice as sd
from pynput import keyboard
import difflib
import numpy as np
import re
from dotenv import load_dotenv
from ai_handler import AIHandler
from interruption import InterruptionHandler, ListeningState
from audio_utils import initialize_audio
from state_management import AudioState
from watchdog import WatchdogMonitor
from cleanup import CleanupHandler
from tts_handler import TTSHandler
from stt_handler import STTHandler
from audio_processing import AudioProcessor

# Load environment variables from .env file
load_dotenv()

# Verify credentials path
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if not credentials_path or not os.path.exists(credentials_path):
    raise Exception(f"Google credentials file not found at: {credentials_path}")

# Suppress ALSA warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pygame.mixer")

# Redirect ALSA errors to /dev/null
try:
    from ctypes import *
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        return  # Just return instead of pass

    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    
    try:
        asound = cdll.LoadLibrary('libasound.so.2')
        asound.snd_lib_error_set_handler(c_error_handler)
    except OSError:
        print("Warning: Could not load ALSA library. Audio might not work properly.")
            
except Exception as e:
    print(f"Warning: Could not set ALSA error handler: {e}")


import os
import pyaudio
import queue
import threading
from google.cloud import speech

class AudioTranscriber:
    def __init__(self):
        self.state = AudioState()
        try:
            # Initialize STT Handler first
            self.stt_handler = STTHandler()
            
            # Get RATE from STT Handler for other components
            self.RATE = self.stt_handler.RATE
            
            # Initialize audio settings first
            initialize_audio()
            
            # Initialize pygame mixer with error handling
            try:
                pygame.mixer.init()
            except pygame.error as e:
                print(f"Warning: Could not initialize pygame mixer: {e}")
                # Continue anyway as some features might still work
                
            # Initialize AI handler
            self.aiHandler = AIHandler()
            
            # Add stop_generation flag
            self.stop_generation = False
            
            # Initialize audio parameters
            self.current_sentence = ""
            self.last_transcript = ""
            
            # Add mic monitoring settings
            self.MONITOR_CHANNELS = 1
            self.MONITOR_DTYPE = 'float32'
            self.monitoring_active = True
            self.mic_volume = 0.5
            self.noise_gate_threshold = 0.5
            
            # Create a thread-safe queue for audio data
            self.audio_queue = queue.Queue()
            
            # Initialize the Speech client
            self.client = speech.SpeechClient()
            
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

            # New additions
            self.openai_client = OpenAI()
            self.last_speech_time = datetime.now()
            self.last_final_transcript = ""
            self.is_processing = False
            self.last_sentence_complete = False

            # Initialize TTS Handler
            self.tts_handler = TTSHandler()
            
            # Add new attributes
            self.listening_state = ListeningState.FULL_LISTENING
            self.is_speaking = False
            
            # Update OpenAI model
            self.model = "gpt-4o-mini"  
            
            # Add transcript tracking
            self.last_interim_timestamp = time.time()
            self.interim_cooldown = 0.5  # seconds between interim updates
            
            # Add new flag for interrupt handling
            self.last_interrupt_time = time.time()
            self.interrupt_cooldown = 1.0  # 1 second cooldown between interrupts
            
            # Path to prerecorded interrupt acknowledgment
            self.interrupt_audio_path = Path("interruption.mp3")
            
            if not self.interrupt_audio_path.exists():
                print(f"Warning: Interrupt audio file '{self.interrupt_audio_path}' not found!")

            # Add this before keyboard listener initialization
            def on_press(key):
                try:
                    if hasattr(key, 'char') and key.char == '`':
                        print("\nBacktick key interrupt detected!")
                        self.handle_keyboard_interrupt()
                except Exception as e:
                    print(f"Keyboard handling error: {e}")

            # Initialize keyboard listener with the local on_press function
            self.keyboard_listener = keyboard.Listener(on_press=on_press)
            self.keyboard_listener.start()

            # Initialize monitor stream as None
            self.monitor_stream = None

            # Initialize mic monitoring
            self.setup_mic_monitoring()

            # Add these new initializations
            self.current_audio_playing = False
            self.is_processing = False
            self.is_speaking = False
            self.pending_response = None

            self.error_count = 0
            self.max_errors = 3
            self.last_error_time = time.time()
            self.error_cooldown = 5  # seconds
            self.recovery_lock = threading.Lock()
            self.watchdog_active = True
            self.last_activity_time = time.time()
            self.activity_timeout = 300  # Changed from 30 to 300 seconds
            
            # Initialize handlers
            self.cleanup_handler = CleanupHandler(self)
            self.watchdog = WatchdogMonitor(self)
            self.interrupt_handler = InterruptionHandler()
            
            # Start watchdog monitoring
            self.watchdog.start_monitoring()

            # Initialize AudioProcessor
            self.processor = AudioProcessor(self)

        except Exception as e:
            print(f"Error during initialization: {e}")
            raise

    def setup_mic_monitoring(self):
        """Setup real-time mic monitoring with volume control and noise gate."""
        try:
            def audio_callback(indata, outdata, frames, time, status):
                try:
                    # Only print status messages if it's not an underflow
                    if status and not status.input_underflow and not status.output_underflow:
                        print(f'Monitoring status: {status}')
                        
                    if self.monitoring_active and not self.is_speaking and not self.current_audio_playing:
                        # Convert to float32 if not already
                        audio_data = indata.copy()  # Create a copy to avoid modifying input
                        if audio_data.dtype != np.float32:
                            audio_data = audio_data.astype(np.float32)
                        
                        # Apply noise gate
                        mask = np.abs(audio_data) < self.noise_gate_threshold
                        audio_data[mask] = 0
                        
                        # Apply volume control
                        audio_data = audio_data * self.mic_volume
                        
                        # Ensure we don't exceed [-1, 1] range
                        audio_data = np.clip(audio_data, -1.0, 1.0)
                        
                        # Fill output buffer
                        outdata[:] = audio_data
                    else:
                        outdata.fill(0)
                        
                except Exception as e:
                    print(f"Error in audio callback: {e}")
                    outdata.fill(0)  # Ensure output is silent on error

            # Try to create the stream with more conservative settings
            self.monitor_stream = sd.Stream(
                channels=self.MONITOR_CHANNELS,
                dtype=self.MONITOR_DTYPE,
                samplerate=self.RATE,
                callback=audio_callback,
                blocksize=2048,
                latency='high',
                device=None,
                prime_output_buffers_using_stream_callback=True
            )
            
            # Start the stream in a try-except block
            try:
                self.monitor_stream.start()
            except sd.PortAudioError as e:
                print(f"Error starting audio stream: {e}")
                self.monitor_stream = None
                
        except Exception as e:
            print(f"Error setting up mic monitoring: {e}")
            self.monitor_stream = None

    def set_mic_volume(self, volume):
        """Set the microphone monitoring volume (0.0 to 1.0)."""
        self.mic_volume = max(0.0, min(1.0, volume))
        print(f"Mic monitoring volume set to: {self.mic_volume:.2f}")

    def set_noise_gate(self, threshold):
        """Set the noise gate threshold (0.0 to 1.0)."""
        self.noise_gate_threshold = max(0.0, min(1.0, threshold))
        print(f"Noise gate threshold set to: {self.noise_gate_threshold:.3f}")

    def get_ai_response(self, text):
        """Wrapper for AI handler method"""
        return self.aiHandler.get_ai_response(text)

    def handle_keyboard_interrupt(self):
        """Delegate to InterruptionHandler"""
        self.interrupt_handler.handle_keyboard_interrupt(self)

    def handle_interrupt(self, interrupt_type):
        """Delegate to InterruptionHandler"""
        self.interrupt_handler.handle_interrupt(self, interrupt_type)

    def play_acknowledgment(self):
        """Delegate to InterruptionHandler"""
        self.interrupt_handler.play_acknowledgment()

    def process_audio_stream(self):
        """Delegate to AudioProcessor"""
        self.processor.process_audio_stream()

def main():
    try:
        transcriber = AudioTranscriber()
        print("\nStarting audio transcription... Speak into your microphone.")
        print("Press ` (backtick) to interrupt at any time.")
        transcriber.process_audio_stream()
    except KeyboardInterrupt:
        print("\nTranscription stopped by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()













