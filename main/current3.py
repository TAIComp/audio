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
from aiHandler import AIHandler
from interruption import InterruptionHandler, ListeningState
from audio_utils import initialize_audio
from state_management import AudioState
from watchdog import WatchdogMonitor
from cleanup import CleanupHandler
from tts_handler import TTSHandler
from stt_handler import STTHandler

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
        try:
            # Use STT Handler for recognition
            stream, audio, responses = self.stt_handler.start_recognition()
            
            # Start the silence checking thread
            silence_thread = threading.Thread(target=self.check_silence, daemon=True)
            silence_thread.start()

            print("Listening... (Press ` to interrupt)")

            # Process responses with error handling
            while True:
                try:
                    response = next(responses)
                    if response:
                        self.handle_responses([response])
                        self.last_activity_time = time.time()
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Error processing response: {e}")
                    time.sleep(0.1)
                    continue
        
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            try:
                stream.stop_stream()
                stream.close()
                audio.terminate()
            except:
                pass
            try:
                pygame.mixer.quit()
            except:
                pass
            try:
                self.keyboard_listener.stop()
            except:
                pass

    def handle_responses(self, responses):
        """Handle streaming responses with state-based processing and improved interrupt detection."""
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            transcript = result.alternatives[0].transcript.lower().strip()
            current_time = time.time()
            
            # Update activity timestamp using state manager
            self.state.update_activity()

            # Reset last_speech_time when new speech is detected
            if transcript:
                self.last_speech_time = datetime.now()
                # Also update activity time to prevent watchdog from triggering
                self.last_activity_time = time.time()

            # Check for interrupts during INTERRUPT_ONLY state first, before any other processing
            if self.listening_state == ListeningState.INTERRUPT_ONLY:
                if current_time - self.interrupt_handler.last_interrupt_time >= self.interrupt_handler.interrupt_cooldown:
                    # Use the interrupt_commands from the interrupt_handler
                    pattern = r'\b(?:' + '|'.join(re.escape(cmd) for cmd in self.interrupt_handler.interrupt_commands) + r')\b'
                    is_interrupt = re.search(pattern, transcript) is not None
                    
                    if is_interrupt:
                        print(f"\nInterrupt command detected: '{transcript}'")
                        self.interrupt_handler.handle_interrupt(self, "voice")
                        # Clear everything
                        self.cleanup_handler.aggressive_cleanup()
                        return
            
            # Skip processing and clear everything if we're speaking or processing
            if self.is_speaking or self.is_processing:
                # Force clear everything
                self.cleanup_handler.aggressive_cleanup()
                # Create new streaming request to force reset recognition
                self.streaming_config = speech.StreamingRecognitionConfig(
                    config=self.config,
                    interim_results=True
                )
                # Skip this response entirely
                return

            # Always verify state before processing
            if self.listening_state != ListeningState.FULL_LISTENING:
                self.cleanup_handler.aggressive_cleanup()
                return

            # Skip if we just finished processing
            if hasattr(self, '_last_processing_time'):
                if current_time - self._last_processing_time < 1.0:  # 1 second cooldown
                    self.cleanup_handler.aggressive_cleanup()
                    return

            if result.is_final:
                # Verify this isn't a stale response
                if hasattr(self, '_last_transcript_time'):
                    if current_time - self._last_transcript_time < 0.5:  # 500ms cooldown
                        return

                print(f'\nFinal: "{transcript}"')
                self._last_transcript_time = current_time
                
                # Process only if in correct state
                if self.listening_state == ListeningState.FULL_LISTENING and not self.is_processing:
                    is_complete = self.aiHandler.is_sentence_complete(transcript)
                    silence_duration = (datetime.now() - self.last_speech_time).total_seconds()
                    
                    should_process = (
                        (is_complete and silence_duration >= 0.5) or
                        (not is_complete and silence_duration >= 1.0)
                    )
                    
                    if should_process:
                        print(f"\nProcessing triggered - Sentence complete: {is_complete}, Silence: {silence_duration:.1f}s")
                        self.is_processing = True
                        self._last_processing_time = current_time
                        self.last_final_transcript = transcript
                        
                        # Process in a separate thread
                        processing_thread = threading.Thread(
                            target=self.process_transcript,
                            args=(transcript,),
                            daemon=True
                        )
                        processing_thread.start()

    def process_transcript(self, transcript):
        """Process transcript with enhanced state management."""
        try:
            # Verify we should still process
            if self.is_speaking or self.stop_generation:
                self.cleanup_handler.aggressive_cleanup()
                return

            # Start AI response generation
            response_future = threading.Thread(
                target=self.process_complete_sentence,
                args=(transcript,),
                daemon=True
            )
            response_future.start()
            response_future.join()  # Wait for processing to complete
            
        except Exception as e:
            print(f"Error processing transcript: {e}")
        finally:
            self.is_processing = False
            self.listening_state = ListeningState.FULL_LISTENING
            self.cleanup_handler.aggressive_cleanup()
            # Update activity using state manager
            self.state.update_activity()
            print("\nListening... (Press ` to interrupt)")

    def process_complete_sentence(self, sentence):
        try:
            # Clear transcripts and set initial state
            self.cleanup_handler.aggressive_cleanup()
            
            if self.stop_generation:
                self.state.reset_state()
                return

            # Create temp directory structure
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            session_dir = temp_dir / f"audio_{int(time.time() * 1000)}"
            session_dir.mkdir(exist_ok=True)

            # Set state with proper error handling
            try:
                self.listening_state = ListeningState.INTERRUPT_ONLY
                print("\nChanging state to INTERRUPT_ONLY for processing")
            except Exception as e:
                print(f"Error setting listening state: {e}")
                self.cleanup_handler.aggressive_cleanup()
                return

            # Initialize playback variables in protected scope
            audio_buffer = queue.Queue()
            is_playing = False
            playback_active = True

            def play_audio_chunks():
                nonlocal is_playing, playback_active
                try:
                    while playback_active and not self.stop_generation:
                        if not is_playing and not audio_buffer.empty():
                            try:
                                chunk = audio_buffer.get_nowait()
                                if chunk and chunk.exists():
                                    is_playing = True
                                    
                                    # Ensure pygame mixer is initialized
                                    if not pygame.mixer.get_init():
                                        pygame.mixer.init()
                                        
                                    pygame.mixer.music.load(str(chunk))
                                    pygame.mixer.music.play()
                                    
                                    while pygame.mixer.music.get_busy() and not self.stop_generation:
                                        pygame.time.Clock().tick(10)
                                    
                                    pygame.mixer.music.unload()
                                    chunk.unlink()
                                    is_playing = False
                            except Exception as e:
                                print(f"Error in chunk playback: {e}")
                                is_playing = False
                        time.sleep(0.1)
                except Exception as e:
                    print(f"Error in playback thread: {e}")
                finally:
                    is_playing = False
                    try:
                        if pygame.mixer.get_init():
                            pygame.mixer.music.unload()
                    except:
                        pass

            # Start playback thread with error handling
            playback_thread = threading.Thread(target=play_audio_chunks, daemon=True)
            playback_thread.start()

            try:
                # Get AI response and process speech with proper error handling
                response_received = self.tts_handler.process_speech_chunks(
                    self.get_ai_response(sentence),
                    session_dir,
                    audio_buffer,
                    lambda: self.stop_generation
                )

                # Wait for audio completion with timeout
                wait_start = time.time()
                while (not audio_buffer.empty() or is_playing) and time.time() - wait_start < 30:
                    if self.stop_generation:
                        break
                    time.sleep(0.1)

            except Exception as e:
                print(f"Error in speech processing: {e}")
                response_received = False
            finally:
                # Cleanup with proper error handling
                playback_active = False
                try:
                    playback_thread.join(timeout=2.0)
                except:
                    pass

                # Cleanup session directory
                try:
                    for file in session_dir.glob("*"):
                        try:
                            if file.exists():
                                file.unlink()
                        except:
                            pass
                    session_dir.rmdir()
                except:
                    pass

                # Reset states
                self.cleanup_handler.aggressive_cleanup()
                self.is_speaking = False
                self.is_processing = False
                self.stop_generation = False
                self.listening_state = ListeningState.FULL_LISTENING

                # Update activity timestamp
                self.state.update_activity()
                print("\nListening... (Press ` to interrupt)")

        except Exception as e:
            print(f"Critical error in process_complete_sentence: {e}")
            # Emergency cleanup
            self.cleanup_handler.aggressive_cleanup()
            self.state.reset_state()

    def generate_response(self, sentence):
        """Generate AI response in a separate thread."""
        try:
            print("\nGenerating response...")
            response = self.get_ai_response(sentence)
            if response:
                print(f"\nAI Response: {response}")
                self.pending_response = response
        except Exception as e:
            print(f"Error generating response: {e}")
            self.pending_response = None

    def check_silence(self):
        """Monitor for periods of silence and trigger processing when appropriate."""
        while True:
            try:
                # Sleep briefly to prevent excessive CPU usage
                time.sleep(0.1)
                
                # Skip if we're already processing or speaking
                if self.is_processing or self.is_speaking:
                    continue
                    
                current_time = datetime.now()
                silence_duration = (current_time - self.last_speech_time).total_seconds()
                
                # Check if we have a pending transcript and enough silence
                if (self.last_final_transcript and 
                    not self.is_processing and 
                    silence_duration >= 1.0):  # 1 second of silence
                    
                    # Check if sentence appears complete or if we've waited long enough
                    is_complete = self.aiHandler.is_sentence_complete(self.last_final_transcript)
                    should_process = (
                        (is_complete and silence_duration >= 0.5) or  # Process complete sentences faster
                        (not is_complete and silence_duration >= 1.0)  # Wait longer for incomplete ones
                    )
                    
                    if should_process:
                        print(f"\nSilence detected ({silence_duration:.1f}s). Processing transcript.")
                        self.is_processing = True
                        
                        # Process in a separate thread
                        processing_thread = threading.Thread(
                            target=self.process_transcript,
                            args=(self.last_final_transcript,),
                            daemon=True
                        )
                        processing_thread.start()
                        
                        # Clear the transcript after processing starts
                        self.last_final_transcript = ""
                        
            except Exception as e:
                print(f"Error in silence check: {e}")
                time.sleep(1)  # Add delay on error

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












