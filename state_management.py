from enum import Enum
import time
import pygame
import queue
from pathlib import Path
from interruption import ListeningState

class AudioState:
    def __init__(self):
        # Core state flags
        self.is_speaking = False
        self.is_processing = False
        self.stop_generation = False
        self.current_audio_playing = False
        self.monitoring_active = True
        self.watchdog_active = True

        # Transcript tracking
        self.current_sentence = ""
        self.last_transcript = ""
        self.last_final_transcript = ""
        self.last_sentence_complete = False

        # Timing variables
        self.last_speech_time = time.time()
        self.last_activity_time = time.time()
        self.last_interim_timestamp = time.time()
        self.last_interrupt_time = time.time()
        self._last_transcript_time = time.time()
        self._last_processing_time = time.time()
        self._last_state = None

        # State configurations
        self.listening_state = ListeningState.FULL_LISTENING
        self.pending_response = None

        # Error handling
        self.error_count = 0
        self.max_errors = 3
        self.last_error_time = time.time()
        self.error_cooldown = 5
        self.activity_timeout = 300

    def reset_state(self, force=False):
        """Reset all state variables to their default values."""
        try:
            self.is_speaking = False
            self.is_processing = False
            self.stop_generation = False
            self.current_sentence = ""
            self.last_transcript = ""
            self.last_final_transcript = ""
            self.last_sentence_complete = False
            self.last_interim_timestamp = time.time()
            self._last_state = None
            
            # Ensure pygame mixer is in a clean state
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.music.unload()
                    pygame.mixer.quit()
            except:
                pass
            
            print("\nListening... (Press ` to interrupt)")
        except Exception as e:
            print(f"Error resetting state: {e}")

    def aggressive_cleanup(self, audio_queue):
        """Perform aggressive cleanup of all state variables and queues."""
        # Reset all state variables
        self.current_sentence = ""
        self.last_transcript = ""
        self.last_final_transcript = ""
        self.last_sentence_complete = False
        self.last_interim_timestamp = time.time()
        
        # Clear audio queue
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Reset all timing variables
        self._last_transcript_time = time.time()
        self._last_processing_time = time.time()
        self._last_state = None

    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity_time = time.time()

    def should_process_transcript(self, is_complete, silence_duration):
        """Determine if a transcript should be processed based on completion and silence."""
        if self.is_speaking or self.is_processing:
            return False

        if self.listening_state != ListeningState.FULL_LISTENING:
            return False

        return (
            (is_complete and silence_duration >= 0.5) or
            (not is_complete and silence_duration >= 1.0)
        )

    def can_interrupt(self, current_time):
        """Check if interruption is allowed based on cooldown."""
        return (current_time - self.last_interrupt_time) >= self.interrupt_cooldown

    def handle_watchdog_timeout(self, current_time, audio_queue):
        """Check if watchdog should trigger recovery."""
        return (
            current_time - self.last_activity_time > self.activity_timeout and 
            not self.is_processing and 
            not self.is_speaking and
            not audio_queue.empty()
        )

    def prepare_for_processing(self):
        """Set appropriate states before processing begins."""
        self.is_processing = True
        self.listening_state = ListeningState.INTERRUPT_ONLY
        self._last_processing_time = time.time()

    def finish_processing(self):
        """Reset states after processing completes."""
        self.is_processing = False
        self.is_speaking = False
        self.stop_generation = False
        self.listening_state = ListeningState.FULL_LISTENING
        self.current_audio_playing = False
        self.pending_response = None 