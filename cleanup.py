import time
import pygame
import queue
from pathlib import Path
from google.cloud import speech

class CleanupHandler:
    def __init__(self, audio_instance):
        self.audio_instance = audio_instance

    def aggressive_cleanup(self):
        """Enhanced aggressive cleanup."""
        try:
            # Reset transcription-related variables
            self.audio_instance.current_sentence = ""
            self.audio_instance.last_transcript = ""
            self.audio_instance.last_final_transcript = ""
            self.audio_instance.last_sentence_complete = False
            self.audio_instance.last_interim_timestamp = time.time()
            self.audio_instance.current_interrupt_buffer = ""
            self.audio_instance.last_interrupt_buffer_update = time.time()
            
            # Clear audio queue
            while not self.audio_instance.audio_queue.empty():
                try:
                    self.audio_instance.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Force reset recognition state
            if hasattr(self.audio_instance, 'client'):
                try:
                    # Import speech at the top of the file
                    from google.cloud import speech
                    
                    # Create new streaming config using the speech module
                    self.audio_instance.streaming_config = speech.StreamingRecognitionConfig(
                        config=self.audio_instance.config,
                        interim_results=True
                    )
                except Exception as e:
                    print(f"Error resetting recognition state: {e}")
            
            # Reset all timing variables
            self.audio_instance._last_transcript_time = time.time()
            self.audio_instance._last_processing_time = time.time()
            self.audio_instance._last_state = None

        except Exception as e:
            print(f"Error in aggressive cleanup: {e}")

    def cleanup_resources(self):
        """Cleanup resources before shutdown."""
        try:
            # Stop all audio playback
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                pygame.mixer.quit()
            
            # Clean up monitor stream
            if hasattr(self.audio_instance, 'monitor_stream') and self.audio_instance.monitor_stream is not None:
                self.audio_instance.monitor_stream.stop()
                self.audio_instance.monitor_stream.close()
            
            # Clean up temporary files
            self.cleanup_temp_files()
            
        except Exception as e:
            print(f"Cleanup error: {e}")

    def cleanup_temp_files(self):
        """Clean up temporary audio files."""
        temp_dir = Path("temp")
        if temp_dir.exists():
            for session_dir in temp_dir.glob("audio_*"):
                try:
                    for file in session_dir.glob("*"):
                        try:
                            if file.exists():
                                file.unlink()
                        except Exception:
                            pass
                    try:
                        session_dir.rmdir()
                    except Exception:
                        pass
                except Exception:
                    pass 