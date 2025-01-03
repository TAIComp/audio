import pygame
import time
from pathlib import Path
import queue
from enum import Enum

class ListeningState(Enum):
    FULL_LISTENING = "full_listening"
    INTERRUPT_ONLY = "interrupt_only"

class InterruptionHandler:
    def __init__(self):
        self.interrupt_commands = {
            "stop", "end", "shut up",
            "please stop", "stop please",
            "please end", "end please",
            "shut up please", "please shut up",
            "okay stop", "ok stop",
            "can you stop", "could you stop",
            "would you stop", "can you be quiet",
            "silence", "pause"
        }
        self.last_interrupt_time = time.time()
        self.interrupt_cooldown = 1.0  # 1 second cooldown between interrupts
        self.interrupt_audio_path = Path("interruption.mp3")
        
        if not self.interrupt_audio_path.exists():
            print(f"Warning: Interrupt audio file '{self.interrupt_audio_path}' not found!")

    def handle_keyboard_interrupt(self, audio_instance):
        """Handle keyboard interruption."""
        current_time = time.time()
        if current_time - self.last_interrupt_time >= self.interrupt_cooldown:
            print("\nBacktick interrupt detected!")
            self.handle_interrupt(audio_instance, "keyboard")
            # Clear audio queue and force a small delay
            audio_instance.audio_queue.queue.clear()
            time.sleep(0.2)  # Small delay to ensure clean state

    def handle_interrupt(self, audio_instance, interrupt_type):
        """Common interrupt handling logic."""
        try:
            current_time = time.time()
            if current_time - self.last_interrupt_time >= self.interrupt_cooldown:
                # Stop any ongoing audio playback
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    
                # Use cleanup_handler instead of direct aggressive_cleanup
                audio_instance.cleanup_handler.aggressive_cleanup()
                
                # Clear audio queue before playing acknowledgment
                while not audio_instance.audio_queue.empty():
                    try:
                        audio_instance.audio_queue.get_nowait()
                    except queue.Empty:
                        break
                    
                self.play_acknowledgment()
                
                # Use cleanup_handler again
                audio_instance.cleanup_handler.aggressive_cleanup()
                
                # Add small delay and clear queue again after acknowledgment
                time.sleep(0.3)
                while not audio_instance.audio_queue.empty():
                    try:
                        audio_instance.audio_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Signal to stop AI response generation
                audio_instance.stop_generation = True
                
                # Reset all states
                audio_instance.is_speaking = False
                audio_instance.current_audio_playing = False
                audio_instance.is_processing = False
                audio_instance.pending_response = None
                audio_instance.listening_state = ListeningState.FULL_LISTENING
                
                audio_instance.state.reset_state()
                self.last_interrupt_time = current_time
                
                # Final cleanup using cleanup_handler
                audio_instance.cleanup_handler.aggressive_cleanup()
                print("\nListening... (Press ` to interrupt)")
                
        except Exception as e:
            print(f"Error during interrupt handling: {e}")

    def play_acknowledgment(self):
        """Play the prerecorded acknowledgment audio."""
        try:
            # Stop any currently playing audio
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            
            # Verify file exists
            if not self.interrupt_audio_path.exists():
                print(f"Error: Interruption audio file not found at {self.interrupt_audio_path}")
                return
            
            # Load and play the interruption audio
            pygame.mixer.music.load(str(self.interrupt_audio_path))
            pygame.mixer.music.play()
            
            # Wait for the interruption audio to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Cleanup
            pygame.mixer.music.unload()
            
        except Exception as e:
            print(f"Error playing acknowledgment: {e}") 