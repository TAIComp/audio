import time
import threading
import pygame
from pathlib import Path

class WatchdogMonitor:
    def __init__(self, audio_instance):
        self.audio_instance = audio_instance
        self.watchdog_active = True
        self.activity_timeout = 300  # 5 minutes
        self.recovery_lock = threading.Lock()
        
        # Error tracking
        self.error_count = 0
        self.max_errors = 3
        self.last_error_time = time.time()
        self.error_cooldown = 5  # seconds

    def start_monitoring(self):
        """Start the watchdog monitoring thread."""
        self.watchdog_thread = threading.Thread(target=self._watchdog_monitor, daemon=True)
        self.watchdog_thread.start()

    def _watchdog_monitor(self):
        """Monitor system health and recover from deadlocks."""
        while self.watchdog_active:
            try:
                time.sleep(1)
                current_time = time.time()
                
                # Only trigger recovery if we're truly stuck
                if (current_time - self.audio_instance.state.last_activity_time > self.activity_timeout and 
                    not self.audio_instance.state.is_processing and 
                    not self.audio_instance.state.is_speaking and
                    not self.audio_instance.audio_queue.empty()):
                    
                    print("\nWatchdog: System appears frozen, initiating recovery...")
                    self.emergency_recovery()
                    # Add a cooldown after recovery
                    time.sleep(5)
                    self.audio_instance.state.last_activity_time = time.time()
                    
            except Exception as e:
                print(f"Error in watchdog: {e}")
                time.sleep(1)

    def emergency_recovery(self):
        """Emergency recovery procedure."""
        try:
            with self.recovery_lock:
                print("\nInitiating emergency recovery...")
                
                # Force stop all processing
                self.audio_instance.stop_generation = True
                self.audio_instance.state.is_processing = False
                self.audio_instance.state.is_speaking = False
                
                # Clean up audio resources
                self.audio_instance.cleanup_handler.aggressive_cleanup()
                
                # Reinitialize system
                time.sleep(1)
                self.reinitialize_audio_system()
                
                # Reset states
                self.audio_instance.reset_state(force=True)
                
                # Update activity timestamp
                self.audio_instance.state.last_activity_time = time.time()
                
                print("\nEmergency recovery completed")
                
        except Exception as e:
            print(f"Error in emergency recovery: {e}")
            # Force reset everything
            self.audio_instance.__init__()

    def reinitialize_audio_system(self):
        """Reinitialize audio system after recovery."""
        try:
            # Reinitialize pygame mixer
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            pygame.mixer.init()
            
            # Reinitialize mic monitoring
            self.audio_instance.setup_mic_monitoring()
            
            # Reset all flags
            self.audio_instance.state.is_speaking = False
            self.audio_instance.state.is_processing = False
            self.audio_instance.stop_generation = False
            self.audio_instance.state.current_audio_playing = False
            
            print("\nAudio system reinitialized successfully")
        except Exception as e:
            print(f"Error reinitializing audio system: {e}")

    def stop(self):
        """Stop the watchdog monitor."""
        self.watchdog_active = False
        if hasattr(self, 'watchdog_thread'):
            self.watchdog_thread.join(timeout=1.0) 