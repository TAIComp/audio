from enum import Enum
from typing import Optional

class ListeningState(Enum):
    FULL_LISTENING = "full_listening"
    INTERRUPT_ONLY = "interrupt_only"

class StateManager:
    def __init__(self):
        self.current_state = ListeningState.FULL_LISTENING
        self.current_text = ""  # Holds text during FULL_LISTENING
        self.interrupt_text = ""  # Holds text during INTERRUPT_ONLY
        self.is_processing = False
        self.is_speaking = False
        
        # Add callback registry
        self._on_state_change_callbacks = []
        
    def register_state_change_callback(self, callback):
        """Register a callback to be notified of state changes."""
        self._on_state_change_callbacks.append(callback)
        
    def transition_to(self, new_state: ListeningState) -> None:
        """Transition to a new state and notify callbacks."""
        if new_state == self.current_state:
            return
            
        old_state = self.current_state
        self.current_state = new_state
        
        # Clear text buffers when transitioning states
        self.reset_text()
        
        # Notify callbacks of state change
        for callback in self._on_state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                print(f"Error in state change callback: {e}")

    def update_text(self, text: str) -> None:
        """Update the appropriate text buffer based on current state."""
        if self.current_state == ListeningState.FULL_LISTENING:
            self.current_text = text
            print(f"\rCurrent speech: {text}", end="", flush=True)
        else:  # INTERRUPT_ONLY
            self.interrupt_text = text
            print(f"\rMonitoring for interrupts: {text}", end="", flush=True)

    def get_current_text(self) -> str:
        """Get the appropriate text based on current state."""
        return self.current_text if self.current_state == ListeningState.FULL_LISTENING else self.interrupt_text

    def reset(self) -> None:
        """Reset to initial state and clear all buffers."""
        self.current_state = ListeningState.FULL_LISTENING
        self.current_text = ""
        self.interrupt_text = ""
        self.is_processing = False
        self.is_speaking = False 

    def reset_text(self) -> None:
        """Reset both text buffers."""
        self.current_text = ""
        self.interrupt_text = "" 