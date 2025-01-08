from pathlib import Path
import pygame
import time
from typing import Optional, Callable

class InterruptionHandler:
    def __init__(self, interrupt_audio_path: Path = Path("interruption.mp3")):
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
        self.interrupt_audio_path = interrupt_audio_path
        self.last_interrupt_time = 0
        self.interrupt_cooldown = 1.0
        self.keyboard_interrupt_key = '`'
        self._initialize_audio()

    def _initialize_audio(self) -> None:
        try:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            pygame.mixer.init(frequency=44100)
        except Exception as e:
            print(f"Error initializing pygame mixer: {e}")

    def handle_keyboard_interrupt(self, key: str) -> bool:
        """Handle keyboard interruption event."""
        if key == self.keyboard_interrupt_key:
            return self.handle_interrupt()
        return False

    def handle_interrupt(self) -> bool:
        """Handle an interruption event."""
        try:
            current_time = time.time()
            if current_time - self.last_interrupt_time >= self.interrupt_cooldown:
                if self.interrupt_audio_path.exists():
                    pygame.mixer.music.load(str(self.interrupt_audio_path))
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    pygame.mixer.music.unload()
                
                self.last_interrupt_time = current_time
                return True
            return False
            
        except Exception as e:
            print(f"Error handling interrupt: {e}")
            return False

    def is_interrupt_command(self, text: str) -> bool:
        """Check if the given text contains an interrupt command."""
        text = text.lower().strip()
        # Check for exact matches
        if text in self.interrupt_commands:
            return True
        # Check for partial matches (if command appears anywhere in the text)
        return any(cmd in text for cmd in self.interrupt_commands)

    def cleanup(self) -> None:
        """Clean up pygame mixer resources."""
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                pygame.mixer.quit()
        except Exception as e:
            print(f"Error during interrupt cleanup: {e}")