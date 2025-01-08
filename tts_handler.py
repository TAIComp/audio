from google.cloud import texttospeech
from typing import Optional, Iterator
from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import pygame
import os

@dataclass
class TTSConfig:
    """Configuration for text-to-speech processing"""
    language_code: str = "en-US"
    voice_name: str = "en-US-Casual-K"
    speaking_rate: float = 1.0
    output_dir: Path = Path("temp/tts_output")

class TextToSpeechHandler:
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self.client = texttospeech.TextToSpeechClient()
        self.audio_queue = queue.Queue()
        self.is_playing = False
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pygame mixer
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
        
        self._should_stop = False
        self.sentence_buffer = []
        self.is_first_sentence = True

    def stream_text_to_speech(self, text_stream: Iterator[str]) -> None:
        """Process streaming text input with immediate playback."""
        current_sentence = ""
        for text_chunk in text_stream:
            current_sentence += text_chunk
            sentences = self._split_into_sentences(current_sentence)
            
            if len(sentences) > 1:
                # Process complete sentences
                for sentence in sentences[:-1]:
                    if sentence.strip():
                        audio_path = self._generate_speech(sentence)
                        self.audio_queue.put(audio_path)
                        
                        # Start playback immediately for first sentence
                        if self.is_first_sentence:
                            self.is_first_sentence = False
                            self._start_playback()
                
                current_sentence = sentences[-1]

        # Process any remaining text
        if current_sentence.strip():
            audio_path = self._generate_speech(current_sentence)
            self.audio_queue.put(audio_path)
            if self.is_first_sentence:
                self._start_playback()

    def _generate_speech(self, text: str) -> Path:
        """Generate speech from text and return audio file path."""
        response = self.client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=texttospeech.VoiceSelectionParams(
                language_code=self.config.language_code,
                name=self.config.voice_name
            ),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=self.config.speaking_rate
            )
        )
        
        output_path = self.config.output_dir / f"tts_{os.urandom(8).hex()}.mp3"
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        return output_path

    def _playback_worker(self) -> None:
        """Handle audio playback in background thread."""
        while True:
            try:
                audio_path = self.audio_queue.get()
                if audio_path is None:
                    break
                
                self.is_playing = True
                pygame.mixer.music.load(str(audio_path))
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                self.is_playing = False
                audio_path.unlink()
                
            except Exception as e:
                print(f"Playback error: {e}")

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'([.!?])\s+', text)
        result = []
        
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i + 1] in '.!?':
                result.append(sentences[i] + sentences[i + 1])
                i += 2
            else:
                result.append(sentences[i])
                i += 1
        
        return result

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.audio_queue.put(None)
        if pygame.mixer.get_init():
            pygame.mixer.quit()

    def stop_playback(self) -> None:
        """Stop current audio playback and clear audio queue."""
        self._should_stop = True
        self.is_playing = False
        
        # Stop current playback
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        
        # Clear the audio queue
        while not self.audio_queue.empty():
            try:
                audio_path = self.audio_queue.get_nowait()
                if audio_path and audio_path.exists():
                    audio_path.unlink()  # Delete the audio file
            except queue.Empty:
                break

    def _start_playback(self) -> None:
        """Start the playback thread if not already running."""
        if not hasattr(self, 'playback_thread') or not self.playback_thread.is_alive():
            self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
            self.playback_thread.start()
