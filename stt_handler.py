from google.cloud import speech
from typing import Optional, Callable
import queue
import threading
from dataclasses import dataclass
import time

@dataclass
class STTConfig:
    """Configuration for speech-to-text processing"""
    language_code: str = "en-US"
    sample_rate: int = 16000
    enable_interim: bool = True
    enable_punctuation: bool = True

class SpeechToTextHandler:
    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or STTConfig()
        self.client = speech.SpeechClient()
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.should_stop = threading.Event()
        self.current_transcript = ""
        self.stream_reset_requested = False
        self.stream_active = True
        
        # Callbacks for handling transcripts
        self.partial_transcript_callback: Optional[Callable[[str], None]] = None
        self.final_transcript_callback: Optional[Callable[[str], None]] = None

    def start_recognition(self) -> None:
        """Start speech recognition in background thread."""
        if not self.is_listening:
            self.should_stop.clear()
            self.recognition_thread = threading.Thread(
                target=self._process_audio_stream,
                daemon=True
            )
            self.recognition_thread.start()
            self.is_listening = True

    def _process_audio_stream(self) -> None:
        """Process audio stream and emit transcripts."""
        while self.stream_active:
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.config.sample_rate,
                language_code=self.config.language_code,
                enable_automatic_punctuation=self.config.enable_punctuation,
                use_enhanced=True
            )
            
            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=self.config.enable_interim,
                single_utterance=False
            )

            def audio_generator():
                while not self.should_stop.is_set():
                    try:
                        chunk = self.audio_queue.get(timeout=0.1)
                        if chunk is None:
                            self.stream_reset_requested = True
                            return
                        if self.stream_reset_requested:
                            continue
                        yield speech.StreamingRecognizeRequest(audio_content=chunk)
                    except queue.Empty:
                        continue

            try:
                responses = self.client.streaming_recognize(streaming_config, audio_generator())
                
                for response in responses:
                    if self.should_stop.is_set() or self.stream_reset_requested:
                        break

                    if response.results:
                        result = response.results[0]
                        transcript = result.alternatives[0].transcript

                        if result.is_final:
                            if self.final_transcript_callback:
                                self.final_transcript_callback(transcript)
                        else:
                            if self.partial_transcript_callback:
                                self.partial_transcript_callback(transcript)
                                
            except Exception as e:
                print(f"Stream error, restarting: {e}")
                continue
            
            if self.stream_reset_requested:
                self.stream_reset_requested = False

    def add_audio_data(self, audio_chunk: bytes) -> None:
        """Add audio data to processing queue."""
        if self.is_listening:
            self.audio_queue.put(audio_chunk)

    def stop_recognition(self) -> None:
        """Stop speech recognition."""
        self.should_stop.set()
        self.audio_queue.put(None)


    def clear_state(self) -> None:
        """Clear internal state and audio queue."""
        # Clear the audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clear current transcript
        self.current_transcript = ""
        
        # Force stream reset
        self.stream_reset_requested = True
        self.audio_queue.put(None)
        
        # Wait for stream reset to complete
        time.sleep(0.1)  # Brief pause to ensure reset
        
        # Clear any remaining audio data
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def cleanup(self) -> None:
        """Cleanup STT resources."""
        self.stream_active = False
        self.should_stop.set()
        self.audio_queue.put(None)