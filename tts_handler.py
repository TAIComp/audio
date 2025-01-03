from google.cloud import texttospeech
from pathlib import Path
import pygame
import threading
import queue
import time
import re

class TTSHandler:
    def __init__(self):
        # Initialize Text-to-Speech client
        self.tts_client = texttospeech.TextToSpeechClient()
        
        # Configure Text-to-Speech with a male voice
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Casual-K",  # Casual male voice
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
        
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,  # Normal speed
            pitch=0.0  # Normal pitch
        )

    def text_to_speech(self, text):
        """Convert text to speech and save as MP3."""
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config
            )
            
            # Create output directory if it doesn't exist
            output_dir = Path("ai_responses")
            output_dir.mkdir(exist_ok=True)
            
            # Use a fixed filename instead of timestamp
            output_path = output_dir / "latest_response.mp3"
            
            # Save the audio file (overwrites existing file)
            with open(output_path, "wb") as out:
                out.write(response.audio_content)
                
            return output_path
            
        except Exception as e:
            print(f"Text-to-Speech error: {e}")
            return None

    def play_audio_response(self, audio_path, state_manager):
        """Play audio response with state management."""
        try:
            state_manager.is_speaking = True
            state_manager.listening_state = "INTERRUPT_ONLY"
            
            # Clear any pending transcripts and audio data
            state_manager.aggressive_cleanup()
            
            # Temporarily disable mic monitoring during playback
            state_manager.monitoring_active = False
            
            pygame.mixer.music.load(str(audio_path))
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                if not state_manager.is_speaking:  # Check if interrupted
                    pygame.mixer.music.stop()
                    break
                    
        except Exception as e:
            print(f"Audio playback error: {e}")
        finally:
            # Cleanup and reset states
            state_manager.cleanup_after_playback()

    def process_speech_chunks(self, text_chunks, session_dir, audio_buffer, stop_flag):
        """Process text chunks into speech and queue them for playback."""
        current_sentence = ""
        chunk_counter = 0
        response_received = False
        
        for text_chunk in text_chunks:
            if stop_flag():  # Check if we should stop
                break
                
            if text_chunk:
                response_received = True
                current_sentence += text_chunk
                sentences = self._split_into_sentences(current_sentence)
                
                while len(sentences) >= 2:
                    sentence_text = sentences[0] + sentences[1]
                    self._process_single_sentence(sentence_text, session_dir, chunk_counter, audio_buffer)
                    chunk_counter += 1
                    sentences = sentences[2:]
                    current_sentence = ''.join(sentences)
        
        # Process any remaining text
        if current_sentence.strip():
            self._process_single_sentence(current_sentence, session_dir, chunk_counter, audio_buffer)
            
        return response_received

    def _split_into_sentences(self, text):
        """Split text into sentences."""
        return re.split(r'([.!?])\s*', text)

    def _process_single_sentence(self, sentence_text, session_dir, chunk_counter, audio_buffer):
        """Process a single sentence into speech."""
        try:
            synthesis_input = texttospeech.SynthesisInput(text=sentence_text)
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config
            )
            
            temp_path = session_dir / f"chunk_{chunk_counter}.mp3"
            with open(temp_path, "wb") as out:
                out.write(response.audio_content)
            
            audio_buffer.put(temp_path)
            
        except Exception as e:
            print(f"Error processing sentence: {e}") 