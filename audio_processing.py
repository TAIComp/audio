import threading
import time
from datetime import datetime
import pygame
from pathlib import Path
import queue
import re
from interruption import ListeningState

class AudioProcessor:
    def __init__(self, audio_transcriber):
        self.transcriber = audio_transcriber

    def process_audio_stream(self):
        """Main audio processing loop."""
        try:
            # Use STT Handler for recognition
            stream, audio, responses = self.transcriber.stt_handler.start_recognition()
            
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
                        self.transcriber.state.update_activity()
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Error processing response: {e}")
                    time.sleep(0.1)
                    continue
        
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            self.transcriber.cleanup_handler.cleanup_resources()

    def handle_responses(self, responses):
        """Handle streaming responses with state-based processing."""
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            transcript = result.alternatives[0].transcript.lower().strip()
            current_time = time.time()
            
            # Update activity timestamp
            self.transcriber.state.update_activity()

            # Reset last_speech_time when new speech is detected
            if transcript:
                self.transcriber.last_speech_time = datetime.now()

            # Handle interrupt-only state
            if self.transcriber.listening_state == ListeningState.INTERRUPT_ONLY:
                self._handle_interrupt_state(transcript, current_time)
                continue

            # Skip if speaking or processing
            if self.transcriber.is_speaking or self.transcriber.is_processing:
                self.transcriber.cleanup_handler.aggressive_cleanup()
                return

            if result.is_final:
                self._handle_final_transcript(transcript, current_time)

    def process_transcript(self, transcript):
        """Process transcript with enhanced state management."""
        try:
            # Clear any existing audio and transcripts before processing
            self.transcriber.cleanup_handler.aggressive_cleanup()
            self.transcriber.last_final_transcript = ""  # Clear the last transcript
            self.transcriber.current_sentence = ""       # Clear current sentence
            
            # Verify we should still process
            if self.transcriber.is_speaking or self.transcriber.stop_generation:
                return

            # Start AI response generation
            response_future = threading.Thread(
                target=self.process_complete_sentence,
                args=(transcript,),
                daemon=True
            )
            response_future.start()
            response_future.join()
            
        except Exception as e:
            print(f"Error processing transcript: {e}")
        finally:
            self.transcriber.is_processing = False
            self.transcriber.listening_state = ListeningState.FULL_LISTENING
            self.transcriber.cleanup_handler.aggressive_cleanup()
            self.transcriber.state.update_activity()
            
            # Clear all transcripts after processing
            self.transcriber.last_final_transcript = ""
            self.transcriber.current_sentence = ""
            self.transcriber.last_transcript = ""
            
            print("\nListening... (Press ` to interrupt)")

    def process_complete_sentence(self, sentence):
        """Process complete sentences and generate AI responses."""
        try:
            # Clear transcripts and set initial state
            self.transcriber.cleanup_handler.aggressive_cleanup()
            
            if self.transcriber.stop_generation:
                self.transcriber.state.reset_state()
                return

            # Create temp directory structure
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            session_dir = temp_dir / f"audio_{int(time.time() * 1000)}"
            session_dir.mkdir(exist_ok=True)

            # Set state
            self.transcriber.listening_state = ListeningState.INTERRUPT_ONLY
            print("\nChanging state to INTERRUPT_ONLY for processing")

            # Process speech with TTS handler
            audio_buffer = queue.Queue()
            self._handle_speech_processing(sentence, session_dir, audio_buffer)

        except Exception as e:
            print(f"Critical error in process_complete_sentence: {e}")
            self.transcriber.cleanup_handler.aggressive_cleanup()
            self.transcriber.state.reset_state()

    def check_silence(self):
        """Monitor for periods of silence and trigger processing."""
        while True:
            try:
                time.sleep(0.1)
                
                if self.transcriber.is_processing or self.transcriber.is_speaking:
                    continue
                    
                current_time = datetime.now()
                silence_duration = (current_time - self.transcriber.last_speech_time).total_seconds()
                
                if (self.transcriber.last_final_transcript and 
                    not self.transcriber.is_processing and 
                    silence_duration >= 1.0):
                    
                    is_complete = self.transcriber.aiHandler.is_sentence_complete(
                        self.transcriber.last_final_transcript
                    )
                    should_process = (
                        (is_complete and silence_duration >= 0.5) or
                        (not is_complete and silence_duration >= 1.0)
                    )
                    
                    if should_process:
                        print(f"\nSilence detected ({silence_duration:.1f}s). Processing transcript.")
                        self.transcriber.is_processing = True
                        
                        processing_thread = threading.Thread(
                            target=self.process_transcript,
                            args=(self.transcriber.last_final_transcript,),
                            daemon=True
                        )
                        processing_thread.start()
                        
                        self.transcriber.last_final_transcript = ""
                        
            except Exception as e:
                print(f"Error in silence check: {e}")
                time.sleep(1)

    def _handle_interrupt_state(self, transcript, current_time):
        """Handle interruption state processing."""
        if (current_time - self.transcriber.interrupt_handler.last_interrupt_time >= 
            self.transcriber.interrupt_handler.interrupt_cooldown):
            pattern = r'\b(?:' + '|'.join(
                re.escape(cmd) for cmd in self.transcriber.interrupt_handler.interrupt_commands
            ) + r')\b'
            if re.search(pattern, transcript):
                print(f"\nInterrupt command detected: '{transcript}'")
                self.transcriber.interrupt_handler.handle_interrupt(self.transcriber, "voice")
                self.transcriber.cleanup_handler.aggressive_cleanup()

    def _handle_final_transcript(self, transcript, current_time):
        """Handle final transcript processing."""
        if hasattr(self.transcriber, '_last_transcript_time'):
            if current_time - self.transcriber._last_transcript_time < 1.0:
                return

        # Check if this transcript is too similar to the last one
        if self.transcriber.last_final_transcript and \
           transcript.strip() in self.transcriber.last_final_transcript:
            return

        print(f'\nFinal: "{transcript}"')
        self.transcriber._last_transcript_time = current_time
        
        if (self.transcriber.listening_state == ListeningState.FULL_LISTENING and 
            not self.transcriber.is_processing):
            is_complete = self.transcriber.aiHandler.is_sentence_complete(transcript)
            silence_duration = (datetime.now() - self.transcriber.last_speech_time).total_seconds()
            
            should_process = (
                (is_complete and silence_duration >= 0.5) or
                (not is_complete and silence_duration >= 1.0)
            )
            
            if should_process:
                print(f"\nProcessing triggered - Sentence complete: {is_complete}, Silence: {silence_duration:.1f}s")
                self.transcriber.is_processing = True
                self.transcriber._last_processing_time = current_time
                self.transcriber.last_final_transcript = transcript
                
                processing_thread = threading.Thread(
                    target=self.process_transcript,
                    args=(transcript,),
                    daemon=True
                )
                processing_thread.start()

    def _handle_speech_processing(self, sentence, session_dir, audio_buffer):
        """Handle speech processing and playback."""
        playback_active = True
        is_playing = False

        def play_audio_chunks():
            nonlocal is_playing, playback_active
            try:
                while playback_active and not self.transcriber.stop_generation:
                    if not is_playing and not audio_buffer.empty():
                        chunk = audio_buffer.get_nowait()
                        if chunk and chunk.exists():
                            is_playing = True
                            pygame.mixer.music.load(str(chunk))
                            pygame.mixer.music.play()
                            while pygame.mixer.music.get_busy() and not self.transcriber.stop_generation:
                                pygame.time.Clock().tick(10)
                            pygame.mixer.music.unload()
                            chunk.unlink()
                            is_playing = False
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error in playback thread: {e}")
                is_playing = False

        playback_thread = threading.Thread(target=play_audio_chunks, daemon=True)
        playback_thread.start()

        try:
            response_received = self.transcriber.tts_handler.process_speech_chunks(
                self.transcriber.get_ai_response(sentence),
                session_dir,
                audio_buffer,
                lambda: self.transcriber.stop_generation
            )

            wait_start = time.time()
            while (not audio_buffer.empty() or is_playing) and time.time() - wait_start < 30:
                if self.transcriber.stop_generation:
                    break
                time.sleep(0.1)

        finally:
            playback_active = False
            try:
                playback_thread.join(timeout=2.0)
            except:
                pass
            self._cleanup_session(session_dir)

    def _cleanup_session(self, session_dir):
        """Clean up session directory and reset states."""
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

        self.transcriber.cleanup_handler.aggressive_cleanup()
        self.transcriber.is_speaking = False
        self.transcriber.is_processing = False
        self.transcriber.stop_generation = False
        self.transcriber.listening_state = ListeningState.FULL_LISTENING 