import sounddevice as sd
import numpy as np
import time
from pathlib import Path
from dotenv import load_dotenv
from stt_handler import SpeechToTextHandler, STTConfig
from tts_handler import TextToSpeechHandler, TTSConfig
from ai_handler import create_default_service, ChatConfig, AnalysisConfig
from state_manager import StateManager, ListeningState
from interruption import InterruptionHandler
import os
import threading
from pynput import keyboard as kb

def main():
    # Initialize configurations and handlers
    load_dotenv()
    
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

    # Initialize handlers and managers
    state_manager = StateManager()
    interruption_handler = InterruptionHandler()
    
    stt_config = STTConfig(
        language_code="en-US",
        enable_interim=True,
        enable_punctuation=True
    )
    
    tts_config = TTSConfig(
        language_code="en-US",
        voice_name="en-US-Casual-K",
        speaking_rate=1.0
    )
    
    chat_config = ChatConfig(
        model_name="gpt-4",
        temperature=0.7
    )
    
    analysis_config = AnalysisConfig(
        model_name="gpt-4",
        temperature=0
    )

    stt_handler = SpeechToTextHandler(stt_config)
    tts_handler = TextToSpeechHandler(tts_config)
    ai_service = create_default_service()

    # Add these variables at the start of main()
    processing_timer = None
    accumulated_text = ""
    
    def process_after_delay(text: str, delay: float):
        """Process text after specified delay, unless new text arrives"""
        nonlocal processing_timer, accumulated_text
        
        # Cancel any existing timer
        if processing_timer:
            processing_timer.cancel()
        
        # Update accumulated text
        accumulated_text = text
        
        def delayed_process():
            nonlocal processing_timer, accumulated_text
            state_manager.transition_to(ListeningState.INTERRUPT_ONLY)
            # Process the accumulated text
            threading.Thread(target=process_ai_response, args=(accumulated_text,), daemon=True).start()
            processing_timer = None
            accumulated_text = ""
            
        # Start new timer
        processing_timer = threading.Timer(delay, delayed_process)
        processing_timer.start()

    def process_ai_response(text: str):
        """Process AI response and handle TTS."""
        nonlocal state_manager, ai_service, chat_config, tts_handler, stt_handler
        try:
            # Get AI response
            response_stream = ai_service.generate_chat_response(text, chat_config)
            
            # Convert response to speech and wait for completion
            tts_handler.stream_text_to_speech(response_stream)
            
            # Wait until TTS is no longer playing
            while tts_handler.is_playing:
                time.sleep(0.1)
            
            print("\nAudio response complete")
            
            # Clear STT state before state transition
            stt_handler.clear_state()
            state_manager.reset_text()
            state_manager.transition_to(ListeningState.FULL_LISTENING)
            
        except Exception as e:
            print(f"Error processing AI response: {e}")
            stt_handler.clear_state()
            state_manager.reset_text()
            state_manager.transition_to(ListeningState.FULL_LISTENING)

    def handle_state_change(old_state: ListeningState, new_state: ListeningState):
        """Handle state transitions."""
        if new_state == ListeningState.FULL_LISTENING:
            print("\nListening... (Press ` or say \"shut up\" to interrupt)")
        elif new_state == ListeningState.INTERRUPT_ONLY:
            print("\nNow in INTERRUPT_ONLY mode - monitoring for interrupt commands")

    def handle_interrupt_and_clearing():
        """Handle interruption by stopping playback, clearing queues, and resetting state."""
        # First stop and clear TTS audio
        tts_handler.stop_playback()
        
        # Play interrupt acknowledgment
        interruption_handler.handle_interrupt()
        
        # Clear STT buffers and reset recognition
        stt_handler.clear_state()
        
        # Reset state manager
        state_manager.reset_text()
        
        # Ensure state transition happens after cleanup
        state_manager.transition_to(ListeningState.FULL_LISTENING)
        
        # Brief pause to ensure everything is reset
        time.sleep(0.1)
        
        print("\nReady for new input...")

    def handle_partial(text: str):
        """Handle partial transcripts based on current state."""
        state_manager.update_text(text)
        
        if state_manager.current_state == ListeningState.INTERRUPT_ONLY:
            current_text = state_manager.get_current_text()
            if interruption_handler.is_interrupt_command(current_text.lower().strip()):
                print("\nInterrupt command detected!")
                handle_interrupt_and_clearing()

    def handle_final(text: str):
        """Handle final transcripts based on current state."""
        nonlocal processing_timer, accumulated_text
        
        # Skip empty transcripts
        if not text.strip():
            return
        
        if state_manager.current_state == ListeningState.FULL_LISTENING:
            # If we're still waiting on a timer, update the accumulated text
            if processing_timer:
                state_manager.update_text(text)
                print(f"\nUpdating transcript: {text}")
                process_after_delay(text, 2.0)  # Always use 2-second delay for updates
                return
                
            state_manager.update_text(text)
            print(f"\nFinal transcript: {state_manager.get_current_text()}")
            
            # Process the text through AI
            is_complete = ai_service.analyze_completion(text, analysis_config)
            
            if is_complete:
                print("\nSentence complete, processing in 1 second...")
                process_after_delay(text, 1.0)
            else:
                print("\nIncomplete sentence, processing in 2 seconds...")
                process_after_delay(text, 2.0)

    def on_press(key):
        """Handle keyboard press events."""
        try:
            if key.char == '`' and state_manager.current_state == ListeningState.INTERRUPT_ONLY:
                print("\nKeyboard interrupt detected!")
                handle_interrupt_and_clearing()
        except AttributeError:
            # Special key pressed, ignore
            pass

    def handle_keyboard_interrupt():
        """Handle keyboard interrupts in a separate thread."""
        with kb.Listener(on_press=on_press) as listener:
            listener.join()

    # Register callbacks
    state_manager.register_state_change_callback(handle_state_change)
    stt_handler.partial_transcript_callback = handle_partial
    stt_handler.final_transcript_callback = handle_final

    # Start keyboard interrupt handler in a separate thread
    keyboard_thread = threading.Thread(target=handle_keyboard_interrupt, daemon=True)
    keyboard_thread.start()

    # Start recognition
    stt_handler.start_recognition()

    def audio_callback(indata, frames, time, status):
        """Handle audio input data"""
        if status:
            print(f"Status: {status}")
        stt_handler.add_audio_data(indata.tobytes())

    try:
        print("\nListening... (Press ` or say \"shut up\" to interrupt)")
        # Update the stream parameters
        with sd.InputStream(callback=audio_callback,
                           channels=1,
                           samplerate=16000,
                           dtype=np.int16,
                           blocksize=4000,
                           device=None,
                           latency=0.1):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stt_handler.cleanup()
        tts_handler.cleanup()
        interruption_handler.cleanup()

if __name__ == "__main__":
    main()
