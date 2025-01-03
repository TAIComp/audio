import os
import pygame
import warnings
from ctypes import *

def initialize_audio():
    """Initialize audio system and suppress warnings."""
    # Hide Pygame support prompt
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    
    # Suppress ALSA warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="pygame.mixer")
    
    # Redirect ALSA errors to /dev/null
    try:
        ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        def py_error_handler(filename, line, function, err, fmt):
            return  # Just return instead of pass

        c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        
        try:
            asound = cdll.LoadLibrary('libasound.so.2')
            asound.snd_lib_error_set_handler(c_error_handler)
        except OSError:
            print("Warning: Could not load ALSA library. Audio might not work properly.")
                
    except Exception as e:
        print(f"Warning: Could not set ALSA error handler: {e}")

    # Initialize pygame mixer with more robust error handling
    try:
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        
        pygame.mixer.init(
            frequency=16000,  # Standard frequency
            size=-16,         # 16-bit signed
            channels=1,       # Mono
            buffer=2048       # Larger buffer for stability
        )
        
        # Verify initialization
        if not pygame.mixer.get_init():
            raise Exception("Mixer initialization failed verification")
            
    except Exception as e:
        print(f"Warning: Could not initialize pygame mixer: {e}")
        # Try alternative initialization
        try:
            pygame.mixer.init(
                frequency=44100,  # Try standard frequency
                size=-16,
                channels=2,
                buffer=4096
            )
        except Exception as e:
            print(f"Critical: Alternative mixer initialization failed: {e}") 