"""
Test script for TTS installation
"""

import sys
import traceback

print(f"Python version: {sys.version}")

# Test TTS import
try:
    from TTS.api import TTS
    print("TTS package imported successfully")
except Exception as e:
    print(f"Error importing TTS: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test soundfile import
try:
    import soundfile as sf
    print("soundfile package imported successfully")
except Exception as e:
    print(f"Error importing soundfile: {e}")
    traceback.print_exc()

# Test numpy import
try:
    import numpy as np
    print("numpy package imported successfully")
except Exception as e:
    print(f"Error importing numpy: {e}")
    traceback.print_exc()

# Test langdetect import
try:
    from langdetect import detect
    print("langdetect package imported successfully")
except Exception as e:
    print(f"Error importing langdetect: {e}")
    traceback.print_exc()

# List available models
try:
    print("\nListing available models...")
    models = TTS().list_models()
    print(f"Available models: {len(models)}")
    for i, model in enumerate(models[:5]):
        print(f"  {i+1}. {model}")
    if len(models) > 5:
        print(f"  ... and {len(models) - 5} more")
except Exception as e:
    print(f"Error listing models: {e}")
    traceback.print_exc()

# Try to load a model
try:
    print("\nTrying to load a model...")
    model_name = "tts_models/en/ljspeech/tacotron2-DDC"
    print(f"Loading model: {model_name}")
    tts = TTS(model_name=model_name, progress_bar=True)
    print("Model loaded successfully")
    
    # Try to generate speech
    print("\nTrying to generate speech...")
    text = "Hello, this is a test of the TARS text-to-speech system."
    print(f"Text: {text}")
    wav = tts.tts(text)
    print("Speech generated successfully")
    
    # Try to save to file
    try:
        import io
        import soundfile as sf
        print("\nTrying to save to file...")
        sf.write("test_tts_output.wav", wav, 22050, format='WAV')
        print("Speech saved to test_tts_output.wav")
    except Exception as e:
        print(f"Error saving to file: {e}")
        traceback.print_exc()
        
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

print("\nTTS test completed.")
