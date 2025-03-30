# TARS Speech System

TARS includes a powerful text-to-speech (TTS) system that allows it to speak text in multiple languages and voices. This feature enables TARS to provide auditory feedback, narrate its reasoning process, and interact with users through speech.

## Overview

The TARS Speech System provides:

1. **Multi-language support**: Speak text in English, French, Spanish, German, Italian, Dutch, Russian, and more
2. **Multiple voice models**: Choose from different voice models for each language
3. **Voice cloning**: Clone voices using reference audio samples
4. **DSL integration**: Use speech directly in TARS DSL scripts
5. **Cross-platform audio playback**: Works on Windows, Linux, and macOS

## Usage

### Command Line

You can use the TARS Speech System from the command line:

```bash
# Speak text in the default voice
tarscli speech speak --text "Hello, I am TARS"

# Speak text in a specific language
tarscli speech speak --text "Bonjour, je suis TARS" --language fr

# Use a specific voice model
tarscli speech speak --text "Hello" --voice "tts_models/en/ljspeech/tacotron2-DDC"

# Clone a voice using a reference audio file
tarscli speech speak --text "Hello" --speaker-wav reference.wav

# List available voice models
tarscli speech list-voices

# Configure speech settings
tarscli speech configure --default-voice "tts_models/en/ljspeech/tacotron2-DDC" --default-language en
```

### TARS DSL Integration

The TARS Speech System can be integrated into TARS DSL scripts:

```tars
speech_module {
  enabled true
  default_voice "tts_models/en/ljspeech/tacotron2-DDC"
  language "en"
  preload_voices true
  max_concurrency 1
  log_transcripts true
}

# Simple speak command
speak "Hello, I am TARS"

# Extended speak command with options
speak_extended {
  text "Bonjour, je suis TARS"
  language "fr"
  voice "tts_models/fr/mai/tacotron2-DDC"
  agentId "FrenchAgent"
}

# Speak multiple texts with different settings
speak_multi [
  { text: "Hello, I am TARS", language: "en" }
  { text: "Bonjour, je suis TARS", language: "fr" }
  { text: "Hola, soy TARS", language: "es" }
]
```

## Voice Models

The TARS Speech System supports various voice models:

### Standard Voices

- **English**: `tts_models/en/ljspeech/tacotron2-DDC`
- **French**: `tts_models/fr/mai/tacotron2-DDC`
- **Spanish**: `tts_models/es/mai/tacotron2-DDC`
- **German**: `tts_models/de/thorsten/tacotron2-DDC`
- **Italian**: `tts_models/it/mai_female/glow-tts`
- **Dutch**: `tts_models/nl/mai/tacotron2-DDC`
- **Russian**: `tts_models/ru/multi-dataset/vits`

### Voice Cloning

- **Multi-speaker**: `tts_models/multilingual/multi-dataset/your_tts`

## Voice Cloning

The TARS Speech System supports voice cloning using reference audio samples. This allows TARS to speak in a voice similar to the reference audio.

To use voice cloning:

1. Prepare a reference audio file (WAV format, 5-10 seconds of clean speech)
2. Use the `--speaker-wav` option to specify the reference audio file:

```bash
tarscli speech speak --text "Hello" --speaker-wav reference.wav
```

Or in TARS DSL:

```tars
speak_extended {
  text "Hello"
  speaker_wav "reference.wav"
}
```

## Configuration

The TARS Speech System can be configured using:

1. **Command Line**:
   ```bash
   tarscli speech configure --default-voice "tts_models/en/ljspeech/tacotron2-DDC" --default-language en
   ```

2. **TARS DSL**:
   ```tars
   speech_module {
     enabled true
     default_voice "tts_models/en/ljspeech/tacotron2-DDC"
     language "en"
     preload_voices true
     max_concurrency 1
     log_transcripts true
   }
   ```

## Implementation Details

The TARS Speech System is implemented using:

1. **Coqui TTS**: A deep learning toolkit for Text-to-Speech
2. **Flask**: A lightweight web server for the TTS API
3. **F# and C#**: Integration with TARS CLI and DSL
4. **Cross-platform audio playback**: NAudio (Windows), aplay (Linux), afplay (macOS)

## Extending the Speech System

To add support for new languages or voice models:

1. Find a suitable model in the [Coqui TTS Model Zoo](https://github.com/coqui-ai/TTS/wiki/Models-and-Languages)
2. Add the model to the `VoiceModels` map in `TarsDslSpeechExtensions.fs`
3. Use the model in your TARS DSL scripts or command line commands

## Troubleshooting

If you encounter issues with the TARS Speech System:

1. **Python not installed**: Install Python 3.7 or later
2. **TTS not installed**: Run `pip install TTS flask soundfile numpy langdetect`
3. **Audio playback issues**: Ensure your system has audio output configured correctly
4. **Voice model not found**: Check the model name and ensure it's available in the Coqui TTS Model Zoo
