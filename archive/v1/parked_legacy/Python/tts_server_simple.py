"""
Simple TTS server that doesn't require the TTS package
Uses gTTS (Google Text-to-Speech) instead
"""

from flask import Flask, request, Response
import sys
import io
import os
import traceback
import tempfile
import time

app = Flask(__name__)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

@app.route("/status", methods=["GET"])
def status():
    return {"status": "ok"}

@app.route("/diagnostics", methods=["GET"])
def diagnostics():
    return {
        "python_version": sys.version,
        "import_errors": {},
        "models_loaded": ["google_tts"]
    }

# Try to import gtts
try:
    from gtts import gTTS
    gtts_import_error = None
except Exception as e:
    gtts_import_error = str(e)
    traceback.print_exc()
    print(f"Error importing gtts: {e}")

@app.route("/speak", methods=["POST"])
def speak():
    # Check for import errors
    if gtts_import_error:
        return {"error": f"gtts module could not be imported: {gtts_import_error}"}, 500
    
    try:
        data = request.json
        text = data["text"]
        language = data.get("language", "en")
        agent_id = data.get("agentId", "TARS")
        
        print(f"Generating speech for text: {text} in language: {language}")
        
        # Generate speech using gTTS
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_file.close()
        
        tts.save(temp_file.name)
        
        # Read the file
        with open(temp_file.name, 'rb') as f:
            audio_data = f.read()
        
        # Delete the temporary file
        os.unlink(temp_file.name)
        
        # Log the speech
        try:
            with open("spoken_trace.tars", "a", encoding="utf-8") as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] [{agent_id}] [{language}]: {text}\n")
        except Exception as e:
            print(f"Error logging speech: {e}")
            # Continue even if logging fails
        
        return Response(audio_data, mimetype="audio/mpeg")
    except Exception as e:
        print(f"Unexpected error in speak endpoint: {e}")
        traceback.print_exc()
        return {"error": f"Unexpected error: {str(e)}"}, 500

@app.route("/voices", methods=["GET"])
def voices():
    return {"voices": ["google_tts"]}

if __name__ == "__main__":
    print("Starting simple TTS server on http://0.0.0.0:5002")
    print(f"Python version: {sys.version}")
    
    # Install gtts if not already installed
    if gtts_import_error:
        print("Installing gtts...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "gtts"])
        try:
            from gtts import gTTS
            print("Successfully installed gtts")
        except Exception as e:
            print(f"Failed to install gtts: {e}")
    
    app.run(host="0.0.0.0", port=5002, debug=True)
