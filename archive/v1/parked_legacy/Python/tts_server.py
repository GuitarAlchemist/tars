from flask import Flask, request, Response
from TTS.api import TTS
import io
import soundfile as sf
import os
import numpy as np
from langdetect import detect

app = Flask(__name__)

# Cache for loaded models
models = {}
default_model = "tts_models/en/ljspeech/tacotron2-DDC"

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

@app.route("/speak", methods=["POST"])
def speak():
    data = request.json
    text = data["text"]
    model_name = data.get("voice", default_model)
    language = data.get("language", None)
    speaker_wav = data.get("speaker_wav", None)
    agent_id = data.get("agentId", "TARS")

    # Auto-detect language if not specified
    if language is None:
        try:
            language = detect(text)
        except:
            language = "en"

    # Get or load the model
    if model_name not in models:
        print(f"Loading model: {model_name}")
        if "multi-dataset" in model_name and "your_tts" in model_name:
            # Multi-speaker model that supports voice cloning
            models[model_name] = TTS(model_name=model_name, progress_bar=False)
        else:
            # Standard single-speaker model
            models[model_name] = TTS(model_name=model_name, progress_bar=False)

    tts = models[model_name]

    # Generate speech
    if speaker_wav and hasattr(tts, "tts_with_vc"):
        # Voice cloning
        wav = tts.tts_with_vc(text=text, speaker_wav=speaker_wav)
    else:
        # Regular TTS
        wav = tts.tts(text=text)

    # Convert to WAV format
    buf = io.BytesIO()
    sf.write(buf, wav, 22050, format='WAV')
    buf.seek(0)

    # Log the speech
    with open("spoken_trace.tars", "a", encoding="utf-8") as f:
        timestamp = np.datetime64('now')
        f.write(f"[{timestamp}] [{agent_id}] [{language}]: {text}\n")

    return Response(buf.read(), mimetype="audio/wav")

@app.route("/preload", methods=["POST"])
def preload():
    data = request.json
    model_names = data.get("models", [default_model])

    for model_name in model_names:
        if model_name not in models:
            print(f"Preloading model: {model_name}")
            models[model_name] = TTS(model_name=model_name, progress_bar=False)

    return {"status": "ok", "loaded_models": list(models.keys())}

@app.route("/status", methods=["GET"])
def status():
    return {
        "status": "ok",
        "loaded_models": list(models.keys()),
        "default_model": default_model
    }

if __name__ == "__main__":
    # Load default model on startup
    models[default_model] = TTS(model_name=default_model, progress_bar=False)
    app.run(host="0.0.0.0", port=5002)