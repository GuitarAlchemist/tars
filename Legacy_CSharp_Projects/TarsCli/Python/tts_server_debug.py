import sys
import traceback
import os
import io
import logging
import time
from datetime import datetime
from flask import Flask, request, Response

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"tts_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("tts_server")
logger.info(f"Starting TTS server with Python {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Log file: {log_file}")

# Try to import required packages and log any errors
try:
    import numpy as np
    logger.info("Successfully imported numpy")
except ImportError as e:
    logger.error(f"Failed to import numpy: {e}")
    logger.error("Please install numpy with: pip install numpy")
    sys.exit(1)

try:
    from flask import Flask, request, Response
    logger.info("Successfully imported flask")
except ImportError as e:
    logger.error(f"Failed to import flask: {e}")
    logger.error("Please install flask with: pip install flask")
    sys.exit(1)

# Set up error handling for TTS import
tts_import_error = None
models = {}
default_model = 'tts_models/en/ljspeech/tacotron2-DDC'

try:
    from TTS.api import TTS
    logger.info("Successfully imported TTS")
except Exception as e:
    tts_import_error = str(e)
    logger.error(f"Error importing TTS: {e}")
    logger.error(traceback.format_exc())
    logger.error("Please install TTS with: pip install TTS==0.17.6")

try:
    import soundfile as sf
    logger.info("Successfully imported soundfile")
except ImportError as e:
    logger.error(f"Failed to import soundfile: {e}")
    logger.error("Please install soundfile with: pip install soundfile")
    sys.exit(1)

app = Flask(__name__)

@app.route("/status", methods=["GET"])
def status():
    if tts_import_error:
        logger.error(f"TTS module could not be imported: {tts_import_error}")
        return {"status": "error", "message": f"TTS module could not be imported: {tts_import_error}"}, 500
    return {"status": "ok", "models": list(models.keys())}

@app.route("/speak", methods=["POST"])
def speak():
    # Check for import errors
    if tts_import_error:
        logger.error(f"TTS module could not be imported: {tts_import_error}")
        return {"error": f"TTS module could not be imported: {tts_import_error}"}, 500
    
    try:
        data = request.json
        text = data.get("text", "")
        model_name = data.get("model", default_model)
        language = data.get("language", None)
        speaker_wav = data.get("speaker_wav", None)
        agent_id = data.get("agent_id", "default")
        
        if not text:
            logger.error("No text provided")
            return {"error": "No text provided"}, 400
            
        logger.info(f"Generating speech for: {text}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Language: {language}")
        logger.info(f"Speaker WAV: {speaker_wav}")
        
        # Load model if not already loaded
        if model_name not in models:
            try:
                logger.info(f"Loading model: {model_name}")
                models[model_name] = TTS(model_name=model_name, progress_bar=False)
                logger.info(f"Model loaded: {model_name}")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                logger.error(traceback.format_exc())
                return {"error": f"Failed to load model {model_name}: {str(e)}"}, 500
        
        # Generate speech
        try:
            model = models[model_name]
            
            # Set up parameters for speech generation
            kwargs = {}
            
            if language:
                kwargs["language"] = language
                
            if speaker_wav:
                kwargs["speaker_wav"] = speaker_wav
                
            # Generate speech
            logger.info(f"Generating speech with kwargs: {kwargs}")
            start_time = time.time()
            wav = model.tts(text=text, **kwargs)
            end_time = time.time()
            logger.info(f"Speech generation took {end_time - start_time:.2f} seconds")
            
            # Convert to WAV format
            try:
                buf = io.BytesIO()
                sf.write(buf, wav, 22050, format='WAV')
                buf.seek(0)
                
                # Log the speech
                try:
                    with open("spoken_trace.tars", "a", encoding="utf-8") as f:
                        timestamp = np.datetime64('now')
                        f.write(f"[{timestamp}] [{agent_id}] [{language}]: {text}\n")
                except Exception as e:
                    logger.error(f"Error logging speech: {e}")
                    # Continue even if logging fails
                
                logger.info("Successfully generated speech")
                return Response(buf.read(), mimetype="audio/wav")
            except Exception as e:
                logger.error(f"Error converting to WAV: {e}")
                logger.error(traceback.format_exc())
                return {"error": f"Failed to convert to WAV: {str(e)}"}, 500
        except Exception as e:
            logger.error(f"Unexpected error in speak endpoint: {e}")
            logger.error(traceback.format_exc())
            return {"error": f"Unexpected error: {str(e)}"}, 500

@app.route("/preload", methods=["POST"])
def preload():
    # Check for import errors
    if tts_import_error:
        logger.error(f"TTS module could not be imported: {tts_import_error}")
        return {"error": f"TTS module could not be imported: {tts_import_error}"}, 500
        
    try:
        data = request.json
        model_names = data.get("models", [default_model])
        
        for model_name in model_names:
            if model_name not in models:
                logger.info(f"Preloading model: {model_name}")
                try:
                    models[model_name] = TTS(model_name=model_name, progress_bar=False)
                except Exception as e:
                    logger.error(f"Error preloading model {model_name}: {e}")
                    # Continue with other models even if one fails
        
        return {"status": "ok", "loaded_models": list(models.keys())}
    except Exception as e:
        logger.error(f"Error in preload endpoint: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Failed to preload models: {str(e)}"}, 500

# Try to load default model on startup if TTS is available
if not tts_import_error:
    try:
        logger.info(f"Loading default model: {default_model}")
        models[default_model] = TTS(model_name=default_model, progress_bar=False)
        logger.info(f"Successfully loaded default model: {default_model}")
    except Exception as e:
        logger.error(f"Error loading default model: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("Starting Flask server on port 5002")
    app.run(host="0.0.0.0", port=5002, debug=True)
