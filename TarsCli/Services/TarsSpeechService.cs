using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;

namespace TarsCli.Services
{
    /// <summary>
    /// Service for text-to-speech functionality in TARS
    /// </summary>
    public class TarsSpeechService
    {
        private readonly ILogger<TarsSpeechService> _logger;
        private readonly IConfiguration _configuration;
        private Process? _serverProcess;
        private bool _isInitialized = false;
        private readonly HttpClient _httpClient = new HttpClient();

        public TarsSpeechService(
            ILogger<TarsSpeechService> logger,
            IConfiguration configuration)
        {
            _logger = logger;
            _configuration = configuration;
        }

        /// <summary>
        /// Initialize the speech service
        /// </summary>
        public void Initialize()
        {
            if (_isInitialized)
                return;

            try
            {
                _logger.LogInformation("Initializing TarsSpeechService");

                // Check if Python is installed
                if (!IsPythonInstalled())
                {
                    _logger.LogError("Python is not installed. Please install Python 3.7 or later.");
                    return;
                }

                // Install TTS and dependencies
                InstallTts();

                // Start the TTS server
                StartServer();

                // Create spoken_trace.tars if it doesn't exist
                if (!File.Exists("spoken_trace.tars"))
                {
                    File.WriteAllText("spoken_trace.tars", "# TARS Speech Transcript\n\n");
                }

                _isInitialized = true;
                _logger.LogInformation("TarsSpeechService initialized successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error initializing TarsSpeechService");
            }
        }

        /// <summary>
        /// Check if Python is installed
        /// </summary>
        private bool IsPythonInstalled()
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = "--version",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = Process.Start(psi);
                process?.WaitForExit();

                return process?.ExitCode == 0;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Install TTS and dependencies
        /// </summary>
        private void InstallTts()
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = "-m pip install TTS flask soundfile numpy langdetect",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = Process.Start(psi);
                process?.WaitForExit();

                if (process?.ExitCode != 0)
                {
                    _logger.LogError("Failed to install TTS and dependencies");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error installing TTS and dependencies");
            }
        }

        /// <summary>
        /// Start the TTS server
        /// </summary>
        private void StartServer()
        {
            try
            {
                // Check if server is already running
                if (_serverProcess != null && !_serverProcess.HasExited)
                {
                    return;
                }

                // Create Python directory if it doesn't exist
                Directory.CreateDirectory("Python");

                // Create tts_server.py if it doesn't exist
                if (!File.Exists("Python/tts_server.py"))
                {
                    File.WriteAllText("Python/tts_server.py", GetTtsServerScript());
                }

                var psi = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = "Python/tts_server.py",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                _serverProcess = new Process
                {
                    StartInfo = psi
                };

                _serverProcess.Start();

                // Wait for server to start
                System.Threading.Thread.Sleep(3000);

                _logger.LogInformation("TTS server started");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error starting TTS server");
            }
        }

        /// <summary>
        /// Get the TTS server script
        /// </summary>
        private string GetTtsServerScript()
        {
            return @"from flask import Flask, request, Response
from TTS.api import TTS
import io
import soundfile as sf
import os
import numpy as np
from langdetect import detect

app = Flask(__name__)

# Cache for loaded models
models = {}
default_model = ""tts_models/en/ljspeech/tacotron2-DDC""

# Ensure models directory exists
os.makedirs(""models"", exist_ok=True)

@app.route(""/speak"", methods=[""POST""])
def speak():
    data = request.json
    text = data[""text""]
    model_name = data.get(""voice"", default_model)
    language = data.get(""language"", None)
    speaker_wav = data.get(""speaker_wav"", None)
    agent_id = data.get(""agentId"", ""TARS"")

    # Auto-detect language if not specified
    if language is None:
        try:
            language = detect(text)
        except:
            language = ""en""

    # Get or load the model
    if model_name not in models:
        print(f""Loading model: {model_name}"")
        if ""multi-dataset"" in model_name and ""your_tts"" in model_name:
            # Multi-speaker model that supports voice cloning
            models[model_name] = TTS(model_name=model_name, progress_bar=False)
        else:
            # Standard single-speaker model
            models[model_name] = TTS(model_name=model_name, progress_bar=False)

    tts = models[model_name]

    # Generate speech
    if speaker_wav and hasattr(tts, ""tts_with_vc""):
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
    with open(""spoken_trace.tars"", ""a"", encoding=""utf-8"") as f:
        timestamp = np.datetime64('now')
        f.write(f""[{timestamp}] [{agent_id}] [{language}]: {text}\n"")

    return Response(buf.read(), mimetype=""audio/wav"")

@app.route(""/preload"", methods=[""POST""])
def preload():
    data = request.json
    model_names = data.get(""models"", [default_model])

    for model_name in model_names:
        if model_name not in models:
            print(f""Preloading model: {model_name}"")
            models[model_name] = TTS(model_name=model_name, progress_bar=False)

    return {""status"": ""ok"", ""loaded_models"": list(models.keys())}

@app.route(""/status"", methods=[""GET""])
def status():
    return {
        ""status"": ""ok"",
        ""loaded_models"": list(models.keys()),
        ""default_model"": default_model
    }

if __name__ == ""__main__"":
    # Load default model on startup
    models[default_model] = TTS(model_name=default_model, progress_bar=False)
    app.run(host=""0.0.0.0"", port=5002)";
        }

        /// <summary>
        /// Speak text using the default voice
        /// </summary>
        /// <param name="text">Text to speak</param>
        public void Speak(string text)
        {
            Speak(text, null, null, null, null);
        }

        /// <summary>
        /// Speak text with specific voice settings
        /// </summary>
        /// <param name="text">Text to speak</param>
        /// <param name="voice">Voice model to use</param>
        /// <param name="language">Language code</param>
        /// <param name="speakerWav">Path to speaker reference audio for voice cloning</param>
        /// <param name="agentId">Agent identifier</param>
        public void Speak(string text, string? voice = null, string? language = null, string? speakerWav = null, string? agentId = null)
        {
            Initialize();

            Task.Run(async () =>
            {
                try
                {
                    var bytes = await SpeakToBytes(text, voice, language, speakerWav, agentId);
                    PlayAudio(bytes);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error speaking text");
                }
            });
        }

        /// <summary>
        /// Speak text and get audio bytes
        /// </summary>
        private async Task<byte[]> SpeakToBytes(string text, string? voice = null, string? language = null, string? speakerWav = null, string? agentId = null)
        {
            var requestObj = new Dictionary<string, object>
            {
                ["text"] = text
            };

            if (voice != null)
                requestObj["voice"] = voice;

            if (language != null)
                requestObj["language"] = language;

            if (speakerWav != null)
                requestObj["speaker_wav"] = speakerWav;

            if (agentId != null)
                requestObj["agentId"] = agentId;

            var json = JsonSerializer.Serialize(requestObj);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync("http://localhost:5002/speak", content);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsByteArrayAsync();
        }

        /// <summary>
        /// Play audio bytes
        /// </summary>
        private void PlayAudio(byte[] wavBytes)
        {
            try
            {
                var tempFile = Path.Combine(Path.GetTempPath(), $"tars_speech_{Guid.NewGuid()}.wav");
                File.WriteAllBytes(tempFile, wavBytes);

                ProcessStartInfo psi;

                if (OperatingSystem.IsWindows())
                {
                    psi = new ProcessStartInfo
                    {
                        FileName = "powershell",
                        Arguments = $"-c (New-Object Media.SoundPlayer '{tempFile}').PlaySync()",
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };
                }
                else if (OperatingSystem.IsLinux())
                {
                    psi = new ProcessStartInfo
                    {
                        FileName = "aplay",
                        Arguments = tempFile,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };
                }
                else if (OperatingSystem.IsMacOS())
                {
                    psi = new ProcessStartInfo
                    {
                        FileName = "afplay",
                        Arguments = tempFile,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };
                }
                else
                {
                    _logger.LogError("Unsupported platform for audio playback");
                    return;
                }

                using var process = Process.Start(psi);
                process?.WaitForExit();

                // Delete temp file
                try
                {
                    File.Delete(tempFile);
                }
                catch
                {
                    // Ignore errors when deleting temp file
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error playing audio");
            }
        }

        /// <summary>
        /// Speak multiple texts in sequence
        /// </summary>
        /// <param name="texts">List of texts to speak</param>
        public void SpeakSequence(IEnumerable<string> texts)
        {
            Initialize();

            Task.Run(async () =>
            {
                try
                {
                    foreach (var text in texts)
                    {
                        var bytes = await SpeakToBytes(text);
                        PlayAudio(bytes);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error speaking sequence");
                }
            });
        }

        /// <summary>
        /// Apply speech configuration
        /// </summary>
        /// <param name="enabled">Whether speech is enabled</param>
        /// <param name="defaultVoice">Default voice model</param>
        /// <param name="language">Default language</param>
        /// <param name="preloadVoices">Whether to preload voice models</param>
        /// <param name="maxConcurrency">Maximum concurrent speech operations</param>
        public void Configure(bool enabled = true, string? defaultVoice = null, string? language = null,
                             bool? preloadVoices = null, int? maxConcurrency = null)
        {
            Initialize();

            if (defaultVoice != null || language != null)
            {
                Task.Run(async () =>
                {
                    try
                    {
                        var requestObj = new Dictionary<string, object>();

                        if (defaultVoice != null)
                            requestObj["default_voice"] = defaultVoice;

                        if (language != null)
                            requestObj["language"] = language;

                        var json = JsonSerializer.Serialize(requestObj);
                        var content = new StringContent(json, Encoding.UTF8, "application/json");
                        var response = await _httpClient.PostAsync("http://localhost:5002/preload", content);
                        response.EnsureSuccessStatusCode();
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error configuring speech");
                    }
                });
            }
        }

        /// <summary>
        /// Shutdown the speech service
        /// </summary>
        public void Shutdown()
        {
            if (!_isInitialized)
                return;

            try
            {
                _logger.LogInformation("Shutting down TarsSpeechService");

                if (_serverProcess != null && !_serverProcess.HasExited)
                {
                    _serverProcess.Kill();
                    _serverProcess = null;
                }

                _isInitialized = false;
                _logger.LogInformation("TarsSpeechService shut down successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error shutting down TarsSpeechService");
            }
        }
    }
}
