using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Net;
using System.Net.Sockets;
using Microsoft.Extensions.Configuration;

namespace TarsCli.Services;

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

            // Create spoken_trace.tars if it doesn't exist
            if (!File.Exists("spoken_trace.tars"))
            {
                File.WriteAllText("spoken_trace.tars", "# TARS Speech Transcript\n\n");
            }

            // Check if Python is installed
            if (!IsPythonInstalled())
            {
                _logger.LogError("Python is not installed. Please install Python 3.7 or later.");
                return;
            }

            // Create Python directory if it doesn't exist
            var pythonDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Python");
            Directory.CreateDirectory(pythonDir);

            // Copy the debug script to the Python directory
            var debugScriptPath = Path.Combine(pythonDir, "tts_server_debug.py");

            // Try multiple possible locations for the script
            var possibleScriptPaths = new List<string>
            {
                // Path when running from bin directory
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "TarsCli", "Python", "tts_server_debug.py"),

                // Path when running from repository root
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TarsCli", "Python", "tts_server_debug.py"),

                // Path when running from TarsCli directory
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Python", "tts_server_debug.py"),

                // Absolute path based on repository structure
                Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(AppDomain.CurrentDomain.BaseDirectory) ?? string.Empty) ?? string.Empty, "TarsCli", "Python", "tts_server_debug.py")
            };

            // Find the first path that exists
            var repoScriptPath = possibleScriptPaths.FirstOrDefault(File.Exists);

            if (repoScriptPath != null)
            {
                try
                {
                    // Check if the file already exists and is identical
                    bool needsCopy = true;
                    if (File.Exists(debugScriptPath))
                    {
                        try
                        {
                            // Compare file contents to see if we need to copy
                            byte[] sourceBytes = File.ReadAllBytes(repoScriptPath);
                            byte[] destBytes = File.ReadAllBytes(debugScriptPath);

                            if (sourceBytes.SequenceEqual(destBytes))
                            {
                                needsCopy = false;
                                _logger.LogInformation($"Debug script already exists and is up to date at {debugScriptPath}");
                            }
                        }
                        catch (IOException)
                        {
                            // If we can't read the file, assume we need to copy
                            needsCopy = true;
                        }
                    }

                    if (needsCopy)
                    {
                        // Try to copy the file, but handle the case where it's in use
                        try
                        {
                            File.Copy(repoScriptPath, debugScriptPath, true);
                            _logger.LogInformation($"Copied debug script from {repoScriptPath} to {debugScriptPath}");
                        }
                        catch (IOException ex) when (ex.Message.Contains("being used by another process"))
                        {
                            // If the file is in use, we can still proceed if it exists
                            if (File.Exists(debugScriptPath))
                            {
                                _logger.LogWarning($"Debug script is in use and cannot be updated. Using existing file at {debugScriptPath}");
                            }
                            else
                            {
                                throw; // Rethrow if the file doesn't exist
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, $"Error copying debug script from {repoScriptPath} to {debugScriptPath}");
                    throw;
                }
            }
            else
            {
                _logger.LogError($"Debug script not found in any of the expected locations. TTS will not be available.");
                _logger.LogInformation($"Searched in: {string.Join(", ", possibleScriptPaths)}");
                return;
            }

            // Install TTS and dependencies
            InstallTts();

            // Start the TTS server
            StartServer();

            _isInitialized = true;
            _logger.LogInformation("TarsSpeechService initialized successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing TarsSpeechService");
        }
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
        // Make sure we're initialized
        Initialize();

        // Use the Python implementation
        _logger.LogInformation("Using Python implementation for TTS");

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
    /// Check if a port is in use
    /// </summary>
    /// <param name="port">Port number to check</param>
    /// <returns>True if the port is in use, false otherwise</returns>
    private bool IsPortInUse(int port)
    {
        try
        {
            using (var client = new TcpClient())
            {
                var result = client.BeginConnect(IPAddress.Loopback, port, null, null);
                var success = result.AsyncWaitHandle.WaitOne(TimeSpan.FromMilliseconds(100));
                if (success)
                {
                    client.EndConnect(result);
                    return true; // Port is in use
                }
                else
                {
                    return false; // Port is not in use
                }
            }
        }
        catch
        {
            return false; // Error occurred, assume port is not in use
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
            _logger.LogInformation("Installing TTS and dependencies");

            // Install required packages
            var packages = new[] { "numpy", "torch", "soundfile", "flask", "requests" };

            foreach (var package in packages)
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"-m pip install {package}",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = Process.Start(psi);
                process?.WaitForExit();

                if (process?.ExitCode != 0)
                {
                    _logger.LogWarning($"Failed to install {package}");
                }
                else
                {
                    _logger.LogInformation($"Successfully installed {package}");
                }
            }

            // Now install TTS with specific version that works with Python 3.12
            var ttsPsi = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = "-m pip install TTS==0.17.6",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using (var ttsProcess = Process.Start(ttsPsi))
            {
                ttsProcess?.WaitForExit();
                if (ttsProcess?.ExitCode != 0)
                {
                    _logger.LogError("Failed to install TTS package");
                }
                else
                {
                    _logger.LogInformation("Successfully installed TTS package");
                }
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
            if (_serverProcess != null)
            {
                try
                {
                    if (!_serverProcess.HasExited)
                    {
                        _logger.LogInformation("TTS server is already running");
                        return;
                    }
                }
                catch (InvalidOperationException)
                {
                    // Process may have been terminated externally
                    _logger.LogWarning("Previous TTS server process is no longer valid. Starting a new one.");
                    _serverProcess = null;
                }
            }

            // Check if there's already a TTS server running by checking the port
            if (IsPortInUse(5002))
            {
                _logger.LogInformation("TTS server port 5002 is already in use. Assuming server is running.");
                return;
            }

            // Create Python directory if it doesn't exist
            var pythonDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Python");
            Directory.CreateDirectory(pythonDir);

            // Get the path to the debug script
            var serverScriptPath = Path.Combine(pythonDir, "tts_server_debug.py");
            if (!File.Exists(serverScriptPath))
            {
                _logger.LogError($"TTS server script not found at {serverScriptPath}");
                return;
            }

            // Start the server
            var psi = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = serverScriptPath,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = pythonDir
            };

            var outputBuilder = new StringBuilder();
            var errorBuilder = new StringBuilder();

            _serverProcess = new Process
            {
                StartInfo = psi,
                EnableRaisingEvents = true
            };

            _serverProcess.OutputDataReceived += (sender, args) =>
            {
                if (!string.IsNullOrEmpty(args.Data))
                {
                    outputBuilder.AppendLine(args.Data);
                    _logger.LogInformation($"TTS server: {args.Data}");
                }
            };

            _serverProcess.ErrorDataReceived += (sender, args) =>
            {
                if (!string.IsNullOrEmpty(args.Data))
                {
                    errorBuilder.AppendLine(args.Data);
                    _logger.LogWarning($"TTS server error: {args.Data}");
                }
            };

            _serverProcess.Start();
            _serverProcess.BeginOutputReadLine();
            _serverProcess.BeginErrorReadLine();

            // Wait for server to start
            Thread.Sleep(3000);

            // Check if server is running by making a request to the status endpoint
            try
            {
                var response = _httpClient.GetAsync("http://localhost:5002/status").Result;
                if (response.IsSuccessStatusCode)
                {
                    _logger.LogInformation("TTS server started and responding to requests");
                }
                else
                {
                    _logger.LogWarning($"TTS server started but returned status code {response.StatusCode}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning($"TTS server may not be fully initialized: {ex.Message}");
                // Continue anyway, as the server might still be starting up
            }

            _logger.LogInformation("TTS server started");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error starting TTS server");
        }
    }

    /// <summary>
    /// Speak text and get audio bytes
    /// </summary>
    private async Task<byte[]> SpeakToBytes(string text, string? voice = null, string? language = null, string? speakerWav = null, string? agentId = null)
    {
        var requestObj = new Dictionary<string, object>
        {
            { "text", text }
        };

        if (!string.IsNullOrEmpty(voice))
        {
            requestObj["model"] = voice;
        }

        if (!string.IsNullOrEmpty(language))
        {
            requestObj["language"] = language;
        }

        if (!string.IsNullOrEmpty(speakerWav))
        {
            requestObj["speaker_wav"] = speakerWav;
        }

        if (!string.IsNullOrEmpty(agentId))
        {
            requestObj["agent_id"] = agentId;
        }

        var content = new StringContent(
            JsonSerializer.Serialize(requestObj),
            Encoding.UTF8,
            "application/json");

        var response = await _httpClient.PostAsync("http://localhost:5002/speak", content);

        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync();
            throw new Exception($"TTS server returned error: {response.StatusCode} - {errorContent}");
        }

        return await response.Content.ReadAsByteArrayAsync();
    }

    /// <summary>
    /// Play audio bytes
    /// </summary>
    private void PlayAudio(byte[] audioBytes)
    {
        try
        {
            // Create a temporary file for the audio
            var tempFile = Path.Combine(Path.GetTempPath(), $"tars_speech_{Guid.NewGuid()}.wav");
            File.WriteAllBytes(tempFile, audioBytes);

            // Play the audio using the default player
            var psi = new ProcessStartInfo
            {
                FileName = tempFile,
                UseShellExecute = true
            };

            Process.Start(psi);

            // Schedule the temp file for deletion after a delay
            Task.Run(async () =>
            {
                await Task.Delay(10000); // Wait 10 seconds
                try
                {
                    if (File.Exists(tempFile))
                    {
                        File.Delete(tempFile);
                    }
                }
                catch
                {
                    // Ignore errors when deleting temp file
                }
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error playing audio");
        }
    }

    /// <summary>
    /// Configure the speech service
    /// </summary>
    /// <param name="enabled">Whether speech is enabled</param>
    /// <param name="defaultVoice">Default voice to use</param>
    /// <param name="defaultLanguage">Default language to use</param>
    public void Configure(bool enabled, string? defaultVoice = null, string? defaultLanguage = null)
    {
        _logger.LogInformation($"Configuring speech service: enabled={enabled}, defaultVoice={defaultVoice}, defaultLanguage={defaultLanguage}");

        // Store configuration in appsettings or similar if needed
        // For now, just log the configuration
    }

    /// <summary>
    /// Get available voices
    /// </summary>
    /// <returns>List of available voices</returns>
    public List<string> GetAvailableVoices()
    {
        _logger.LogInformation("Getting available voices");

        var voices = new List<string>();

        try
        {
            // Add some default voices for demonstration purposes
            voices.Add("en-US-Standard-A (Female)");
            voices.Add("en-US-Standard-B (Male)");
            voices.Add("en-US-Standard-C (Female)");
            voices.Add("en-US-Standard-D (Male)");
            voices.Add("en-US-Standard-E (Female)");
            voices.Add("en-US-Standard-F (Female)");
            voices.Add("en-US-Standard-G (Male)");
            voices.Add("en-US-Standard-H (Female)");
            voices.Add("en-US-Standard-I (Male)");
            voices.Add("en-US-Standard-J (Male)");
            voices.Add("fr-FR-Standard-A (Female)");
            voices.Add("fr-FR-Standard-B (Male)");
            voices.Add("fr-FR-Standard-C (Female)");
            voices.Add("fr-FR-Standard-D (Male)");
            voices.Add("fr-FR-Standard-E (Female)");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting available voices");
        }

        return voices;
    }
}