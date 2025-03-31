using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services
{
    /// <summary>
    /// Service for capturing console output and using it to improve code
    /// </summary>
    public class ConsoleCaptureService
    {
        private readonly ILogger<ConsoleCaptureService> _logger;
        private readonly OllamaService _ollamaService;
        private readonly McpService _mcpService;
        private TextWriter _originalOut;
        private TextWriter _originalError;
        private MemoryStream _memoryStream;
        private StreamWriter _streamWriter;
        private bool _isCapturing = false;
        private readonly List<string> _capturedOutput = new();

        public ConsoleCaptureService(
            ILogger<ConsoleCaptureService> logger,
            OllamaService ollamaService,
            McpService mcpService)
        {
            _logger = logger;
            _ollamaService = ollamaService;
            _mcpService = mcpService;
        }

        /// <summary>
        /// Start capturing console output
        /// </summary>
        public void StartCapture()
        {
            if (_isCapturing)
            {
                _logger.LogWarning("Console capture is already in progress");
                return;
            }

            _logger.LogInformation("Starting console output capture");
            
            // Save the original console output and error writers
            _originalOut = Console.Out;
            _originalError = Console.Error;
            
            // Create a memory stream and writer to capture output
            _memoryStream = new MemoryStream();
            _streamWriter = new StreamWriter(_memoryStream) { AutoFlush = true };
            
            // Redirect console output to our stream
            Console.SetOut(_streamWriter);
            Console.SetError(_streamWriter);
            
            _isCapturing = true;
            _capturedOutput.Clear();
            
            _logger.LogInformation("Console output capture started");
        }

        /// <summary>
        /// Stop capturing console output
        /// </summary>
        /// <returns>The captured output</returns>
        public string StopCapture()
        {
            if (!_isCapturing)
            {
                _logger.LogWarning("No console capture in progress");
                return string.Empty;
            }

            _logger.LogInformation("Stopping console output capture");
            
            // Restore the original console output and error writers
            Console.SetOut(_originalOut);
            Console.SetError(_originalError);
            
            // Get the captured output
            _memoryStream.Position = 0;
            using var reader = new StreamReader(_memoryStream);
            var capturedText = reader.ReadToEnd();
            
            // Clean up
            _streamWriter.Dispose();
            _memoryStream.Dispose();
            
            _isCapturing = false;
            _capturedOutput.Add(capturedText);
            
            _logger.LogInformation($"Console output capture stopped. Captured {capturedText.Length} characters");
            
            return capturedText;
        }

        /// <summary>
        /// Get all captured output
        /// </summary>
        public IReadOnlyList<string> GetCapturedOutput() => _capturedOutput.AsReadOnly();

        /// <summary>
        /// Analyze captured output and suggest code improvements
        /// </summary>
        /// <param name="capturedOutput">The captured console output</param>
        /// <param name="filePath">The file path to improve</param>
        /// <returns>Suggested code improvements</returns>
        public async Task<string> AnalyzeAndSuggestImprovements(string capturedOutput, string filePath)
        {
            if (string.IsNullOrEmpty(capturedOutput))
            {
                _logger.LogWarning("No captured output to analyze");
                return "No captured output to analyze";
            }

            if (string.IsNullOrEmpty(filePath) || !File.Exists(filePath))
            {
                _logger.LogWarning($"File not found: {filePath}");
                return $"File not found: {filePath}";
            }

            _logger.LogInformation($"Analyzing captured output for file: {filePath}");
            
            // Read the current file content
            var currentCode = await File.ReadAllTextAsync(filePath);
            
            // Create a prompt for the LLM to analyze the output and suggest improvements
            var prompt = $@"
You are an expert C# developer tasked with improving code based on console output.

CURRENT CODE:
```csharp
{currentCode}
```

CONSOLE OUTPUT:
```
{capturedOutput}
```

Based on the console output, suggest specific improvements to the code. Focus on:
1. Fixing errors and warnings
2. Improving error handling
3. Enhancing performance
4. Making the code more robust

Provide your suggestions in the following format:
1. ISSUE: [Brief description of the issue]
   LOCATION: [Line number or method name]
   SUGGESTION: [Specific code change]
   REASON: [Why this change improves the code]

2. [Next issue...]

Finally, provide the complete improved code.
";

            try
            {
                // Use Ollama to generate suggestions
                var response = await _ollamaService.GenerateCompletion(prompt, "llama3");
                
                _logger.LogInformation("Generated code improvement suggestions");
                return response;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error analyzing captured output");
                return $"Error analyzing captured output: {ex.Message}";
            }
        }

        /// <summary>
        /// Apply suggested improvements to a file
        /// </summary>
        /// <param name="filePath">The file to improve</param>
        /// <param name="suggestions">The suggested improvements</param>
        /// <returns>Result of the operation</returns>
        public async Task<string> ApplyImprovements(string filePath, string suggestions)
        {
            if (string.IsNullOrEmpty(filePath) || !File.Exists(filePath))
            {
                _logger.LogWarning($"File not found: {filePath}");
                return $"File not found: {filePath}";
            }

            if (string.IsNullOrEmpty(suggestions))
            {
                _logger.LogWarning("No suggestions to apply");
                return "No suggestions to apply";
            }

            _logger.LogInformation($"Applying improvements to file: {filePath}");
            
            try
            {
                // Extract the improved code from the suggestions
                var improvedCode = ExtractImprovedCode(suggestions);
                
                if (string.IsNullOrEmpty(improvedCode))
                {
                    _logger.LogWarning("No improved code found in suggestions");
                    return "No improved code found in suggestions";
                }
                
                // Create a backup of the original file
                var backupPath = $"{filePath}.bak";
                File.Copy(filePath, backupPath, true);
                _logger.LogInformation($"Created backup at: {backupPath}");
                
                // Write the improved code to the file
                await File.WriteAllTextAsync(filePath, improvedCode);
                
                _logger.LogInformation($"Applied improvements to: {filePath}");
                return $"Successfully applied improvements to: {filePath}";
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error applying improvements to: {filePath}");
                return $"Error applying improvements: {ex.Message}";
            }
        }

        /// <summary>
        /// Extract the improved code from the suggestions
        /// </summary>
        private string ExtractImprovedCode(string suggestions)
        {
            // Look for the improved code section, which typically comes after "complete improved code"
            var lines = suggestions.Split('\n');
            var improvedCodeLines = new List<string>();
            bool inImprovedCodeSection = false;
            bool foundImprovedCodeMarker = false;
            
            foreach (var line in lines)
            {
                // Look for markers indicating the start of the improved code
                if (!foundImprovedCodeMarker && 
                    (line.Contains("complete improved code", StringComparison.OrdinalIgnoreCase) ||
                     line.Contains("improved code:", StringComparison.OrdinalIgnoreCase) ||
                     line.Contains("here's the improved code", StringComparison.OrdinalIgnoreCase)))
                {
                    foundImprovedCodeMarker = true;
                    continue;
                }
                
                // Once we've found the marker, look for the code block
                if (foundImprovedCodeMarker && !inImprovedCodeSection && line.Contains("```"))
                {
                    inImprovedCodeSection = true;
                    continue;
                }
                
                // Capture the code until the end of the block
                if (inImprovedCodeSection)
                {
                    if (line.Contains("```"))
                    {
                        break;
                    }
                    
                    improvedCodeLines.Add(line);
                }
            }
            
            return string.Join("\n", improvedCodeLines);
        }

        /// <summary>
        /// Use MCP to generate and apply code improvements
        /// </summary>
        /// <param name="capturedOutput">The captured console output</param>
        /// <param name="filePath">The file path to improve</param>
        /// <returns>Result of the operation</returns>
        public async Task<string> AutoImproveCode(string capturedOutput, string filePath)
        {
            if (string.IsNullOrEmpty(capturedOutput))
            {
                _logger.LogWarning("No captured output to analyze");
                return "No captured output to analyze";
            }

            if (string.IsNullOrEmpty(filePath) || !File.Exists(filePath))
            {
                _logger.LogWarning($"File not found: {filePath}");
                return $"File not found: {filePath}";
            }

            _logger.LogInformation($"Auto-improving code for file: {filePath}");
            
            try
            {
                // Read the current file content
                var currentCode = await File.ReadAllTextAsync(filePath);
                
                // Create a command for the MCP service to generate improved code
                var command = $@"
Analyze this C# code and the console output. Fix any errors, warnings, or issues you find.
Make the code more robust, with better error handling and null checks.

CODE:
{currentCode}

CONSOLE OUTPUT:
{capturedOutput}

Generate an improved version of the code that addresses all issues.
";

                // Use the MCP service to generate improved code
                var response = await _mcpService.ExecuteCommand(command);
                
                // Create a backup of the original file
                var backupPath = $"{filePath}.bak";
                File.Copy(filePath, backupPath, true);
                _logger.LogInformation($"Created backup at: {backupPath}");
                
                // Extract the code from the response and apply it
                var improvedCode = ExtractCodeFromMcpResponse(response);
                
                if (string.IsNullOrEmpty(improvedCode))
                {
                    _logger.LogWarning("No improved code found in MCP response");
                    return "No improved code found in MCP response";
                }
                
                // Write the improved code to the file
                await File.WriteAllTextAsync(filePath, improvedCode);
                
                _logger.LogInformation($"Auto-improved code in: {filePath}");
                return $"Successfully auto-improved code in: {filePath}";
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error auto-improving code: {filePath}");
                return $"Error auto-improving code: {ex.Message}";
            }
        }

        /// <summary>
        /// Extract code from an MCP response
        /// </summary>
        private string ExtractCodeFromMcpResponse(string response)
        {
            // Look for code blocks in the response
            var lines = response.Split('\n');
            var codeLines = new List<string>();
            bool inCodeBlock = false;
            
            foreach (var line in lines)
            {
                if (line.Contains("```csharp") || line.Contains("```C#") || (line.Contains("```") && !inCodeBlock))
                {
                    inCodeBlock = true;
                    continue;
                }
                
                if (inCodeBlock)
                {
                    if (line.Contains("```"))
                    {
                        inCodeBlock = false;
                        continue;
                    }
                    
                    codeLines.Add(line);
                }
            }
            
            return string.Join("\n", codeLines);
        }
    }
}
