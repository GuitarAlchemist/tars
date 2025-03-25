using System.Diagnostics;
using System.Text;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services;

public class RetroactionService
{
    private readonly OllamaService _ollamaService;
    private readonly ILogger<RetroactionService> _logger;
    private readonly string _projectRoot;

    public RetroactionService(
        OllamaService ollamaService,
        ILogger<RetroactionService> logger,
        IConfiguration configuration)
    {
        _ollamaService = ollamaService;
        _logger = logger;
        _projectRoot = configuration["Tars:ProjectRoot"] ?? 
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), 
                "source", "repos", "tars");
    }

    public async Task<bool> ProcessFile(string filePath, string taskDescription, string model = "codellama:13b-code")
    {
        try
        {
            // Ensure file exists
            if (!File.Exists(filePath))
            {
                _logger.LogError($"File not found: {filePath}");
                return false;
            }

            // Create output directory
            string outputDir = Path.Combine(_projectRoot, "output", $"v{DateTime.UtcNow:yyyyMMdd}");
            Directory.CreateDirectory(outputDir);

            // Read the file
            string fileContent = await File.ReadAllTextAsync(filePath);
            string fileName = Path.GetFileName(filePath);

            // Save original version
            string originalPath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(fileName)}_original{Path.GetExtension(fileName)}");
            await File.WriteAllTextAsync(originalPath, fileContent);

            // Generate prompt for Ollama
            string prompt = GeneratePrompt(fileContent, taskDescription, filePath);

            // Get completion from Ollama
            _logger.LogInformation("Sending to Ollama for processing...");
            string improvedCode = await _ollamaService.GenerateCompletion(prompt, model);

            // Extract code from the response
            string extractedCode = ExtractCodeFromResponse(improvedCode, Path.GetExtension(filePath));
            
            // Save the improved version
            string improvedPath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(fileName)}_improved{Path.GetExtension(fileName)}");
            await File.WriteAllTextAsync(improvedPath, extractedCode);

            // Save the full response
            string responsePath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(fileName)}_response.md");
            await File.WriteAllTextAsync(responsePath, improvedCode);

            // Save metadata
            SaveTarsMetadata(filePath, taskDescription, model, outputDir);

            // Ask user if they want to apply changes
            Console.WriteLine("\nImproved code generated. Would you like to apply these changes? (y/n)");
            string? response = Console.ReadLine()?.ToLower();
            
            if (response == "y" || response == "yes")
            {
                await File.WriteAllTextAsync(filePath, extractedCode);
                _logger.LogInformation("Changes applied successfully");
                return true;
            }
            else
            {
                _logger.LogInformation("Changes not applied");
                return true; // Still consider this a success since the process completed
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in retroaction process");
            return false;
        }
    }

    private string GeneratePrompt(string code, string task, string filePath)
    {
        return $@"You are TARS, an AI assistant specialized in improving code.

TASK: {task}

FILE: {filePath}

CODE:
```
{code}
```

Please provide an improved version of this code that addresses the task. 
Return the improved code in a code block, followed by a brief explanation of the changes.";
    }

    private string ExtractCodeFromResponse(string response, string fileExtension)
    {
        // Extract code from markdown code blocks
        if (response.Contains("```"))
        {
            var codeBlockStart = response.IndexOf("```");
            var codeBlockEnd = response.IndexOf("```", codeBlockStart + 3);
            
            if (codeBlockEnd > codeBlockStart)
            {
                var codeBlock = response.Substring(codeBlockStart + 3, codeBlockEnd - codeBlockStart - 3);
                
                // Remove language identifier if present
                var firstLineEnd = codeBlock.IndexOf('\n');
                if (firstLineEnd > 0)
                {
                    var firstLine = codeBlock.Substring(0, firstLineEnd).Trim();
                    if (firstLine.Contains(fileExtension.TrimStart('.')) || 
                        IsLanguageIdentifier(firstLine, fileExtension))
                    {
                        codeBlock = codeBlock.Substring(firstLineEnd + 1);
                    }
                }
                
                return codeBlock.Trim();
            }
        }
        
        // If no code blocks found, return the raw response
        return response;
    }

    private bool IsLanguageIdentifier(string line, string extension)
    {
        // Map file extensions to common language identifiers in markdown
        var extensionMap = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase)
        {
            { ".cs", new List<string> { "csharp", "c#" } },
            { ".js", new List<string> { "javascript" } },
            { ".py", new List<string> { "python" } },
            { ".md", new List<string> { "markdown" } },
            { ".html", new List<string> { "html" } },
            { ".css", new List<string> { "css" } },
            { ".json", new List<string> { "json" } }
        };

        if (extensionMap.TryGetValue(extension, out var identifiers))
        {
            return identifiers.Any(id => line.Equals(id, StringComparison.OrdinalIgnoreCase));
        }

        return false;
    }

    private void SaveTarsMetadata(string filePath, string task, string model, string outputDir)
    {
        try
        {
            string metadataPath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(filePath)}.tars");
            
            string metadata = $@"TARS Iteration {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}
---
file: ""{filePath}""
task: ""{task}""
model: ""{model}""
timestamp: {DateTime.UtcNow:yyyy-MM-ddTHH:mm:ssZ}
";

            File.WriteAllText(metadataPath, metadata);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving TARS metadata");
        }
    }
}