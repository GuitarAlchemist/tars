using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Text.RegularExpressions;

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
            var outputDir = Path.Combine(_projectRoot, "output", $"v{DateTime.UtcNow:yyyyMMdd}");
            Directory.CreateDirectory(outputDir);

            // Read the file
            var fileContent = await File.ReadAllTextAsync(filePath);
            var fileName = Path.GetFileName(filePath);

            // Save original version
            var originalPath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(fileName)}_original{Path.GetExtension(fileName)}");
            await File.WriteAllTextAsync(originalPath, fileContent);

            // Generate prompt for Ollama
            var prompt = GeneratePrompt(fileContent, taskDescription, filePath);

            // Get completion from Ollama
            _logger.LogInformation("Sending to Ollama for processing...");
            var improvedCode = await _ollamaService.GenerateCompletion(prompt, model);

            // Extract code from the response
            var extractedCode = ExtractCodeFromResponse(improvedCode, Path.GetExtension(filePath));
            
            // Save the improved version
            var improvedPath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(fileName)}_improved{Path.GetExtension(fileName)}");
            await File.WriteAllTextAsync(improvedPath, extractedCode);

            // Save the full response
            var responsePath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(fileName)}_response.md");
            await File.WriteAllTextAsync(responsePath, improvedCode);

            // Save metadata
            SaveTarsMetadata(filePath, taskDescription, model, outputDir);

            // Ask user if they want to apply changes
            Console.WriteLine("\nImproved code generated. Would you like to apply these changes? (y/n)");
            var response = Console.ReadLine()?.ToLower();
            
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
        try
        {
            // Look for code blocks with the specific language
            var codeBlockPattern = $"```(?:{GetLanguageIdentifier(fileExtension)})?\\s*\\n([\\s\\S]*?)\\n```";
            var match = Regex.Match(response, codeBlockPattern, RegexOptions.IgnoreCase);
            
            if (match.Success && match.Groups.Count > 1)
            {
                return match.Groups[1].Value.Trim();
            }
            
            // Fallback: look for any code block
            var anyCodeBlockPattern = "```\\s*\\n([\\s\\S]*?)\\n```";
            match = Regex.Match(response, anyCodeBlockPattern);
            
            if (match.Success && match.Groups.Count > 1)
            {
                return match.Groups[1].Value.Trim();
            }
            
            // If no code blocks found, check if the entire response looks like code
            if (IsLikelyCode(response, fileExtension))
            {
                return response.Trim();
            }
            
            _logger.LogWarning("Could not extract code from response. Returning original response.");
            return response;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting code from response");
            return response;
        }
    }

    private bool IsLikelyCode(string text, string fileExtension)
    {
        // Basic heuristics to determine if text is likely code
        switch (fileExtension.ToLower())
        {
            case ".cs":
                return text.Contains("namespace") || text.Contains("class") || text.Contains("using ");
            case ".js":
            case ".ts":
                return text.Contains("function") || text.Contains("const") || text.Contains("let");
            case ".py":
                return text.Contains("def ") || text.Contains("import ") || text.Contains("class ");
            case ".md":
                return text.Contains("#") || text.Contains("##");
            default:
                return false;
        }
    }

    private string GetLanguageIdentifier(string fileExtension)
    {
        return fileExtension.ToLower() switch
        {
            ".cs" => "csharp|cs|c#",
            ".js" => "javascript|js",
            ".ts" => "typescript|ts",
            ".py" => "python|py",
            ".md" => "markdown|md",
            _ => ""
        };
    }

    private void SaveTarsMetadata(string filePath, string task, string model, string outputDir)
    {
        try
        {
            var metadataPath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(filePath)}.tars");
            
            var metadata = $@"TARS Iteration {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}
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