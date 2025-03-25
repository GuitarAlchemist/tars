using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services;

public class DiagnosticsService
{
    private readonly ILogger<DiagnosticsService> _logger;
    private readonly IConfiguration _configuration;
    private readonly OllamaService _ollamaService;

    public DiagnosticsService(
        ILogger<DiagnosticsService> logger,
        IConfiguration configuration,
        OllamaService ollamaService)
    {
        _logger = logger;
        _configuration = configuration;
        _ollamaService = ollamaService;
    }

    public async Task<DiagnosticsResult> RunInitialDiagnosticsAsync()
    {
        var result = new DiagnosticsResult();
        
        // System information
        result.SystemInfo = GetSystemInfo();
        _logger.LogInformation("System: {OS}, {Cores} cores, {Memory}GB RAM", 
            result.SystemInfo.OperatingSystem,
            result.SystemInfo.ProcessorCores,
            result.SystemInfo.AvailableMemoryGB);
        
        // Ollama configuration
        result.OllamaConfig = GetOllamaConfig();
        _logger.LogInformation("Ollama configured at {BaseUrl} with default model {DefaultModel}", 
            result.OllamaConfig.BaseUrl,
            result.OllamaConfig.DefaultModel);
        
        // Check required models
        result.ModelStatus = await CheckRequiredModelsAsync();
        foreach (var model in result.ModelStatus)
        {
            if (model.Value)
                _logger.LogInformation("Model {Model} is available", model.Key);
            else
                _logger.LogWarning("Required model {Model} is not available", model.Key);
        }
        
        // Project configuration
        result.ProjectConfig = GetProjectConfig();
        _logger.LogInformation("Project root: {ProjectRoot}", result.ProjectConfig.ProjectRoot);
        
        return result;
    }

    private SystemInfo GetSystemInfo()
    {
        var info = new SystemInfo
        {
            OperatingSystem = RuntimeInformation.OSDescription,
            ProcessorCores = Environment.ProcessorCount,
            AvailableMemoryGB = GetAvailableMemoryGB()
        };
        
        return info;
    }

    private double GetAvailableMemoryGB()
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "wmic",
                        Arguments = "OS get FreePhysicalMemory /Value",
                        RedirectStandardOutput = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };
                process.Start();
                string output = process.StandardOutput.ReadToEnd();
                process.WaitForExit();

                var match = System.Text.RegularExpressions.Regex.Match(output, @"FreePhysicalMemory=(\d+)");
                if (match.Success && long.TryParse(match.Groups[1].Value, out long memoryKB))
                {
                    return Math.Round(memoryKB / 1024.0 / 1024.0, 2);
                }
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "free",
                        Arguments = "-m",
                        RedirectStandardOutput = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };
                process.Start();
                string output = process.StandardOutput.ReadToEnd();
                process.WaitForExit();

                var lines = output.Split('\n');
                if (lines.Length > 1)
                {
                    var memInfo = lines[1].Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    if (memInfo.Length > 6 && int.TryParse(memInfo[3], out int memoryMB))
                    {
                        return Math.Round(memoryMB / 1024.0, 2);
                    }
                }
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "sysctl",
                        Arguments = "-n hw.memsize",
                        RedirectStandardOutput = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };
                process.Start();
                string output = process.StandardOutput.ReadToEnd();
                process.WaitForExit();

                if (long.TryParse(output.Trim(), out long totalBytes))
                {
                    return Math.Round(totalBytes / 1024.0 / 1024.0 / 1024.0, 2);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error determining available memory");
        }
        
        return 0;
    }

    private OllamaConfig GetOllamaConfig()
    {
        return new OllamaConfig
        {
            BaseUrl = _configuration["Ollama:BaseUrl"] ?? "http://localhost:11434",
            DefaultModel = _configuration["Ollama:DefaultModel"] ?? "codellama:13b-code",
            RequiredModels = (_configuration["Ollama:RequiredModels"] ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries)
        };
    }

    private async Task<Dictionary<string, bool>> CheckRequiredModelsAsync()
    {
        var result = new Dictionary<string, bool>();
        var requiredModels = (_configuration["Ollama:RequiredModels"] ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries);
        
        if (requiredModels.Length == 0)
        {
            // Add default model if no required models specified
            requiredModels = new[] { _configuration["Ollama:DefaultModel"] ?? "codellama:13b-code" };
        }
        
        foreach (var model in requiredModels)
        {
            try
            {
                var modelName = model.Trim();
                var testResult = await _ollamaService.GenerateCompletion("test", modelName);
                result[modelName] = !string.IsNullOrEmpty(testResult) && !testResult.StartsWith("Error");
            }
            catch
            {
                result[model.Trim()] = false;
            }
        }
        
        return result;
    }

    private ProjectConfig GetProjectConfig()
    {
        return new ProjectConfig
        {
            ProjectRoot = _configuration["Tars:ProjectRoot"] ?? "."
        };
    }
}

public class DiagnosticsResult
{
    public SystemInfo SystemInfo { get; set; } = new();
    public OllamaConfig OllamaConfig { get; set; } = new();
    public Dictionary<string, bool> ModelStatus { get; set; } = new();
    public ProjectConfig ProjectConfig { get; set; } = new();
    
    public bool IsReady => ModelStatus.All(m => m.Value);
}

public class SystemInfo
{
    public string OperatingSystem { get; set; } = string.Empty;
    public int ProcessorCores { get; set; }
    public double AvailableMemoryGB { get; set; }
}

public class OllamaConfig
{
    public string BaseUrl { get; set; } = string.Empty;
    public string DefaultModel { get; set; } = string.Empty;
    public string[] RequiredModels { get; set; } = Array.Empty<string>();
}

public class ProjectConfig
{
    public string ProjectRoot { get; set; } = string.Empty;
}