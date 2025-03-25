using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json.Serialization;
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

    public async Task<DiagnosticsResult> RunInitialDiagnosticsAsync(bool verbose = false)
    {
        try
        {
            _logger.LogOperationStarted("diagnostics");
            
            if (verbose)
            {
                Console.WriteLine("Starting diagnostics...");
                Console.WriteLine("Calling RunInitialDiagnosticsAsync...");
            }
            
            // Run system diagnostics
            Console.WriteLine("Starting system diagnostics check...");
            
            // Get system information
            Console.WriteLine("Checking system information...");
            var systemInfo = GetSystemInfo();
            
            // Check Ollama configuration
            Console.WriteLine("Checking Ollama configuration...");
            var ollamaConfig = GetOllamaConfig();
            
            // Check required models
            Console.WriteLine("Checking required models availability...");
            var modelStatus = await CheckRequiredModelsAsync();
            
            // Check project configuration
            Console.WriteLine("Checking project configuration...");
            var projectConfig = GetProjectConfig();
            
            if (verbose)
            {
                Console.WriteLine("Diagnostics completed.");
                Console.WriteLine("RunInitialDiagnosticsAsync completed.");
            }
            
            _logger.LogOperationCompleted("diagnostics");
            
            return new DiagnosticsResult
            {
                SystemInfo = systemInfo,
                OllamaConfig = ollamaConfig,
                ModelStatus = modelStatus,
                ProjectConfig = projectConfig
                // IsReady is calculated automatically based on ModelStatus
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running diagnostics");
            
            // Use CliSupport for colored output
            CliSupport.WriteColorLine($"Critical error running diagnostics: {ex.Message}", ConsoleColor.Red);
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
            
            // Return a minimal result instead of throwing
            return new DiagnosticsResult
            {
                SystemInfo = new SystemInfo { OperatingSystem = "Error", ProcessorCores = 0, AvailableMemoryGB = 0 },
                OllamaConfig = new OllamaConfig { BaseUrl = "Error", DefaultModel = "Error" },
                ModelStatus = new Dictionary<string, bool>(),
                ProjectConfig = new ProjectConfig { ProjectRoot = "Error" }
            };
        }
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
                var output = process.StandardOutput.ReadToEnd();
                process.WaitForExit();

                var match = System.Text.RegularExpressions.Regex.Match(output, @"FreePhysicalMemory=(\d+)");
                if (match.Success && long.TryParse(match.Groups[1].Value, out var memoryKB))
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
                var output = process.StandardOutput.ReadToEnd();
                process.WaitForExit();

                var lines = output.Split('\n');
                if (lines.Length > 1)
                {
                    var memInfo = lines[1].Split([' '], StringSplitOptions.RemoveEmptyEntries);
                    if (memInfo.Length > 6 && int.TryParse(memInfo[3], out var memoryMB))
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
                var output = process.StandardOutput.ReadToEnd();
                process.WaitForExit();

                if (long.TryParse(output.Trim(), out var totalBytes))
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
            requiredModels = [_configuration["Ollama:DefaultModel"] ?? "codellama:13b-code"];
        }
        
        // Get list of installed models
        var installedModels = await GetInstalledModelsAsync();
        
        foreach (var model in requiredModels)
        {
            try
            {
                var modelName = model.Trim();
                
                // Check if the exact model is installed
                if (installedModels.Contains(modelName))
                {
                    result[modelName] = true;
                    continue;
                }
                
                // Check for alternative embedding models
                if ((modelName.Contains("all-minilm") || modelName.Contains("nomic-embed")) && 
                    installedModels.Any(m => m.Contains("all-minilm") || m.Contains("nomic-embed") || 
                                            m.Contains("gte-small") || m.Contains("e5-small")))
                {
                    result[modelName] = true;
                    continue;
                }
                
                // If not found in installed models, try a test generation
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

    public async Task<bool> RunPowerShellScript(string scriptPath, string arguments = "")
    {
        try
        {
            // Check if PowerShell Core is installed
            var isPwshInstalled = IsPowerShellCoreInstalled();
            var psExecutable = isPwshInstalled ? "pwsh" : "powershell.exe";
            
            if (!isPwshInstalled)
            {
                _logger.LogWarning("PowerShell Core (pwsh) is not installed. Using Windows PowerShell instead.");
                _logger.LogInformation("For better cross-platform compatibility, consider installing PowerShell Core:");
                _logger.LogInformation("https://github.com/PowerShell/PowerShell#get-powershell");
            }
            
            var startInfo = new ProcessStartInfo
            {
                FileName = psExecutable,
                Arguments = $"-ExecutionPolicy Bypass -File \"{scriptPath}\" {arguments}",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = false
            };

            using var process = Process.Start(startInfo);
            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();
            
            await process.WaitForExitAsync();
            
            _logger.LogInformation($"PowerShell Output: {output}");
            if (!string.IsNullOrEmpty(error))
                _logger.LogError($"PowerShell Error: {error}");
                
            return process.ExitCode == 0;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Failed to run PowerShell script: {scriptPath}");
            return false;
        }
    }

    public bool IsPowerShellCoreInstalled()
    {
        try
        {
            using var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "pwsh",
                    Arguments = "-Command \"exit\"",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };
            
            process.Start();
            process.WaitForExit();
            return true;
        }
        catch
        {
            return false;
        }
    }

    private async Task<List<string>> GetInstalledModelsAsync()
    {
        try
        {
            var baseUrl = _configuration["Ollama:BaseUrl"] ?? "http://localhost:11434";
            using var httpClient = new HttpClient();
            var response = await httpClient.GetAsync($"{baseUrl}/api/tags");
            
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync();
                var tagsResponse = System.Text.Json.JsonSerializer.Deserialize<TagsResponse>(content);
                
                if (tagsResponse?.Models != null)
                {
                    return tagsResponse.Models.Select(m => m.Name).ToList();
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting installed models");
        }
        
        return new List<string>();
    }

    private class TagsResponse
    {
        [JsonPropertyName("models")]
        public List<ModelInfo> Models { get; set; } = new();
    }

    private class ModelInfo
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = "";
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
    public string[] RequiredModels { get; set; } = [];
}

public class ProjectConfig
{
    public string ProjectRoot { get; set; } = string.Empty;
}