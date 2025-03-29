using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Diagnostics;
using System.Text;

namespace TarsCli.Services;

public class ScriptExecutionService
{
    private readonly ILogger<ScriptExecutionService> _logger;
    private readonly IConfiguration _configuration;
    private readonly string _projectRoot;

    public ScriptExecutionService(
        ILogger<ScriptExecutionService> logger,
        IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;
        _projectRoot = _configuration["Tars:ProjectRoot"] ?? 
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), 
                "source", "repos", "tars");
    }

    public async Task<(bool Success, string Output)> ExecuteScriptAsync(string sessionName, string scriptName)
    {
        try
        {
            _logger.LogInformation($"Executing script '{scriptName}' in session '{sessionName}'");
            
            // Check if session exists
            var sessionDir = Path.Combine(_projectRoot, "sessions", sessionName);
            
            if (!Directory.Exists(sessionDir))
            {
                _logger.LogError($"Session '{sessionName}' does not exist");
                return (false, $"Session '{sessionName}' does not exist. Use 'tarscli init {sessionName}' to create it.");
            }
            
            // Check if script file exists
            var scriptPath = Path.Combine(sessionDir, "plans", scriptName);
            
            if (!File.Exists(scriptPath))
            {
                _logger.LogError($"Script file '{scriptName}' does not exist in session '{sessionName}'");
                return (false, $"Script file '{scriptName}' does not exist in session '{sessionName}'.");
            }
            
            // Create a log file for this run
            var logDir = Path.Combine(sessionDir, "logs");
            var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            var logFile = Path.Combine(logDir, $"run_{timestamp}.log");
            
            // Execute the script using dotnet-script
            var outputBuilder = new StringBuilder();
            outputBuilder.AppendLine($"=== TARS Script Execution Log ===");
            outputBuilder.AppendLine($"Session: {sessionName}");
            outputBuilder.AppendLine($"Script: {scriptName}");
            outputBuilder.AppendLine($"Started: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}");
            outputBuilder.AppendLine();
            
            var startTime = DateTime.UtcNow;
            
            // Check if dotnet-script is installed
            var dotnetScriptInstalled = await CheckDotnetScriptInstalledAsync();
            
            if (!dotnetScriptInstalled)
            {
                _logger.LogWarning("dotnet-script is not installed. Installing...");
                outputBuilder.AppendLine("[INFO] dotnet-script is not installed. Installing...");
                
                var installResult = await InstallDotnetScriptAsync();
                
                if (!installResult)
                {
                    _logger.LogError("Failed to install dotnet-script");
                    outputBuilder.AppendLine("[ERROR] Failed to install dotnet-script");
                    await File.WriteAllTextAsync(logFile, outputBuilder.ToString());
                    return (false, "Failed to install dotnet-script. Please install it manually using 'dotnet tool install -g dotnet-script'.");
                }
                
                outputBuilder.AppendLine("[INFO] dotnet-script installed successfully");
            }
            
            // Execute the script
            outputBuilder.AppendLine($"[INFO] Executing script: {scriptPath}");
            
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "dotnet",
                    Arguments = $"script \"{scriptPath}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = sessionDir
                }
            };
            
            process.Start();
            
            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();
            
            await process.WaitForExitAsync();
            
            var endTime = DateTime.UtcNow;
            var executionTime = (endTime - startTime).TotalSeconds;
            
            outputBuilder.AppendLine(output);
            
            if (!string.IsNullOrEmpty(error))
            {
                outputBuilder.AppendLine("[ERROR] Script execution failed with errors:");
                outputBuilder.AppendLine(error);
            }
            
            outputBuilder.AppendLine();
            outputBuilder.AppendLine($"[INFO] Execution time: {executionTime:F1} seconds");
            outputBuilder.AppendLine();
            outputBuilder.AppendLine("=== End of Log ===");
            
            // Save the log
            await File.WriteAllTextAsync(logFile, outputBuilder.ToString());
            
            _logger.LogInformation($"Script execution completed. Log saved to {logFile}");
            
            return (process.ExitCode == 0, outputBuilder.ToString());
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing script '{scriptName}' in session '{sessionName}'");
            return (false, $"Error executing script: {ex.Message}");
        }
    }

    private async Task<bool> CheckDotnetScriptInstalledAsync()
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "dotnet",
                    Arguments = "tool list -g",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true
                }
            };
            
            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync();
            await process.WaitForExitAsync();
            
            return output.Contains("dotnet-script");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking if dotnet-script is installed");
            return false;
        }
    }

    private async Task<bool> InstallDotnetScriptAsync()
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "dotnet",
                    Arguments = "tool install -g dotnet-script",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }
            };
            
            process.Start();
            await process.WaitForExitAsync();
            
            return process.ExitCode == 0;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error installing dotnet-script");
            return false;
        }
    }
}
