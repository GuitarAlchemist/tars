using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsEngine.Services.Compilation
{
    /// <summary>
    /// Executes F# scripts using the F# Interactive (FSI) tool.
    /// </summary>
    public class FSharpScriptExecutor
    {
        private readonly ILogger<FSharpScriptExecutor> _logger;
        private readonly string _tempDirectory;
        private readonly string _fsiPath;

        /// <summary>
        /// Initializes a new instance of the <see cref="FSharpScriptExecutor"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        public FSharpScriptExecutor(ILogger<FSharpScriptExecutor> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _tempDirectory = Path.Combine(Path.GetTempPath(), "TarsEngine", "FSharpScripts");
            _fsiPath = GetFsiPath();

            // Ensure temp directory exists
            if (!Directory.Exists(_tempDirectory))
            {
                Directory.CreateDirectory(_tempDirectory);
            }

            _logger.LogInformation("FSharpScriptExecutor initialized. Temp directory: {TempDirectory}, FSI path: {FsiPath}", _tempDirectory, _fsiPath);
        }

        /// <summary>
        /// Executes an F# script.
        /// </summary>
        /// <param name="scriptCode">The F# script code to execute.</param>
        /// <returns>The execution result.</returns>
        public async Task<ScriptExecutionResult> ExecuteScriptAsync(string scriptCode)
        {
            return await ExecuteScriptAsync(scriptCode, null);
        }

        /// <summary>
        /// Executes an F# script with the specified arguments.
        /// </summary>
        /// <param name="scriptCode">The F# script code to execute.</param>
        /// <param name="arguments">The arguments to pass to the script.</param>
        /// <returns>The execution result.</returns>
        public async Task<ScriptExecutionResult> ExecuteScriptAsync(string scriptCode, string[] arguments)
        {
            try
            {
                _logger.LogInformation("Executing F# script");

                // Create a temporary file for the F# script
                var scriptFilePath = Path.Combine(_tempDirectory, $"{Guid.NewGuid()}.fsx");
                await File.WriteAllTextAsync(scriptFilePath, scriptCode);

                // Execute the script
                var result = await ExecuteScriptFileAsync(scriptFilePath, arguments);

                // Clean up temporary file
                File.Delete(scriptFilePath);

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error executing F# script");
                return new ScriptExecutionResult
                {
                    Success = false,
                    Output = string.Empty,
                    Errors = new List<string> { $"Error executing F# script: {ex.Message}" }
                };
            }
        }

        /// <summary>
        /// Executes an F# script file.
        /// </summary>
        /// <param name="scriptFilePath">The path to the F# script file.</param>
        /// <returns>The execution result.</returns>
        public async Task<ScriptExecutionResult> ExecuteScriptFileAsync(string scriptFilePath)
        {
            return await ExecuteScriptFileAsync(scriptFilePath, null);
        }

        /// <summary>
        /// Executes an F# script file with the specified arguments.
        /// </summary>
        /// <param name="scriptFilePath">The path to the F# script file.</param>
        /// <param name="arguments">The arguments to pass to the script.</param>
        /// <returns>The execution result.</returns>
        public async Task<ScriptExecutionResult> ExecuteScriptFileAsync(string scriptFilePath, string[] arguments)
        {
            try
            {
                _logger.LogInformation("Executing F# script file: {ScriptFilePath}", scriptFilePath);

                if (!File.Exists(scriptFilePath))
                {
                    throw new FileNotFoundException($"F# script file not found: {scriptFilePath}");
                }

                // Build the arguments
                var args = new StringBuilder();
                args.Append("--quiet ");
                args.Append($"\"{scriptFilePath}\" ");

                if (arguments != null && arguments.Length > 0)
                {
                    args.Append("-- ");
                    foreach (var arg in arguments)
                    {
                        args.Append($"\"{arg}\" ");
                    }
                }

                // Create process start info
                var startInfo = new ProcessStartInfo
                {
                    FileName = _fsiPath,
                    Arguments = args.ToString(),
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                // Start the process
                using var process = new Process { StartInfo = startInfo };
                var outputBuilder = new StringBuilder();
                var errorBuilder = new StringBuilder();

                process.OutputDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        outputBuilder.AppendLine(e.Data);
                    }
                };

                process.ErrorDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        errorBuilder.AppendLine(e.Data);
                    }
                };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                await process.WaitForExitAsync();

                // Get the output and error
                var output = outputBuilder.ToString();
                var error = errorBuilder.ToString();

                _logger.LogInformation("F# script execution output: {Output}", output);

                if (!string.IsNullOrEmpty(error))
                {
                    _logger.LogWarning("F# script execution error: {Error}", error);
                }

                // Parse errors
                var errors = new List<string>();
                if (!string.IsNullOrEmpty(error))
                {
                    foreach (var line in error.Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries))
                    {
                        errors.Add(line);
                    }
                }

                return new ScriptExecutionResult
                {
                    Success = process.ExitCode == 0,
                    Output = output,
                    Errors = errors
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error executing F# script file: {ScriptFilePath}", scriptFilePath);
                return new ScriptExecutionResult
                {
                    Success = false,
                    Output = string.Empty,
                    Errors = new List<string> { $"Error executing F# script file: {ex.Message}" }
                };
            }
        }

        private string GetFsiPath()
        {
            // Try to find fsi.exe in the PATH
            var paths = Environment.GetEnvironmentVariable("PATH")?.Split(Path.PathSeparator);
            if (paths != null)
            {
                foreach (var path in paths)
                {
                    var fsiPath = Path.Combine(path, "fsi.exe");
                    if (File.Exists(fsiPath))
                    {
                        return fsiPath;
                    }
                }
            }

            // Try to find fsi.exe in the .NET SDK directory
            var dotnetPath = GetDotNetPath();
            if (!string.IsNullOrEmpty(dotnetPath))
            {
                var sdkDir = Path.GetDirectoryName(dotnetPath);
                if (sdkDir != null)
                {
                    var fsiPath = Path.Combine(sdkDir, "FSharp", "fsi.exe");
                    if (File.Exists(fsiPath))
                    {
                        return fsiPath;
                    }
                }
            }

            // Use dotnet fsi as a fallback
            return "dotnet fsi";
        }

        private string GetDotNetPath()
        {
            // Try to find dotnet.exe in the PATH
            var paths = Environment.GetEnvironmentVariable("PATH")?.Split(Path.PathSeparator);
            if (paths != null)
            {
                foreach (var path in paths)
                {
                    var dotnetPath = Path.Combine(path, "dotnet.exe");
                    if (File.Exists(dotnetPath))
                    {
                        return dotnetPath;
                    }
                }
            }

            return "dotnet";
        }
    }

    /// <summary>
    /// Represents the result of executing an F# script.
    /// </summary>
    public class ScriptExecutionResult
    {
        /// <summary>
        /// Gets or sets a value indicating whether the script execution was successful.
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Gets or sets the output of the script execution.
        /// </summary>
        public string Output { get; set; }

        /// <summary>
        /// Gets or sets the errors that occurred during script execution.
        /// </summary>
        public List<string> Errors { get; set; }
    }
}
