using System.Diagnostics;
using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis.CSharp.Scripting;
using Microsoft.CodeAnalysis.Scripting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Interfaces.Compilation;

namespace TarsEngine.Services;

/// <summary>
/// Service for executing metascripts in a sandbox environment
/// </summary>
public class MetascriptSandbox
{
    private readonly ILogger<MetascriptSandbox> _logger;
    private readonly string _sandboxDirectory;
    private readonly Dictionary<string, ScriptOptions> _languageOptions;
    private readonly Dictionary<string, Func<string, Dictionary<string, object>, CancellationToken, Task<object>>> _languageExecutors;
    private readonly IFSharpCompiler _fsharpCompiler;

    /// <summary>
    /// Initializes a new instance of the <see cref="MetascriptSandbox"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="configuration">The configuration</param>
    /// <param name="fsharpCompiler">The F# compiler service</param>
    public MetascriptSandbox(
        ILogger<MetascriptSandbox> logger,
        IConfiguration configuration,
        IFSharpCompiler fsharpCompiler) // Ensure IFSharpCompiler is declared in the correct namespace
    {
        _logger = logger;
        _fsharpCompiler = fsharpCompiler;
        _sandboxDirectory = configuration.GetValue<string>("SandboxDirectory") ?? Path.Combine(Path.GetTempPath(), "TarsScriptSandbox");
        _languageOptions = new Dictionary<string, ScriptOptions>();
        _languageExecutors = new Dictionary<string, Func<string, Dictionary<string, object>, CancellationToken, Task<object>>>();
        
        InitializeLanguageOptions();
        InitializeLanguageExecutors();
    }

    /// <summary>
    /// Executes a metascript in the sandbox
    /// </summary>
    /// <param name="metascript">The metascript to execute</param>
    /// <param name="context">The execution context</param>
    /// <param name="options">Optional execution options</param>
    /// <param name="cancellationToken">The cancellation token</param>
    /// <returns>The execution result</returns>
    public async Task<MetascriptExecutionResult> ExecuteMetascriptAsync(
        GeneratedMetascript metascript,
        Dictionary<string, object> context,
        Dictionary<string, string>? options = null,
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("Executing metascript: {MetascriptName} ({MetascriptId})", metascript.Name, metascript.Id);

            // Create execution result
            var result = new MetascriptExecutionResult
            {
                MetascriptId = metascript.Id,
                StartedAt = DateTime.UtcNow,
                Status = MetascriptExecutionStatus.Executing
            };

            // Parse options
            var timeoutMs = ParseOption(options, "TimeoutMs", 30000);
            var memoryLimitMb = ParseOption(options, "MemoryLimitMb", 100);
            var captureOutput = ParseOption(options, "CaptureOutput", true);
            var validateResult = ParseOption(options, "ValidateResult", true);
            var trackPerformance = ParseOption(options, "TrackPerformance", true);

            // Create cancellation token with timeout
            using var timeoutCts = new CancellationTokenSource(timeoutMs);
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(timeoutCts.Token, cancellationToken);

            // Create sandbox context
            var sandboxContext = CreateSandboxContext(metascript, context, options);

            // Execute metascript
            var stopwatch = Stopwatch.StartNew();
            var memoryBefore = GC.GetTotalMemory(true);

            try
            {
                // Get language executor
                if (!_languageExecutors.TryGetValue(metascript.Language.ToLowerInvariant(), out var executor))
                {
                    throw new NotSupportedException($"Language not supported: {metascript.Language}");
                }

                // Execute metascript
                var executionResult = await executor(metascript.Code, sandboxContext, linkedCts.Token);

                // Process execution result
                result.IsSuccessful = true;
                result.Status = MetascriptExecutionStatus.Succeeded;
                result.Output = executionResult?.ToString();

                // Extract changes from execution result
                if (executionResult is Dictionary<string, object> resultDict)
                {
                    if (resultDict.TryGetValue("Changes", out var changesObj) && changesObj is List<MetascriptChange> changes)
                    {
                        result.Changes = changes;
                        result.AffectedFiles = changes.Select(c => c.FilePath).Distinct().ToList();
                    }
                }
            }
            catch (OperationCanceledException)
            {
                result.IsSuccessful = false;
                result.Status = timeoutCts.Token.IsCancellationRequested
                    ? MetascriptExecutionStatus.TimedOut
                    : MetascriptExecutionStatus.Cancelled;
                result.Error = $"Execution {result.Status.ToString().ToLowerInvariant()}";
            }
            catch (Exception ex)
            {
                result.IsSuccessful = false;
                result.Status = MetascriptExecutionStatus.Failed;
                result.Error = ex.ToString();
            }
            finally
            {
                stopwatch.Stop();
                result.ExecutionTimeMs = stopwatch.ElapsedMilliseconds;
                result.CompletedAt = DateTime.UtcNow;

                if (trackPerformance)
                {
                    var memoryAfter = GC.GetTotalMemory(false);
                    var memoryUsed = memoryAfter - memoryBefore;
                    result.Metadata["MemoryUsedBytes"] = memoryUsed.ToString();
                }
            }

            // Validate result
            if (validateResult)
            {
                ValidateExecutionResult(result, metascript, options);
            }

            _logger.LogInformation("Metascript execution completed: {MetascriptName} ({MetascriptId}), Status: {Status}, Time: {ExecutionTimeMs}ms",
                metascript.Name, metascript.Id, result.Status, result.ExecutionTimeMs);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing metascript: {MetascriptName} ({MetascriptId})", metascript.Name, metascript.Id);

            return new MetascriptExecutionResult
            {
                MetascriptId = metascript.Id,
                IsSuccessful = false,
                Status = MetascriptExecutionStatus.Failed,
                Error = ex.ToString(),
                StartedAt = DateTime.UtcNow,
                CompletedAt = DateTime.UtcNow
            };
        }
    }

    /// <summary>
    /// Validates a metascript
    /// </summary>
    /// <param name="metascript">The metascript to validate</param>
    /// <param name="options">Optional validation options</param>
    /// <returns>The validation result</returns>
    public async Task<MetascriptValidationResult> ValidateMetascriptAsync(
        GeneratedMetascript metascript,
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Validating metascript: {MetascriptName} ({MetascriptId})", metascript.Name, metascript.Id);

            var startTime = DateTime.UtcNow;
            var stopwatch = Stopwatch.StartNew();

            // Create validation result
            var result = new MetascriptValidationResult
            {
                MetascriptId = metascript.Id,
                Status = MetascriptValidationStatus.Validating
            };

            // Validate code based on language
            var (isValid, errors, warnings) = await ValidateCodeAsync(metascript, options);

            stopwatch.Stop();

            result.IsValid = isValid;
            result.Errors = errors;
            result.Warnings = warnings;
            result.Status = isValid ? MetascriptValidationStatus.Valid : MetascriptValidationStatus.Invalid;
            result.ValidationTimeMs = stopwatch.ElapsedMilliseconds;

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating metascript: {MetascriptName} ({MetascriptId})", metascript.Name, metascript.Id);
            return new MetascriptValidationResult
            {
                MetascriptId = metascript.Id,
                IsValid = false,
                Status = MetascriptValidationStatus.Invalid,
                Errors = [ex.ToString()]
            };
        }
    }

    /// <summary>
    /// Gets the supported metascript languages
    /// </summary>
    /// <returns>The list of supported languages</returns>
    public List<string> GetSupportedLanguages()
    {
        return _languageExecutors.Keys.ToList();
    }

    /// <summary>
    /// Gets the available execution options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    public Dictionary<string, string> GetAvailableOptions()
    {
        return new Dictionary<string, string>
        {
            { "TimeoutMs", "Execution timeout in milliseconds (default: 30000)" },
            { "MemoryLimitMb", "Memory limit in megabytes (default: 100)" },
            { "CaptureOutput", "Whether to capture output (default: true)" },
            { "ValidateResult", "Whether to validate the execution result (default: true)" },
            { "TrackPerformance", "Whether to track performance metrics (default: true)" },
            { "AllowFileAccess", "Whether to allow file access (default: false)" },
            { "AllowNetworkAccess", "Whether to allow network access (default: false)" },
            { "AllowProcessExecution", "Whether to allow process execution (default: false)" },
            { "AllowReflection", "Whether to allow reflection (default: false)" },
            { "SandboxMode", "Sandbox mode (Strict, Standard, Permissive) (default: Strict)" }
        };
    }

    /// <summary>
    /// Creates a sandbox context for metascript execution
    /// </summary>
    /// <param name="metascript">The metascript</param>
    /// <param name="context">The execution context</param>
    /// <param name="options">The execution options</param>
    /// <returns>The sandbox context</returns>
    private Dictionary<string, object> CreateSandboxContext(
        GeneratedMetascript metascript,
        Dictionary<string, object> context,
        Dictionary<string, string>? options)
    {
        var sandboxContext = new Dictionary<string, object>(context);

        // Add metascript parameters
        foreach (var parameter in metascript.Parameters)
        {
            sandboxContext[parameter.Key] = parameter.Value;
        }

        // Add sandbox utilities
        sandboxContext["Logger"] = _logger;
        sandboxContext["MetascriptId"] = metascript.Id;
        sandboxContext["MetascriptName"] = metascript.Name;
        sandboxContext["ExecutionTime"] = DateTime.UtcNow;
        sandboxContext["SandboxDirectory"] = _sandboxDirectory;

        // Add sandbox permissions based on options
        var allowFileAccess = ParseOption(options, "AllowFileAccess", false);
        var allowNetworkAccess = ParseOption(options, "AllowNetworkAccess", false);
        var allowProcessExecution = ParseOption(options, "AllowProcessExecution", false);
        var allowReflection = ParseOption(options, "AllowReflection", false);

        sandboxContext["AllowFileAccess"] = allowFileAccess;
        sandboxContext["AllowNetworkAccess"] = allowNetworkAccess;
        sandboxContext["AllowProcessExecution"] = allowProcessExecution;
        sandboxContext["AllowReflection"] = allowReflection;

        // Add sandbox utilities
        if (allowFileAccess)
        {
            sandboxContext["FileSystem"] = new SandboxFileSystem(_sandboxDirectory, _logger);
        }

        return sandboxContext;
    }

    /// <summary>
    /// Validates a metascript execution result
    /// </summary>
    /// <param name="result">The execution result</param>
    /// <param name="metascript">The metascript</param>
    /// <param name="options">The execution options</param>
    private void ValidateExecutionResult(
        MetascriptExecutionResult result,
        GeneratedMetascript metascript,
        Dictionary<string, string>? options)
    {
        try
        {
            // Validate execution status
            if (result.Status != MetascriptExecutionStatus.Succeeded)
            {
                return;
            }

            // Validate changes
            if (result.Changes != null)
            {
                foreach (var change in result.Changes)
                {
                    // Validate file path
                    if (string.IsNullOrEmpty(change.FilePath))
                    {
                        result.Metadata["ValidationWarning"] = "Change has empty file path";
                    }
                    else if (!Path.IsPathRooted(change.FilePath) && !change.FilePath.Contains('/') && !change.FilePath.Contains('\\'))
                    {
                        result.Metadata["ValidationWarning"] = $"Change has invalid file path: {change.FilePath}";
                    }

                    // Validate content
                    if (change.Type == MetascriptChangeType.Modification && string.IsNullOrEmpty(change.NewContent))
                    {
                        result.Metadata["ValidationWarning"] = "Modification change has empty new content";
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating execution result for metascript: {MetascriptName} ({MetascriptId})", metascript.Name, metascript.Id);
            result.Metadata["ValidationError"] = ex.ToString();
        }
    }

    /// <summary>
    /// Validates metascript code
    /// </summary>
    /// <param name="metascript">The metascript to validate</param>
    /// <param name="options">Optional validation options</param>
    /// <returns>Validation result tuple (isValid, errors, warnings)</returns>
    private Task<(bool IsValid, List<string> Errors, List<string> Warnings)> ValidateCodeAsync(
        GeneratedMetascript metascript,
        Dictionary<string, string>? options)
    {
        var errors = new List<string>();
        var warnings = new List<string>();

        try
        {
            // Validate metascript code
            return ValidateMetascriptCodeAsync(metascript, options);
        }
        catch (Exception ex)
        {
            errors.Add($"Error validating code: {ex.Message}");
            return Task.FromResult((false, errors, warnings));
        }
    }

    /// <summary>
    /// Validates metascript code
    /// </summary>
    /// <param name="metascript">The metascript to validate</param>
    /// <param name="options">Optional validation options</param>
    /// <returns>Validation result tuple (isValid, errors, warnings)</returns>
    private Task<(bool IsValid, List<string> Errors, List<string> Warnings)> ValidateMetascriptCodeAsync(
        GeneratedMetascript metascript,
        Dictionary<string, string>? options)
    {
        var errors = new List<string>();
        var warnings = new List<string>();

        try
        {
            // Validate code based on language
            switch (metascript.Language.ToLowerInvariant())
            {
                case "csharp":
                    return ValidateCSharpCodeAsync(metascript.Code, options);

                case "fsharp":
                    return ValidateFSharpCodeAsync(metascript.Code, options);

                case "meta":
                    return ValidateMetaCodeAsync(metascript.Code, options);

                default:
                    warnings.Add($"No specific validation for language: {metascript.Language}");
                    return Task.FromResult((true, errors, warnings));
            }
        }
        catch (Exception ex)
        {
            errors.Add($"Error validating metascript code: {ex.Message}");
            return Task.FromResult((false, errors, warnings));
        }
    }

    /// <summary>
    /// Validates C# code
    /// </summary>
    /// <param name="code">The code</param>
    /// <param name="options">The validation options</param>
    /// <returns>The validation result</returns>
    private Task<(bool IsValid, List<string> Errors, List<string> Warnings)> ValidateCSharpCodeAsync(
        string code,
        Dictionary<string, string>? options)
    {
        var errors = new List<string>();
        var warnings = new List<string>();

        try
        {
            // Create script options
            var scriptOptions = _languageOptions["csharp"];

            // Try to compile the code
            var script = CSharpScript.Create(code, scriptOptions);
            var diagnostics = script.Compile();

            // Process diagnostics
            foreach (var diagnostic in diagnostics)
            {
                if (diagnostic.Severity == Microsoft.CodeAnalysis.DiagnosticSeverity.Error)
                {
                    errors.Add($"Error {diagnostic.Id}: {diagnostic.GetMessage()}");
                }
                else if (diagnostic.Severity == Microsoft.CodeAnalysis.DiagnosticSeverity.Warning)
                {
                    warnings.Add($"Warning {diagnostic.Id}: {diagnostic.GetMessage()}");
                }
            }

            return Task.FromResult((errors.Count == 0, errors, warnings));
        }
        catch (Exception ex)
        {
            errors.Add($"Error validating C# code: {ex.Message}");
            return Task.FromResult((false, errors, warnings));
        }
    }

    /// <summary>
    /// Validates F# code
    /// </summary>
    /// <param name="code">The code to validate</param>
    /// <param name="options">Optional validation options</param>
    /// <returns>A tuple containing validation result, errors, and warnings</returns>
    private Task<(bool IsValid, List<string> Errors, List<string> Warnings)> ValidateFSharpCodeAsync(
        string code,
        Dictionary<string, string>? options)
    {
        return Task.Run(async () =>
        {
            var errors = new List<string>();
            var warnings = new List<string>();

            try
            {
                // Try to compile the code using F# compiler service
                var scriptOptions = _languageOptions["fsharp"];
                var result = await _fsharpCompiler.CompileAsync(code, scriptOptions);
                
                foreach (var diagnostic in result.Diagnostics)
                {
                    if (diagnostic.IsError)
                    {
                        errors.Add($"Error: {diagnostic.Message}");
                    }
                    else if (diagnostic.IsWarning)
                    {
                        warnings.Add($"Warning: {diagnostic.Message}");
                    }
                }

                // Check for unsafe operations
                if (code.Contains("System.Reflection") || 
                    code.Contains("System.Runtime") ||
                    code.Contains("System.Diagnostics.Process"))
                {
                    warnings.Add("Code contains potentially unsafe operations");
                }

                return (errors.Count == 0, errors, warnings);
            }
            catch (Exception ex)
            {
                errors.Add($"Error validating F# code: {ex.Message}");
                return (false, errors, warnings);
            }
        });
    }

    /// <summary>
    /// Validates Meta code
    /// </summary>
    /// <param name="code">The code to validate</param>
    /// <param name="options">Optional validation options</param>
    /// <returns>A tuple containing validation result, errors, and warnings</returns>
    private Task<(bool IsValid, List<string> Errors, List<string> Warnings)> ValidateMetaCodeAsync(
        string code,
        Dictionary<string, string>? options)
    {
        return Task.Run(() =>
        {
            var errors = new List<string>();
            var warnings = new List<string>();

            try
            {
                // Check if code is empty
                if (string.IsNullOrWhiteSpace(code))
                {
                    errors.Add("Meta code is empty");
                    return (false, errors, warnings);
                }

                // Validate meta code structure
                if (!code.Contains("@meta"))
                {
                    errors.Add("Missing @meta directive");
                }

                // Validate meta version
                var versionMatch = Regex.Match(code, @"@version\s+(\d+\.\d+)");
                if (!versionMatch.Success)
                {
                    errors.Add("Missing or invalid @version directive");
                }

                // Check for potential security issues
                if (code.Contains("System.Diagnostics.Process") || 
                    code.Contains("System.IO.File") ||
                    code.Contains("System.Net.WebClient"))
                {
                    warnings.Add("Code contains potentially unsafe operations that require explicit permissions");
                }

                return (errors.Count == 0, errors, warnings);
            }
            catch (Exception ex)
            {
                errors.Add($"Error validating Meta code: {ex.Message}");
                return (false, errors, warnings);
            }
        });
    }

    /// <summary>
    /// Initializes language options
    /// </summary>
    private void InitializeLanguageOptions()
    {
        try
        {
            // C# options
            _languageOptions["csharp"] = ScriptOptions.Default
                .WithReferences(
                    typeof(File).Assembly,
                    typeof(Enumerable).Assembly,
                    typeof(Regex).Assembly)
                .WithImports(
                    "System",
                    "System.IO",
                    "System.Linq",
                    "System.Collections.Generic",
                    "System.Text",
                    "System.Text.RegularExpressions");

            // F# options (placeholder)
            _languageOptions["fsharp"] = ScriptOptions.Default;

            // Meta options (placeholder)
            _languageOptions["meta"] = ScriptOptions.Default;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing language options");
        }
    }

    /// <summary>
    /// Initializes language executors
    /// </summary>
    private void InitializeLanguageExecutors()
    {
        try
        {
            // C# executor
            _languageExecutors["csharp"] = async (code, context, cancellationToken) =>
            {
                var scriptOptions = _languageOptions["csharp"];
                var script = CSharpScript.Create(code, scriptOptions, globalsType: typeof(Dictionary<string, object>));
                var result = await script.RunAsync(context, cancellationToken: cancellationToken);
                return result.ReturnValue;
            };

            // F# executor (placeholder)
            _languageExecutors["fsharp"] = (code, context, cancellationToken) =>
                Task.FromException<object>(new NotImplementedException("F# execution not implemented"));

            // Meta executor (placeholder)
            _languageExecutors["meta"] = (code, context, cancellationToken) =>
                Task.FromException<object>(new NotImplementedException("Meta execution not implemented"));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing language executors");
        }
    }

    /// <summary>
    /// Parses an option value
    /// </summary>
    /// <typeparam name="T">The option type</typeparam>
    /// <param name="options">The options dictionary</param>
    /// <param name="key">The option key</param>
    /// <param name="defaultValue">The default value</param>
    /// <returns>The option value</returns>
    private T ParseOption<T>(Dictionary<string, string>? options, string key, T defaultValue)
    {
        if (options == null || !options.TryGetValue(key, out var value))
        {
            return defaultValue;
        }

        try
        {
            return (T)Convert.ChangeType(value, typeof(T));
        }
        catch
        {
            return defaultValue;
        }
    }

    /// <summary>
    /// Sandbox file system for metascript execution
    /// </summary>
    private class SandboxFileSystem
    {
        private readonly string _rootDirectory;
        private readonly ILogger _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="SandboxFileSystem"/> class
        /// </summary>
        /// <param name="rootDirectory">The root directory</param>
        /// <param name="logger">The logger</param>
        public SandboxFileSystem(string rootDirectory, ILogger logger)
        {
            _rootDirectory = rootDirectory;
            _logger = logger;
        }

        /// <summary>
        /// Reads a file
        /// </summary>
        /// <param name="path">The file path</param>
        /// <returns>The file content</returns>
        public string ReadFile(string path)
        {
            var fullPath = GetFullPath(path);
            return File.ReadAllText(fullPath);
        }

        /// <summary>
        /// Writes a file
        /// </summary>
        /// <param name="path">The file path</param>
        /// <param name="content">The file content</param>
        public void WriteFile(string path, string content)
        {
            var fullPath = GetFullPath(path);
            var directory = Path.GetDirectoryName(fullPath);
            
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory!);
            }
            
            File.WriteAllText(fullPath, content);
        }

        /// <summary>
        /// Checks if a file exists
        /// </summary>
        /// <param name="path">The file path</param>
        /// <returns>True if the file exists, false otherwise</returns>
        public bool FileExists(string path)
        {
            var fullPath = GetFullPath(path);
            return File.Exists(fullPath);
        }

        /// <summary>
        /// Lists files in a directory
        /// </summary>
        /// <param name="path">The directory path</param>
        /// <param name="searchPattern">The search pattern</param>
        /// <returns>The list of file paths</returns>
        public List<string> ListFiles(string path, string searchPattern = "*")
        {
            var fullPath = GetFullPath(path);
            return Directory.GetFiles(fullPath, searchPattern)
                .Select(p => p.Substring(_rootDirectory.Length).TrimStart('\\', '/'))
                .ToList();
        }

        /// <summary>
        /// Gets the full path for a relative path
        /// </summary>
        /// <param name="path">The relative path</param>
        /// <returns>The full path</returns>
        private string GetFullPath(string path)
        {
            // Normalize path
            path = path.Replace('\\', '/').TrimStart('/');
            
            // Ensure path is within sandbox
            var fullPath = Path.Combine(_rootDirectory, path);
            var normalizedFullPath = Path.GetFullPath(fullPath);
            
            if (!normalizedFullPath.StartsWith(_rootDirectory))
            {
                throw new UnauthorizedAccessException($"Access to path outside sandbox is not allowed: {path}");
            }
            
            return normalizedFullPath;
        }
    }
}
