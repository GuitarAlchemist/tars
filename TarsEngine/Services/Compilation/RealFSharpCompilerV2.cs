using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis.Scripting;
using Microsoft.Extensions.Logging;
using TarsEngine.Interfaces.Compilation;

namespace TarsEngine.Services.Compilation
{
    /// <summary>
    /// Real implementation of the <see cref="IFSharpCompiler"/> interface using the F# compiler service.
    /// </summary>
    public class RealFSharpCompilerV2 : IFSharpCompiler
    {
        private readonly ILogger<RealFSharpCompilerV2> _logger;
        private readonly string _tempDirectory;
        private readonly string _fscPath;

        /// <summary>
        /// Initializes a new instance of the <see cref="RealFSharpCompilerV2"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        public RealFSharpCompilerV2(ILogger<RealFSharpCompilerV2> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _tempDirectory = Path.Combine(Path.GetTempPath(), "TarsEngine", "FSharpCompilation");
            _fscPath = GetFscPath();

            // Ensure temp directory exists
            if (!Directory.Exists(_tempDirectory))
            {
                Directory.CreateDirectory(_tempDirectory);
            }

            _logger.LogInformation("RealFSharpCompilerV2 initialized. Temp directory: {TempDirectory}, FSC path: {FscPath}", _tempDirectory, _fscPath);
        }

        /// <inheritdoc/>
        public async Task<CompilationResult> CompileAsync(string code, ScriptOptions options)
        {
            try
            {
                _logger.LogInformation("Compiling F# code with script options");

                // Create a temporary file for the F# code
                var tempFilePath = Path.Combine(_tempDirectory, $"{Guid.NewGuid()}.fs");
                await File.WriteAllTextAsync(tempFilePath, code);

                // Output assembly path
                var outputPath = Path.Combine(_tempDirectory, $"{Guid.NewGuid()}.dll");

                // Create F# compiler arguments
                var compilerArgs = new List<string>
                {
                    "--nologo",
                    "--target:library",
                    $"--out:{outputPath}",
                    tempFilePath
                };

                // Add references from script options
                if (options?.MetadataReferences != null)
                {
                    foreach (var reference in options.MetadataReferences)
                    {
                        if (reference.Display != null)
                        {
                            compilerArgs.Add($"--reference:{reference.Display}");
                        }
                    }
                }

                // Add default references
                var defaultReferences = GetDefaultReferences();
                foreach (var reference in defaultReferences)
                {
                    compilerArgs.Add($"--reference:{reference}");
                }

                // Run the F# compiler
                var (success, output, errors) = await RunFSharpCompilerAsync(compilerArgs);

                // Clean up temporary file
                File.Delete(tempFilePath);

                // Create diagnostics from errors
                var diagnostics = new List<CompilationDiagnostic>();
                foreach (var error in errors)
                {
                    diagnostics.Add(new CompilationDiagnostic
                    {
                        IsError = true,
                        Message = error
                    });
                }

                // Return the compilation result
                return new CompilationResult
                {
                    Success = success,
                    CompiledAssembly = success ? Assembly.LoadFrom(outputPath) : null,
                    Diagnostics = diagnostics
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error compiling F# code");
                return new CompilationResult
                {
                    Success = false,
                    Diagnostics = new List<CompilationDiagnostic>
                    {
                        new CompilationDiagnostic
                        {
                            IsError = true,
                            Message = $"Error compiling F# code: {ex.Message}"
                        }
                    }
                };
            }
        }

        /// <inheritdoc/>
        public Task<CompilationResult> CompileAsync(string code)
        {
            return CompileAsync(code, null);
        }

        /// <inheritdoc/>
        public async Task<CompilationResult> CompileAsync(string code, string[] references)
        {
            try
            {
                _logger.LogInformation("Compiling F# code with references");

                // Create a temporary file for the F# code
                var tempFilePath = Path.Combine(_tempDirectory, $"{Guid.NewGuid()}.fs");
                await File.WriteAllTextAsync(tempFilePath, code);

                // Output assembly path
                var outputPath = Path.Combine(_tempDirectory, $"{Guid.NewGuid()}.dll");

                // Create F# compiler arguments
                var compilerArgs = new List<string>
                {
                    "--nologo",
                    "--target:library",
                    $"--out:{outputPath}",
                    tempFilePath
                };

                // Add references
                if (references != null)
                {
                    foreach (var reference in references)
                    {
                        compilerArgs.Add($"--reference:{reference}");
                    }
                }

                // Add default references
                var defaultReferences = GetDefaultReferences();
                foreach (var reference in defaultReferences)
                {
                    compilerArgs.Add($"--reference:{reference}");
                }

                // Run the F# compiler
                var (success, output, errors) = await RunFSharpCompilerAsync(compilerArgs);

                // Clean up temporary file
                File.Delete(tempFilePath);

                // Create errors from output
                var compilationErrors = new List<CompilationError>();
                foreach (var error in errors)
                {
                    compilationErrors.Add(new CompilationError
                    {
                        Message = error
                    });
                }

                // Return the compilation result
                return new CompilationResult
                {
                    Success = success,
                    CompiledAssembly = success ? Assembly.LoadFrom(outputPath) : null,
                    Errors = compilationErrors
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error compiling F# code");
                return new CompilationResult
                {
                    Success = false,
                    Errors = new List<CompilationError>
                    {
                        new CompilationError
                        {
                            Message = $"Error compiling F# code: {ex.Message}"
                        }
                    }
                };
            }
        }

        /// <inheritdoc/>
        public Task<CompilationResult> CompileAsync(string code, string[] references, string outputPath)
        {
            return CompileAsync(code, references, outputPath, false);
        }

        /// <inheritdoc/>
        public async Task<CompilationResult> CompileAsync(string code, string[] references, string outputPath, bool generateExecutable)
        {
            try
            {
                _logger.LogInformation("Compiling F# code with references and output path");

                // Create a temporary file for the F# code
                var tempFilePath = Path.Combine(_tempDirectory, $"{Guid.NewGuid()}.fs");
                await File.WriteAllTextAsync(tempFilePath, code);

                // Create F# compiler arguments
                var compilerArgs = new List<string>
                {
                    "--nologo",
                    generateExecutable ? "--target:exe" : "--target:library",
                    $"--out:{outputPath}",
                    tempFilePath
                };

                // Add references
                if (references != null)
                {
                    foreach (var reference in references)
                    {
                        compilerArgs.Add($"--reference:{reference}");
                    }
                }

                // Add default references
                var defaultReferences = GetDefaultReferences();
                foreach (var reference in defaultReferences)
                {
                    compilerArgs.Add($"--reference:{reference}");
                }

                // Run the F# compiler
                var (success, output, errors) = await RunFSharpCompilerAsync(compilerArgs);

                // Clean up temporary file
                File.Delete(tempFilePath);

                // Create errors from output
                var compilationErrors = new List<CompilationError>();
                foreach (var error in errors)
                {
                    compilationErrors.Add(new CompilationError
                    {
                        Message = error
                    });
                }

                // Return the compilation result
                return new CompilationResult
                {
                    Success = success,
                    CompiledAssembly = success ? Assembly.LoadFrom(outputPath) : null,
                    Errors = compilationErrors
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error compiling F# code");
                return new CompilationResult
                {
                    Success = false,
                    Errors = new List<CompilationError>
                    {
                        new CompilationError
                        {
                            Message = $"Error compiling F# code: {ex.Message}"
                        }
                    }
                };
            }
        }

        /// <inheritdoc/>
        public Task<CompilationResult> CompileAsync(string code, string[] references, string outputPath, bool generateExecutable, string[] defines)
        {
            return CompileAsync(code, references, outputPath, generateExecutable, defines, null);
        }

        /// <inheritdoc/>
        public Task<CompilationResult> CompileAsync(string code, string[] references, string outputPath, bool generateExecutable, string[] defines, string[] sourceFiles)
        {
            return CompileAsync(code, references, outputPath, generateExecutable, defines, sourceFiles, null);
        }

        /// <inheritdoc/>
        public Task<CompilationResult> CompileAsync(string code, string[] references, string outputPath, bool generateExecutable, string[] defines, string[] sourceFiles, string[] resources)
        {
            return CompileAsync(code, references, outputPath, generateExecutable, defines, sourceFiles, resources, null);
        }

        /// <inheritdoc/>
        public async Task<CompilationResult> CompileAsync(string code, string[] references, string outputPath, bool generateExecutable, string[] defines, string[] sourceFiles, string[] resources, string[] otherFlags)
        {
            try
            {
                _logger.LogInformation("Compiling F# code with all options");

                // Create a temporary file for the F# code
                var tempFilePath = Path.Combine(_tempDirectory, $"{Guid.NewGuid()}.fs");
                await File.WriteAllTextAsync(tempFilePath, code);

                // Create F# compiler arguments
                var compilerArgs = new List<string>
                {
                    "--nologo",
                    generateExecutable ? "--target:exe" : "--target:library",
                    $"--out:{outputPath}"
                };

                // Add defines
                if (defines != null)
                {
                    foreach (var define in defines)
                    {
                        compilerArgs.Add($"--define:{define}");
                    }
                }

                // Add references
                if (references != null)
                {
                    foreach (var reference in references)
                    {
                        compilerArgs.Add($"--reference:{reference}");
                    }
                }

                // Add default references
                var defaultReferences = GetDefaultReferences();
                foreach (var reference in defaultReferences)
                {
                    compilerArgs.Add($"--reference:{reference}");
                }

                // Add resources
                if (resources != null)
                {
                    foreach (var resource in resources)
                    {
                        compilerArgs.Add($"--resource:{resource}");
                    }
                }

                // Add other flags
                if (otherFlags != null)
                {
                    compilerArgs.AddRange(otherFlags);
                }

                // Add source files
                compilerArgs.Add(tempFilePath);
                if (sourceFiles != null)
                {
                    compilerArgs.AddRange(sourceFiles);
                }

                // Run the F# compiler
                var (success, output, errors) = await RunFSharpCompilerAsync(compilerArgs);

                // Clean up temporary file
                File.Delete(tempFilePath);

                // Create errors from output
                var compilationErrors = new List<CompilationError>();
                foreach (var error in errors)
                {
                    compilationErrors.Add(new CompilationError
                    {
                        Message = error
                    });
                }

                // Return the compilation result
                return new CompilationResult
                {
                    Success = success,
                    CompiledAssembly = success ? Assembly.LoadFrom(outputPath) : null,
                    Errors = compilationErrors
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error compiling F# code");
                return new CompilationResult
                {
                    Success = false,
                    Errors = new List<CompilationError>
                    {
                        new CompilationError
                        {
                            Message = $"Error compiling F# code: {ex.Message}"
                        }
                    }
                };
            }
        }

        private async Task<(bool Success, string Output, List<string> Errors)> RunFSharpCompilerAsync(List<string> arguments)
        {
            var errors = new List<string>();
            var outputBuilder = new StringBuilder();

            try
            {
                _logger.LogInformation("Running F# compiler with arguments: {Arguments}", string.Join(" ", arguments));

                // Create process start info
                var startInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = _fscPath,
                    Arguments = string.Join(" ", arguments),
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                // Start the process
                using var process = new System.Diagnostics.Process { StartInfo = startInfo };
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
                        errors.Add(e.Data);
                    }
                };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                await process.WaitForExitAsync();

                var output = outputBuilder.ToString();
                _logger.LogInformation("F# compiler output: {Output}", output);

                if (errors.Any())
                {
                    _logger.LogWarning("F# compiler errors: {Errors}", string.Join(Environment.NewLine, errors));
                }

                return (process.ExitCode == 0, output, errors);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running F# compiler");
                errors.Add($"Error running F# compiler: {ex.Message}");
                return (false, string.Empty, errors);
            }
        }

        private string GetFscPath()
        {
            // Try to find fsc.exe in the PATH
            var paths = Environment.GetEnvironmentVariable("PATH")?.Split(Path.PathSeparator);
            if (paths != null)
            {
                foreach (var path in paths)
                {
                    var fscPath = Path.Combine(path, "fsc.exe");
                    if (File.Exists(fscPath))
                    {
                        return fscPath;
                    }
                }
            }

            // Try to find fsc.exe in the .NET SDK directory
            var dotnetPath = GetDotNetPath();
            if (!string.IsNullOrEmpty(dotnetPath))
            {
                var sdkDir = Path.GetDirectoryName(dotnetPath);
                if (sdkDir != null)
                {
                    var fscPath = Path.Combine(sdkDir, "FSharp", "fsc.exe");
                    if (File.Exists(fscPath))
                    {
                        return fscPath;
                    }
                }
            }

            // Use dotnet fsi as a fallback
            return "dotnet";
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

        private List<string> GetDefaultReferences()
        {
            var references = new List<string>();

            // Add references to common assemblies
            var assemblies = new[]
            {
                typeof(object).Assembly, // mscorlib
                typeof(Console).Assembly, // System.Console
                typeof(File).Assembly, // System.IO.FileSystem
                typeof(Path).Assembly, // System.IO.FileSystem.Primitives
                typeof(List<>).Assembly, // System.Collections
                typeof(Enumerable).Assembly, // System.Linq
                typeof(Task).Assembly, // System.Threading.Tasks
                typeof(Uri).Assembly, // System.Private.Uri
                typeof(System.Text.RegularExpressions.Regex).Assembly, // System.Text.RegularExpressions
                typeof(StringBuilder).Assembly, // System.Text.StringBuilder
                typeof(System.Text.Json.JsonSerializer).Assembly, // System.Text.Json
                typeof(System.Xml.Linq.XDocument).Assembly, // System.Xml.XDocument
                typeof(System.Xml.XmlDocument).Assembly // System.Xml
            };

            foreach (var assembly in assemblies)
            {
                if (!string.IsNullOrEmpty(assembly.Location))
                {
                    references.Add(assembly.Location);
                }
            }

            // Add FSharp.Core reference
            var fsharpCorePath = Path.Combine(
                Path.GetDirectoryName(typeof(object).Assembly.Location),
                "..",
                "FSharp.Core.dll");

            if (File.Exists(fsharpCorePath))
            {
                references.Add(fsharpCorePath);
            }

            return references;
        }
    }
}
