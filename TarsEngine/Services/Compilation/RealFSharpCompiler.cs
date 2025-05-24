using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis.Scripting;
using Microsoft.Extensions.Logging;
using TarsEngine.Interfaces.Compilation;

namespace TarsEngine.Services.Compilation
{
    /// <summary>
    /// Real implementation of the <see cref="IFSharpCompiler"/> interface using the F# compiler.
    /// </summary>
    public class RealFSharpCompiler : IFSharpCompiler
    {
        private readonly ILogger<RealFSharpCompiler> _logger;
        private readonly string _tempDirectory;

        /// <summary>
        /// Initializes a new instance of the <see cref="RealFSharpCompiler"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        public RealFSharpCompiler(ILogger<RealFSharpCompiler> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _tempDirectory = Path.Combine(Path.GetTempPath(), "TarsEngine", "FSharpCompilation");

            // Ensure temp directory exists
            if (!Directory.Exists(_tempDirectory))
            {
                Directory.CreateDirectory(_tempDirectory);
            }
        }

        /// <inheritdoc/>
        public async Task<CompilationResult> CompileAsync(string code, ScriptOptions options)
        {
            try
            {
                _logger.LogInformation("Compiling F# code");

                // Create a temporary file for the F# code
                var tempFilePath = Path.Combine(_tempDirectory, $"{Guid.NewGuid()}.fs");
                await File.WriteAllTextAsync(tempFilePath, code);

                // Output assembly path
                var outputPath = Path.Combine(_tempDirectory, $"{Guid.NewGuid()}.dll");

                // Create F# compiler arguments
                var compilerArgs = new List<string>
                {
                    "fsc",
                    "-o", outputPath,
                    "-a", tempFilePath,
                    "--targetprofile:netstandard"
                };

                // Add references from script options
                if (options?.MetadataReferences != null)
                {
                    foreach (var reference in options.MetadataReferences)
                    {
                        if (reference.Display != null)
                        {
                            compilerArgs.Add("-r");
                            compilerArgs.Add(reference.Display);
                        }
                    }
                }

                // Add default references
                var defaultReferences = GetDefaultReferences();
                foreach (var reference in defaultReferences)
                {
                    compilerArgs.Add("-r");
                    compilerArgs.Add(reference);
                }

                // Run the F# compiler
                var diagnostics = await RunFSharpCompilerAsync(compilerArgs);

                // Clean up temporary file
                File.Delete(tempFilePath);

                // Return the compilation result
                return new CompilationResult
                {
                    Diagnostics = diagnostics
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error compiling F# code");
                return new CompilationResult
                {
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

        private async Task<List<CompilationDiagnostic>> RunFSharpCompilerAsync(List<string> arguments)
        {
            var diagnostics = new List<CompilationDiagnostic>();

            try
            {
                // Create process start info
                var startInfo = new ProcessStartInfo
                {
                    FileName = "dotnet",
                    Arguments = $"fsi --exec {string.Join(" ", arguments)}",
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

                // Process output and error
                var output = outputBuilder.ToString();
                var error = errorBuilder.ToString();

                // Log output and error
                _logger.LogInformation("F# compiler output: {Output}", output);
                _logger.LogInformation("F# compiler error: {Error}", error);

                // Parse diagnostics
                if (!string.IsNullOrEmpty(error))
                {
                    foreach (var line in error.Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries))
                    {
                        // Parse error line
                        if (line.Contains("error FS"))
                        {
                            diagnostics.Add(new CompilationDiagnostic
                            {
                                IsError = true,
                                Message = line
                            });
                        }
                        else if (line.Contains("warning FS"))
                        {
                            diagnostics.Add(new CompilationDiagnostic
                            {
                                IsWarning = true,
                                Message = line
                            });
                        }
                    }
                }

                return diagnostics;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running F# compiler");
                diagnostics.Add(new CompilationDiagnostic
                {
                    IsError = true,
                    Message = $"Error running F# compiler: {ex.Message}"
                });
                return diagnostics;
            }
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
                typeof(Regex).Assembly, // System.Text.RegularExpressions
                typeof(StringBuilder).Assembly, // System.Text.StringBuilder
                typeof(JsonSerializer).Assembly, // System.Text.Json
                typeof(XDocument).Assembly, // System.Xml.XDocument
                typeof(XmlDocument).Assembly // System.Xml
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
