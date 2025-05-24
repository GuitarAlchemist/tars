using System;
using System.Collections.Generic;
using System.Reflection;

namespace TarsEngine.Services.Compilation
{
    /// <summary>
    /// Represents the result of a compilation operation.
    /// </summary>
    public class CompilationResult
    {
        /// <summary>
        /// Gets a value indicating whether the compilation was successful.
        /// </summary>
        public bool Success { get; }

        /// <summary>
        /// Gets the errors that occurred during compilation.
        /// </summary>
        public IReadOnlyList<string> Errors { get; }

        /// <summary>
        /// Gets the warnings that occurred during compilation.
        /// </summary>
        public IReadOnlyList<string> Warnings { get; }

        /// <summary>
        /// Gets the output of the compilation.
        /// </summary>
        public string Output { get; }

        /// <summary>
        /// Gets the assembly produced by the compilation.
        /// </summary>
        public Assembly Assembly { get; }

        /// <summary>
        /// Gets the path to the compiled output file.
        /// </summary>
        public string OutputPath { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="CompilationResult"/> class.
        /// </summary>
        /// <param name="success">A value indicating whether the compilation was successful.</param>
        /// <param name="errors">The errors that occurred during compilation.</param>
        /// <param name="warnings">The warnings that occurred during compilation.</param>
        /// <param name="output">The output of the compilation.</param>
        /// <param name="assembly">The assembly produced by the compilation.</param>
        /// <param name="outputPath">The path to the compiled output file.</param>
        public CompilationResult(
            bool success,
            IReadOnlyList<string> errors,
            IReadOnlyList<string> warnings,
            string output,
            Assembly assembly = null,
            string outputPath = null)
        {
            Success = success;
            Errors = errors ?? Array.Empty<string>();
            Warnings = warnings ?? Array.Empty<string>();
            Output = output ?? string.Empty;
            Assembly = assembly;
            OutputPath = outputPath;
        }

        /// <summary>
        /// Creates a successful compilation result.
        /// </summary>
        /// <param name="output">The output of the compilation.</param>
        /// <param name="assembly">The assembly produced by the compilation.</param>
        /// <param name="outputPath">The path to the compiled output file.</param>
        /// <param name="warnings">The warnings that occurred during compilation.</param>
        /// <returns>A successful compilation result.</returns>
        public static CompilationResult CreateSuccess(
            string output = null,
            Assembly assembly = null,
            string outputPath = null,
            IReadOnlyList<string> warnings = null)
        {
            return new CompilationResult(
                true,
                Array.Empty<string>(),
                warnings ?? Array.Empty<string>(),
                output,
                assembly,
                outputPath);
        }

        /// <summary>
        /// Creates a failed compilation result.
        /// </summary>
        /// <param name="errors">The errors that occurred during compilation.</param>
        /// <param name="warnings">The warnings that occurred during compilation.</param>
        /// <param name="output">The output of the compilation.</param>
        /// <returns>A failed compilation result.</returns>
        public static CompilationResult CreateFailure(
            IReadOnlyList<string> errors,
            IReadOnlyList<string> warnings = null,
            string output = null)
        {
            return new CompilationResult(
                false,
                errors ?? Array.Empty<string>(),
                warnings ?? Array.Empty<string>(),
                output);
        }
    }
}
