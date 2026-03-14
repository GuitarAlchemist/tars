using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using TarsEngine.Interfaces.Compilation;

namespace TarsEngine.Interfaces.Compilation
{
    /// <summary>
    /// REAL IMPLEMENTATION NEEDED
    /// </summary>
    public class MockFSharpCompiler : IFSharpCompiler
    {
        /// <inheritdoc/>
        public Task<CompilationResult> CompileAsync(string code)
        {
            // Return a successful compilation result
            return Task.FromResult(new CompilationResult
            {
                Success = true,
                CompiledAssembly = null,
                Errors = new List<CompilationError>()
            });
        }

        /// <inheritdoc/>
        public Task<CompilationResult> CompileAsync(string code, string[] references)
        {
            // Return a successful compilation result
            return Task.FromResult(new CompilationResult
            {
                Success = true,
                CompiledAssembly = null,
                Errors = new List<CompilationError>()
            });
        }

        /// <inheritdoc/>
        public Task<CompilationResult> CompileAsync(string code, string[] references, string outputPath)
        {
            // Return a successful compilation result
            return Task.FromResult(new CompilationResult
            {
                Success = true,
                CompiledAssembly = null,
                Errors = new List<CompilationError>()
            });
        }

        /// <inheritdoc/>
        public Task<CompilationResult> CompileAsync(string code, string[] references, string outputPath, bool generateExecutable)
        {
            // Return a successful compilation result
            return Task.FromResult(new CompilationResult
            {
                Success = true,
                CompiledAssembly = null,
                Errors = new List<CompilationError>()
            });
        }

        /// <inheritdoc/>
        public Task<CompilationResult> CompileAsync(string code, string[] references, string outputPath, bool generateExecutable, string[] defines)
        {
            // Return a successful compilation result
            return Task.FromResult(new CompilationResult
            {
                Success = true,
                CompiledAssembly = null,
                Errors = new List<CompilationError>()
            });
        }

        /// <inheritdoc/>
        public Task<CompilationResult> CompileAsync(string code, string[] references, string outputPath, bool generateExecutable, string[] defines, string[] sourceFiles)
        {
            // Return a successful compilation result
            return Task.FromResult(new CompilationResult
            {
                Success = true,
                CompiledAssembly = null,
                Errors = new List<CompilationError>()
            });
        }

        /// <inheritdoc/>
        public Task<CompilationResult> CompileAsync(string code, string[] references, string outputPath, bool generateExecutable, string[] defines, string[] sourceFiles, string[] resources)
        {
            // Return a successful compilation result
            return Task.FromResult(new CompilationResult
            {
                Success = true,
                CompiledAssembly = null,
                Errors = new List<CompilationError>()
            });
        }

        /// <inheritdoc/>
        public Task<CompilationResult> CompileAsync(string code, string[] references, string outputPath, bool generateExecutable, string[] defines, string[] sourceFiles, string[] resources, string[] otherFlags)
        {
            // Return a successful compilation result
            return Task.FromResult(new CompilationResult
            {
                Success = true,
                CompiledAssembly = null,
                Errors = new List<CompilationError>()
            });
        }
    }
}

