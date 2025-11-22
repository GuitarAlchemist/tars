using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis.Scripting;

namespace TarsEngine.Services.Compilation
{
    /// <summary>
    /// Mock implementation of the <see cref="IFSharpCompiler"/> interface for testing.
    /// </summary>
    public class MockFSharpCompiler : IFSharpCompiler
    {
        /// <inheritdoc/>
        public Task<CompilationResult> CompileAsync(string code, ScriptOptions options)
        {
            // Return a successful compilation result
            return Task.FromResult(new CompilationResult
            {
                Diagnostics = new List<CompilationDiagnostic>()
            });
        }
    }
}
