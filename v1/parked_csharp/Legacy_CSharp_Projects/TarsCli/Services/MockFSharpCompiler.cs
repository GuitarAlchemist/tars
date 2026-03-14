using Microsoft.CodeAnalysis.Scripting;
using TarsEngine.Interfaces.Compilation;

namespace TarsCli.Services;

/// <summary>
/// REAL IMPLEMENTATION NEEDED
/// </summary>
public class MockFSharpCompiler : IFSharpCompiler
{
    /// <summary>
    /// Compiles F# code asynchronously (mock implementation)
    /// </summary>
    /// <param name="code">The F# code to compile</param>
    /// <param name="options">Script options for compilation</param>
    /// <returns>Compilation result containing diagnostics</returns>
    public Task<TarsEngine.Interfaces.Compilation.CompilationResult> CompileAsync(string code, ScriptOptions options)
    {
        // Return a successful compilation result with no diagnostics
        return Task.FromResult(new TarsEngine.Interfaces.Compilation.CompilationResult
        {
            Diagnostics = new List<TarsEngine.Interfaces.Compilation.CompilationDiagnostic>()
        });
    }
}

