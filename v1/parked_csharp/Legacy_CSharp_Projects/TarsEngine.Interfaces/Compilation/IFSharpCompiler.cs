using Microsoft.CodeAnalysis.Scripting;

namespace TarsEngine.Interfaces.Compilation;

/// <summary>
/// Interface for F# compilation services
/// </summary>
public interface IFSharpCompiler
{
    /// <summary>
    /// Compiles F# code asynchronously
    /// </summary>
    /// <param name="code">The F# code to compile</param>
    /// <param name="options">Script options for compilation</param>
    /// <returns>Compilation result containing diagnostics</returns>
    Task<CompilationResult> CompileAsync(string code, ScriptOptions options);
}

/// <summary>
/// Represents the result of a compilation
/// </summary>
public class CompilationResult
{
    /// <summary>
    /// Gets the compilation diagnostics
    /// </summary>
    public IReadOnlyList<CompilationDiagnostic> Diagnostics { get; init; } = new List<CompilationDiagnostic>();
}

/// <summary>
/// Represents a compilation diagnostic message
/// </summary>
public class CompilationDiagnostic
{
    /// <summary>
    /// Gets whether this is an error
    /// </summary>
    public bool IsError { get; init; }

    /// <summary>
    /// Gets whether this is a warning
    /// </summary>
    public bool IsWarning { get; init; }

    /// <summary>
    /// Gets the diagnostic message
    /// </summary>
    public string Message { get; init; } = string.Empty;
}