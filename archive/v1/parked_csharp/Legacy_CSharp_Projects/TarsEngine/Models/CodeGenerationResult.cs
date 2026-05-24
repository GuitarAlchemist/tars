namespace TarsEngine.Models;

/// <summary>
/// Represents the result of a code generation operation
/// </summary>
public class CodeGenerationResult
{
    /// <summary>
    /// Whether the code generation was successful
    /// </summary>
    public bool Success { get; set; } = true;
    
    /// <summary>
    /// The generated code
    /// </summary>
    public string GeneratedCode { get; set; } = string.Empty;
    
    /// <summary>
    /// The error message if the code generation failed
    /// </summary>
    public string ErrorMessage { get; set; } = string.Empty;
    
    /// <summary>
    /// The file path where the code was saved
    /// </summary>
    public string FilePath { get; set; } = string.Empty;
}
