namespace TarsCli.Models;

/// <summary>
/// Represents the status of the debugger
/// </summary>
public class DebuggerStatus
{
    /// <summary>
    /// Whether the debugger is running
    /// </summary>
    public bool IsRunning { get; set; }
        
    /// <summary>
    /// The current file being debugged
    /// </summary>
    public string CurrentFile { get; set; }
        
    /// <summary>
    /// The current line being debugged
    /// </summary>
    public int CurrentLine { get; set; }
        
    /// <summary>
    /// Whether the debugger is paused at a breakpoint
    /// </summary>
    public bool IsPaused { get; set; }
        
    /// <summary>
    /// The list of breakpoints
    /// </summary>
    public List<(string, int)> Breakpoints { get; set; } = new List<(string, int)>();
        
    /// <summary>
    /// The list of watched variables
    /// </summary>
    public List<string> Watches { get; set; } = new List<string>();
}
    
/// <summary>
/// Represents a request to start debugging
/// </summary>
public class StartDebuggingRequest
{
    /// <summary>
    /// The path to the DSL file to debug
    /// </summary>
    public string FilePath { get; set; }
        
    /// <summary>
    /// Whether to enable step mode
    /// </summary>
    public bool StepMode { get; set; }
}
    
/// <summary>
/// Represents a breakpoint
/// </summary>
public class Breakpoint
{
    /// <summary>
    /// The file path
    /// </summary>
    public string File { get; set; }
        
    /// <summary>
    /// The line number
    /// </summary>
    public int Line { get; set; }
}
    
/// <summary>
/// Represents a variable watch
/// </summary>
public class Watch
{
    /// <summary>
    /// The variable name
    /// </summary>
    public string Variable { get; set; }
}
    
/// <summary>
/// Represents a variable value
/// </summary>
public class VariableValue
{
    /// <summary>
    /// The variable name
    /// </summary>
    public string Variable { get; set; }
        
    /// <summary>
    /// The variable value
    /// </summary>
    public string Value { get; set; }
}