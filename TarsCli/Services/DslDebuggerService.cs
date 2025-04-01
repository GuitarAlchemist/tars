using Microsoft.Extensions.Logging;
using TarsEngine.DSL;

namespace TarsCli.Services;

/// <summary>
/// Service for debugging TARS DSL scripts
/// </summary>
public class DslDebuggerService
{
    private readonly ILogger<DslDebuggerService> _logger;
    private readonly DslService _dslService;

    // Breakpoints stored as (file, line) pairs
    private readonly HashSet<(string, int)> _breakpoints = new();

    // Step mode flag
    private bool _stepMode = false;

    // Current file and line
    private string _currentFile = "";
    private int _currentLine = 0;

    // Paused flag
    private bool _isPaused = false;

    // Variable watch list
    private readonly HashSet<string> _watchedVariables = new();

    /// <summary>
    /// Constructor
    /// </summary>
    public DslDebuggerService(ILogger<DslDebuggerService> logger, DslService dslService)
    {
        _logger = logger;
        _dslService = dslService;
    }

    /// <summary>
    /// Add a breakpoint
    /// </summary>
    /// <param name="file">The file path</param>
    /// <param name="line">The line number</param>
    public void AddBreakpoint(string file, int line)
    {
        _breakpoints.Add((file, line));
        _logger.LogInformation($"Breakpoint added at {file}:{line}");
    }

    /// <summary>
    /// Remove a breakpoint
    /// </summary>
    /// <param name="file">The file path</param>
    /// <param name="line">The line number</param>
    public void RemoveBreakpoint(string file, int line)
    {
        _breakpoints.Remove((file, line));
        _logger.LogInformation($"Breakpoint removed at {file}:{line}");
    }

    /// <summary>
    /// Clear all breakpoints
    /// </summary>
    public void ClearBreakpoints()
    {
        _breakpoints.Clear();
        _logger.LogInformation("All breakpoints cleared");
    }

    /// <summary>
    /// List all breakpoints
    /// </summary>
    /// <returns>A list of breakpoints</returns>
    public IEnumerable<(string, int)> ListBreakpoints()
    {
        return _breakpoints;
    }

    /// <summary>
    /// Enable step mode
    /// </summary>
    public void EnableStepMode()
    {
        _stepMode = true;
        _logger.LogInformation("Step mode enabled");
    }

    /// <summary>
    /// Disable step mode
    /// </summary>
    public void DisableStepMode()
    {
        _stepMode = false;
        _logger.LogInformation("Step mode disabled");
    }

    /// <summary>
    /// Add a variable to the watch list
    /// </summary>
    /// <param name="variableName">The variable name</param>
    public void AddWatch(string variableName)
    {
        _watchedVariables.Add(variableName);
        _logger.LogInformation($"Watch added for variable: {variableName}");
    }

    /// <summary>
    /// Remove a variable from the watch list
    /// </summary>
    /// <param name="variableName">The variable name</param>
    public void RemoveWatch(string variableName)
    {
        _watchedVariables.Remove(variableName);
        _logger.LogInformation($"Watch removed for variable: {variableName}");
    }

    /// <summary>
    /// Clear the watch list
    /// </summary>
    public void ClearWatches()
    {
        _watchedVariables.Clear();
        _logger.LogInformation("All watches cleared");
    }

    /// <summary>
    /// List all watched variables
    /// </summary>
    /// <returns>A list of watched variables</returns>
    public IEnumerable<string> ListWatches()
    {
        return _watchedVariables;
    }

    /// <summary>
    /// Run a DSL file with debugging
    /// </summary>
    /// <param name="filePath">The path to the DSL file</param>
    /// <returns>0 if successful, 1 if failed</returns>
    public async Task<int> RunWithDebuggingAsync(string filePath)
    {
        try
        {
            _logger.LogInformation($"Running DSL file with debugging: {filePath}");

            if (!File.Exists(filePath))
            {
                _logger.LogError($"File not found: {filePath}");
                return 1;
            }

            // Read the file
            string code = await File.ReadAllTextAsync(filePath);

            // Parse the DSL
            var program = Parser.parse(code);

            // Set up the debugging environment
            _currentFile = filePath;
            _currentLine = 1;
            _isPaused = false;

            // Convert breakpoints to F# set
            var breakpointSet = new Microsoft.FSharp.Collections.FSharpSet<Tuple<string, int>>(
                _breakpoints.Select(bp => Tuple.Create(bp.Item1, bp.Item2)));

            // Execute the program with debugging
            var result = Interpreter.executeWithDebugging(program, breakpointSet, _stepMode);

            switch (result)
            {
                case Interpreter.ExecutionResult.Success success:
                    _logger.LogInformation($"DSL execution successful: {success}");
                    return 0;
                case Interpreter.ExecutionResult.Error error:
                    _logger.LogError($"DSL execution failed: {error}");
                    return 1;
                default:
                    _logger.LogError($"Unknown execution result: {result}");
                    return 1;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error running DSL file with debugging: {ex.Message}");
            return 1;
        }
    }

    /// <summary>
    /// Continue execution after a breakpoint
    /// </summary>
    public void Continue()
    {
        if (_isPaused)
        {
            _isPaused = false;
            _logger.LogInformation("Continuing execution");
        }
        else
        {
            _logger.LogWarning("Not paused at a breakpoint");
        }
    }

    /// <summary>
    /// Step to the next line
    /// </summary>
    public void StepNext()
    {
        if (_isPaused)
        {
            _isPaused = false;
            _stepMode = true;
            _logger.LogInformation("Stepping to next line");
        }
        else
        {
            _logger.LogWarning("Not paused at a breakpoint");
        }
    }

    /// <summary>
    /// Step into a function
    /// </summary>
    public void StepInto()
    {
        if (_isPaused)
        {
            _isPaused = false;
            _stepMode = true;
            _logger.LogInformation("Stepping into function");
        }
        else
        {
            _logger.LogWarning("Not paused at a breakpoint");
        }
    }

    /// <summary>
    /// Step out of a function
    /// </summary>
    public void StepOut()
    {
        if (_isPaused)
        {
            _isPaused = false;
            _stepMode = false;
            _logger.LogInformation("Stepping out of function");
        }
        else
        {
            _logger.LogWarning("Not paused at a breakpoint");
        }
    }

    /// <summary>
    /// Get the value of a variable
    /// </summary>
    /// <param name="variableName">The variable name</param>
    /// <returns>The variable value</returns>
    public string GetVariableValue(string variableName)
    {
        // In a real implementation, this would get the variable value from the interpreter
        return $"Value of {variableName}";
    }

    /// <summary>
    /// Get the values of all watched variables
    /// </summary>
    /// <returns>A dictionary of variable names and values</returns>
    public Dictionary<string, string> GetWatchedVariableValues()
    {
        var values = new Dictionary<string, string>();

        foreach (var variableName in _watchedVariables)
        {
            values[variableName] = GetVariableValue(variableName);
        }

        return values;
    }

    /// <summary>
    /// Get the current call stack
    /// </summary>
    /// <returns>A list of function names</returns>
    public IEnumerable<string> GetCallStack()
    {
        // In a real implementation, this would get the call stack from the interpreter
        return new[] { "main", "function1", "function2" };
    }
}