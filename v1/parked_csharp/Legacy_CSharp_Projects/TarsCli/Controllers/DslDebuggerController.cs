using Microsoft.AspNetCore.Mvc;
using TarsCli.Services;
using TarsCli.Models;

namespace TarsCli.Controllers;

/// <summary>
/// Controller for the DSL debugger web API
/// </summary>
[ApiController]
[Route("api/[controller]")]
public class DslDebuggerController : ControllerBase
{
    private readonly ILogger<DslDebuggerController> _logger;
    private readonly DslDebuggerService _debuggerService;
        
    /// <summary>
    /// Constructor
    /// </summary>
    public DslDebuggerController(ILogger<DslDebuggerController> logger, DslDebuggerService debuggerService)
    {
        _logger = logger;
        _debuggerService = debuggerService;
    }
        
    /// <summary>
    /// Get the status of the debugger
    /// </summary>
    [HttpGet("status")]
    public ActionResult<DebuggerStatus> GetStatus()
    {
        try
        {
            var status = new DebuggerStatus
            {
                IsRunning = true, // In a real implementation, this would be determined by the debugger service
                CurrentFile = "examples/metascripts/self_improvement_workflow.tars", // Example
                CurrentLine = 42, // Example
                IsPaused = true, // Example
                Breakpoints = _debuggerService.ListBreakpoints().ToList(),
                Watches = _debuggerService.ListWatches().ToList()
            };
                
            return Ok(status);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting debugger status");
            return StatusCode(500, new { error = ex.Message });
        }
    }
        
    /// <summary>
    /// Start a debugging session
    /// </summary>
    [HttpPost("start")]
    public async Task<ActionResult> StartDebugging([FromBody] StartDebuggingRequest request)
    {
        try
        {
            if (string.IsNullOrEmpty(request.FilePath))
            {
                return BadRequest(new { error = "File path is required" });
            }
                
            var result = await _debuggerService.RunWithDebuggingAsync(request.FilePath);
                
            if (result == 0)
            {
                return Ok(new { message = "Debugging started successfully" });
            }
            else
            {
                return StatusCode(500, new { error = "Failed to start debugging" });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error starting debugging");
            return StatusCode(500, new { error = ex.Message });
        }
    }
        
    /// <summary>
    /// Add a breakpoint
    /// </summary>
    [HttpPost("breakpoints")]
    public ActionResult AddBreakpoint([FromBody] Breakpoint breakpoint)
    {
        try
        {
            if (string.IsNullOrEmpty(breakpoint.File) || breakpoint.Line <= 0)
            {
                return BadRequest(new { error = "File and line are required" });
            }
                
            _debuggerService.AddBreakpoint(breakpoint.File, breakpoint.Line);
                
            return Ok(new { message = "Breakpoint added successfully" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding breakpoint");
            return StatusCode(500, new { error = ex.Message });
        }
    }
        
    /// <summary>
    /// Remove a breakpoint
    /// </summary>
    [HttpDelete("breakpoints")]
    public ActionResult RemoveBreakpoint([FromBody] Breakpoint breakpoint)
    {
        try
        {
            if (string.IsNullOrEmpty(breakpoint.File) || breakpoint.Line <= 0)
            {
                return BadRequest(new { error = "File and line are required" });
            }
                
            _debuggerService.RemoveBreakpoint(breakpoint.File, breakpoint.Line);
                
            return Ok(new { message = "Breakpoint removed successfully" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing breakpoint");
            return StatusCode(500, new { error = ex.Message });
        }
    }
        
    /// <summary>
    /// Get all breakpoints
    /// </summary>
    [HttpGet("breakpoints")]
    public ActionResult<IEnumerable<Breakpoint>> GetBreakpoints()
    {
        try
        {
            var breakpoints = _debuggerService.ListBreakpoints()
                .Select(bp => new Breakpoint { File = bp.Item1, Line = bp.Item2 });
                
            return Ok(breakpoints);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting breakpoints");
            return StatusCode(500, new { error = ex.Message });
        }
    }
        
    /// <summary>
    /// Clear all breakpoints
    /// </summary>
    [HttpDelete("breakpoints/all")]
    public ActionResult ClearBreakpoints()
    {
        try
        {
            _debuggerService.ClearBreakpoints();
                
            return Ok(new { message = "All breakpoints cleared" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error clearing breakpoints");
            return StatusCode(500, new { error = ex.Message });
        }
    }
        
    /// <summary>
    /// Add a watch
    /// </summary>
    [HttpPost("watches")]
    public ActionResult AddWatch([FromBody] Watch watch)
    {
        try
        {
            if (string.IsNullOrEmpty(watch.Variable))
            {
                return BadRequest(new { error = "Variable name is required" });
            }
                
            _debuggerService.AddWatch(watch.Variable);
                
            return Ok(new { message = "Watch added successfully" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding watch");
            return StatusCode(500, new { error = ex.Message });
        }
    }
        
    /// <summary>
    /// Remove a watch
    /// </summary>
    [HttpDelete("watches")]
    public ActionResult RemoveWatch([FromBody] Watch watch)
    {
        try
        {
            if (string.IsNullOrEmpty(watch.Variable))
            {
                return BadRequest(new { error = "Variable name is required" });
            }
                
            _debuggerService.RemoveWatch(watch.Variable);
                
            return Ok(new { message = "Watch removed successfully" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing watch");
            return StatusCode(500, new { error = ex.Message });
        }
    }
        
    /// <summary>
    /// Get all watches
    /// </summary>
    [HttpGet("watches")]
    public ActionResult<IEnumerable<Watch>> GetWatches()
    {
        try
        {
            var watches = _debuggerService.ListWatches()
                .Select(variable => new Watch { Variable = variable });
                
            return Ok(watches);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting watches");
            return StatusCode(500, new { error = ex.Message });
        }
    }
        
    /// <summary>
    /// Clear all watches
    /// </summary>
    [HttpDelete("watches/all")]
    public ActionResult ClearWatches()
    {
        try
        {
            _debuggerService.ClearWatches();
                
            return Ok(new { message = "All watches cleared" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error clearing watches");
            return StatusCode(500, new { error = ex.Message });
        }
    }
        
    /// <summary>
    /// Continue execution
    /// </summary>
    [HttpPost("continue")]
    public ActionResult Continue()
    {
        try
        {
            _debuggerService.Continue();
                
            return Ok(new { message = "Execution continued" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error continuing execution");
            return StatusCode(500, new { error = ex.Message });
        }
    }
        
    /// <summary>
    /// Step to the next line
    /// </summary>
    [HttpPost("step/next")]
    public ActionResult StepNext()
    {
        try
        {
            _debuggerService.StepNext();
                
            return Ok(new { message = "Stepped to next line" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stepping to next line");
            return StatusCode(500, new { error = ex.Message });
        }
    }
        
    /// <summary>
    /// Step into a function
    /// </summary>
    [HttpPost("step/into")]
    public ActionResult StepInto()
    {
        try
        {
            _debuggerService.StepInto();
                
            return Ok(new { message = "Stepped into function" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stepping into function");
            return StatusCode(500, new { error = ex.Message });
        }
    }
        
    /// <summary>
    /// Step out of a function
    /// </summary>
    [HttpPost("step/out")]
    public ActionResult StepOut()
    {
        try
        {
            _debuggerService.StepOut();
                
            return Ok(new { message = "Stepped out of function" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stepping out of function");
            return StatusCode(500, new { error = ex.Message });
        }
    }
        
    /// <summary>
    /// Get the value of a variable
    /// </summary>
    [HttpGet("variables/{variableName}")]
    public ActionResult<VariableValue> GetVariableValue(string variableName)
    {
        try
        {
            if (string.IsNullOrEmpty(variableName))
            {
                return BadRequest(new { error = "Variable name is required" });
            }
                
            var value = _debuggerService.GetVariableValue(variableName);
                
            return Ok(new VariableValue { Variable = variableName, Value = value });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting variable value");
            return StatusCode(500, new { error = ex.Message });
        }
    }
        
    /// <summary>
    /// Get the values of all watched variables
    /// </summary>
    [HttpGet("variables/watched")]
    public ActionResult<IEnumerable<VariableValue>> GetWatchedVariableValues()
    {
        try
        {
            var values = _debuggerService.GetWatchedVariableValues()
                .Select(kv => new VariableValue { Variable = kv.Key, Value = kv.Value });
                
            return Ok(values);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting watched variable values");
            return StatusCode(500, new { error = ex.Message });
        }
    }
        
    /// <summary>
    /// Get the current call stack
    /// </summary>
    [HttpGet("callstack")]
    public ActionResult<IEnumerable<string>> GetCallStack()
    {
        try
        {
            var callStack = _debuggerService.GetCallStack();
                
            return Ok(callStack);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting call stack");
            return StatusCode(500, new { error = ex.Message });
        }
    }
}