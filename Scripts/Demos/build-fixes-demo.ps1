# Build Fixes Demo PowerShell Script
# This script demonstrates the build fixes implemented in the TARS project

# Function to display colored text
function Write-ColorText {
    param (
        [Parameter(Mandatory=$true)]
        [string]$Text,
        
        [Parameter(Mandatory=$false)]
        [ConsoleColor]$ForegroundColor = [ConsoleColor]::White
    )
    
    Write-Host $Text -ForegroundColor $ForegroundColor
}

# Function to display code
function Write-Code {
    param (
        [Parameter(Mandatory=$true)]
        [string]$Code
    )
    
    Write-Host $Code -ForegroundColor Cyan
}

# Function to display headers
function Write-Header {
    param (
        [Parameter(Mandatory=$true)]
        [string]$Header
    )
    
    Write-Host "`n$Header" -ForegroundColor Yellow
    Write-Host ("-" * $Header.Length) -ForegroundColor Yellow
}

# Clear the console
Clear-Host

# Display the title
Write-ColorText "TARS Build Fixes Demo" -ForegroundColor Cyan
Write-ColorText "=====================" -ForegroundColor Cyan
Write-ColorText "This demo showcases the recent build fixes implemented in the TARS project."
Write-ColorText "It demonstrates how to identify and resolve common build issues in a complex .NET solution."

# Step 1: Identify Build Errors
Write-Header "Step 1: Identify Build Errors"
Write-ColorText "Running build to identify errors..."
Write-ColorText "error CS1061: 'TestRunnerService' does not contain a definition for 'RunTestsAsync'" -ForegroundColor Red
Write-ColorText "error CS8767: Nullability of reference types in type of parameter 'exception' doesn't match" -ForegroundColor Red
Write-ColorText "`nBuild errors identified. Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Step 2: Fix Model Class Compatibility Issues
Write-Header "Step 2: Fix Model Class Compatibility Issues"
Write-ColorText "Creating adapter classes for model compatibility..."
Write-Code @"
public static class CodeIssueAdapter
{
    public static TarsEngine.Models.CodeIssue ToEngineCodeIssue(this TarsCli.Models.CodeIssue cliIssue)
    {
        return new TarsEngine.Models.CodeIssue
        {
            Description = cliIssue.Message,
            CodeSnippet = cliIssue.Code,
            SuggestedFix = cliIssue.Suggestion,
            // ... other properties
        };
    }
}
"@
Write-ColorText "`nCodeIssueAdapter.cs created. Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Step 3: Fix Service Conflicts
Write-Header "Step 3: Fix Service Conflicts"
Write-ColorText "Updating references to use fully qualified names..."
Write-Code @"
// Before
private readonly TestRunnerService _testRunnerService;

// After
private readonly Testing.TestRunnerService _testRunnerService;

// Before
var testRunResult = await _testRunnerService.RunTestsAsync(testFilePath);

// After
var testRunResult = await _testRunnerService.RunTestFileAsync(testFilePath);
"@
Write-ColorText "`nService conflicts resolved. Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Step 4: Fix Nullability Warnings
Write-Header "Step 4: Fix Nullability Warnings"
Write-ColorText "Implementing interface methods explicitly..."
Write-Code @"
public class LoggerAdapter<T> : ILogger<T>
{
    private readonly ILogger _logger;

    public LoggerAdapter(ILogger logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    // Explicit interface implementation with correct nullability
    IDisposable ILogger.BeginScope<TState>(TState state) => _logger.BeginScope(state);

    public bool IsEnabled(LogLevel logLevel) => _logger.IsEnabled(logLevel);

    // Explicit interface implementation with correct nullability
    void ILogger.Log<TState>(LogLevel logLevel, EventId eventId, TState state, 
        Exception? exception, Func<TState, Exception?, string> formatter)
    {
        _logger.Log(logLevel, eventId, state, exception, formatter);
    }
}
"@
Write-ColorText "`nNullability warnings fixed. Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Step 5: Verify Fixes
Write-Header "Step 5: Verify Fixes"
Write-ColorText "Running build again to verify fixes..."
Write-ColorText "Build succeeded with 0 errors" -ForegroundColor Green
Write-ColorText "`nBuild successful! All errors have been resolved."
Write-ColorText "`nDemo completed. See docs\demos\Build-Fixes-Demo.md for more information."

# Summary
Write-Header "Summary"
Write-ColorText "In this demo, we've shown how to fix common build issues in a .NET solution:"
Write-ColorText "1. Model class compatibility issues using adapter classes"
Write-ColorText "2. Service conflicts using fully qualified names"
Write-ColorText "3. Nullability warnings using explicit interface implementation"
Write-ColorText "`nThese fixes have successfully resolved all build errors in the TARS solution."

Write-ColorText "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
