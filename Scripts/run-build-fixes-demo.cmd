@echo off
setlocal

:: Set the solution file path
set SOLUTION_FILE=tars.sln

:: Set colors for better visibility
set YELLOW=6
set GREEN=2
set RED=4
set CYAN=3
set WHITE=7

:: Function to set text color
call :setColor %CYAN%
echo Running Build Fixes Demo
echo ========================
echo.
call :setColor %WHITE%

:: Step 1: Identify Build Errors
call :setColor %YELLOW%
echo Step 1: Identify Build Errors
echo ----------------------------
call :setColor %WHITE%
echo Running build to identify errors...

:: Simulate build errors
call :setColor %RED%
echo error CS1061: 'TestRunnerService' does not contain a definition for 'RunTestsAsync'
echo error CS8767: Nullability of reference types in type of parameter 'exception' doesn't match
call :setColor %WHITE%
echo.
echo Build errors identified. Press any key to continue...
pause > nul

echo.
:: Step 2: Fix Model Class Compatibility Issues
call :setColor %YELLOW%
echo Step 2: Fix Model Class Compatibility Issues
echo ------------------------------------------
call :setColor %WHITE%
echo Creating adapter classes for model compatibility...

:: Show example adapter class
call :setColor %CYAN%
echo public static class CodeIssueAdapter
echo {
echo     public static TarsEngine.Models.CodeIssue ToEngineCodeIssue(this TarsCli.Models.CodeIssue cliIssue)
echo     {
echo         return new TarsEngine.Models.CodeIssue
echo         {
echo             Description = cliIssue.Message,
echo             CodeSnippet = cliIssue.Code,
echo             SuggestedFix = cliIssue.Suggestion,
echo             // ... other properties
echo         };
echo     }
echo }
call :setColor %WHITE%

echo.
echo CodeIssueAdapter.cs created. Press any key to continue...
pause > nul

echo.
:: Step 3: Fix Service Conflicts
call :setColor %YELLOW%
echo Step 3: Fix Service Conflicts
echo ---------------------------
call :setColor %WHITE%
echo Updating references to use fully qualified names...

:: Show example service conflict fix
call :setColor %CYAN%
echo // Before
echo private readonly TestRunnerService _testRunnerService;
echo.
echo // After
echo private readonly Testing.TestRunnerService _testRunnerService;
echo.
echo // Before
echo var testRunResult = await _testRunnerService.RunTestsAsync(testFilePath);
echo.
echo // After
echo var testRunResult = await _testRunnerService.RunTestFileAsync(testFilePath);
call :setColor %WHITE%

echo.
echo Service conflicts resolved. Press any key to continue...
pause > nul

echo.
:: Step 4: Fix Nullability Warnings
call :setColor %YELLOW%
echo Step 4: Fix Nullability Warnings
echo -----------------------------
call :setColor %WHITE%
echo Implementing interface methods explicitly...

:: Show example nullability fix
call :setColor %CYAN%
echo public class LoggerAdapter^<T^> : ILogger^<T^>
echo {
echo     private readonly ILogger _logger;
echo.
echo     public LoggerAdapter(ILogger logger)
echo     {
echo         _logger = logger ?? throw new ArgumentNullException(nameof(logger));
echo     }
echo.
echo     // Explicit interface implementation with correct nullability
echo     IDisposable ILogger.BeginScope^<TState^>(TState state) =^> _logger.BeginScope(state);
echo.
echo     public bool IsEnabled(LogLevel logLevel) =^> _logger.IsEnabled(logLevel);
echo.
echo     // Explicit interface implementation with correct nullability
echo     void ILogger.Log^<TState^>(LogLevel logLevel, EventId eventId, TState state,
echo         Exception? exception, Func^<TState, Exception?, string^> formatter)
echo     {
echo         _logger.Log(logLevel, eventId, state, exception, formatter);
echo     }
echo }
call :setColor %WHITE%

echo.
echo Nullability warnings fixed. Press any key to continue...
pause > nul

echo.
:: Step 5: Verify Fixes
call :setColor %YELLOW%
echo Step 5: Verify Fixes
echo ------------------
call :setColor %WHITE%
echo Running build again to verify fixes...

:: Simulate successful build
call :setColor %GREEN%
echo Build succeeded with 0 errors
call :setColor %WHITE%
echo.
echo Build successful! All errors have been resolved.
echo.
echo Demo completed. See docs\demos\Build-Fixes-Demo.md for more information.
echo.
pause

goto :eof

:setColor
powershell -Command "$Host.UI.RawUI.ForegroundColor = %1"
goto :eof
