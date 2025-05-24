namespace TarsEngine.FSharp.Core.Compilation

open System
open System.Reflection
open System.Collections.Generic

/// <summary>
/// Represents the result of a compilation operation.
/// </summary>
type CompilationResult = {
    /// <summary>
    /// Gets a value indicating whether the compilation was successful.
    /// </summary>
    Success: bool
    
    /// <summary>
    /// Gets the errors that occurred during compilation.
    /// </summary>
    Errors: string list
    
    /// <summary>
    /// Gets the warnings that occurred during compilation.
    /// </summary>
    Warnings: string list
    
    /// <summary>
    /// Gets the output of the compilation.
    /// </summary>
    Output: string
    
    /// <summary>
    /// Gets the assembly produced by the compilation.
    /// </summary>
    Assembly: Assembly option
    
    /// <summary>
    /// Gets the path to the compiled output file.
    /// </summary>
    OutputPath: string option
}

/// <summary>
/// Represents the options for compilation.
/// </summary>
type CompilationOptions = {
    /// <summary>
    /// Gets the references to include in the compilation.
    /// </summary>
    References: string list
    
    /// <summary>
    /// Gets the output path for the compilation.
    /// </summary>
    OutputPath: string option
    
    /// <summary>
    /// Gets a value indicating whether to optimize the compilation.
    /// </summary>
    Optimize: bool
    
    /// <summary>
    /// Gets a value indicating whether to treat warnings as errors.
    /// </summary>
    TreatWarningsAsErrors: bool
    
    /// <summary>
    /// Gets the warning level.
    /// </summary>
    WarningLevel: int
    
    /// <summary>
    /// Gets the debug type.
    /// </summary>
    DebugType: string option
    
    /// <summary>
    /// Gets the target framework.
    /// </summary>
    TargetFramework: string option
    
    /// <summary>
    /// Gets the language version.
    /// </summary>
    LanguageVersion: string option
    
    /// <summary>
    /// Gets the additional compiler options.
    /// </summary>
    AdditionalOptions: string list
}

/// <summary>
/// Represents the options for script execution.
/// </summary>
type ScriptExecutionOptions = {
    /// <summary>
    /// Gets the references to include in the script execution.
    /// </summary>
    References: string list
    
    /// <summary>
    /// Gets the arguments to pass to the script.
    /// </summary>
    Arguments: obj list
    
    /// <summary>
    /// Gets the timeout for the script execution.
    /// </summary>
    Timeout: TimeSpan option
    
    /// <summary>
    /// Gets a value indicating whether to capture the output of the script.
    /// </summary>
    CaptureOutput: bool
    
    /// <summary>
    /// Gets the working directory for the script execution.
    /// </summary>
    WorkingDirectory: string option
    
    /// <summary>
    /// Gets the environment variables for the script execution.
    /// </summary>
    EnvironmentVariables: IDictionary<string, string> option
}

/// <summary>
/// Represents the result of a script execution.
/// </summary>
type ScriptExecutionResult = {
    /// <summary>
    /// Gets a value indicating whether the script execution was successful.
    /// </summary>
    Success: bool
    
    /// <summary>
    /// Gets the errors that occurred during script execution.
    /// </summary>
    Errors: string list
    
    /// <summary>
    /// Gets the warnings that occurred during script execution.
    /// </summary>
    Warnings: string list
    
    /// <summary>
    /// Gets the output of the script execution.
    /// </summary>
    Output: string
    
    /// <summary>
    /// Gets the return value of the script execution.
    /// </summary>
    ReturnValue: obj option
    
    /// <summary>
    /// Gets the execution time of the script.
    /// </summary>
    ExecutionTime: TimeSpan
}

/// <summary>
/// Module containing functions for working with compilation results.
/// </summary>
module CompilationResult =
    /// <summary>
    /// Creates a successful compilation result.
    /// </summary>
    /// <param name="output">The output of the compilation.</param>
    /// <param name="assembly">The assembly produced by the compilation.</param>
    /// <param name="outputPath">The path to the compiled output file.</param>
    /// <param name="warnings">The warnings that occurred during compilation.</param>
    /// <returns>A successful compilation result.</returns>
    let createSuccess output assembly outputPath warnings =
        {
            Success = true
            Errors = []
            Warnings = warnings
            Output = output
            Assembly = assembly
            OutputPath = outputPath
        }
    
    /// <summary>
    /// Creates a failed compilation result.
    /// </summary>
    /// <param name="errors">The errors that occurred during compilation.</param>
    /// <param name="warnings">The warnings that occurred during compilation.</param>
    /// <param name="output">The output of the compilation.</param>
    /// <returns>A failed compilation result.</returns>
    let createFailure errors warnings output =
        {
            Success = false
            Errors = errors
            Warnings = warnings
            Output = output
            Assembly = None
            OutputPath = None
        }

/// <summary>
/// Module containing functions for working with compilation options.
/// </summary>
module CompilationOptions =
    /// <summary>
    /// Creates default compilation options.
    /// </summary>
    /// <returns>Default compilation options.</returns>
    let createDefault() =
        {
            References = []
            OutputPath = None
            Optimize = false
            TreatWarningsAsErrors = false
            WarningLevel = 4
            DebugType = Some "portable"
            TargetFramework = Some "net9.0"
            LanguageVersion = Some "latest"
            AdditionalOptions = []
        }
    
    /// <summary>
    /// Creates compilation options with the specified references.
    /// </summary>
    /// <param name="references">The references to include in the compilation.</param>
    /// <returns>Compilation options with the specified references.</returns>
    let withReferences references options =
        { options with References = references }
    
    /// <summary>
    /// Creates compilation options with the specified output path.
    /// </summary>
    /// <param name="outputPath">The output path for the compilation.</param>
    /// <returns>Compilation options with the specified output path.</returns>
    let withOutputPath outputPath options =
        { options with OutputPath = Some outputPath }
    
    /// <summary>
    /// Creates compilation options with the specified optimization setting.
    /// </summary>
    /// <param name="optimize">A value indicating whether to optimize the compilation.</param>
    /// <returns>Compilation options with the specified optimization setting.</returns>
    let withOptimize optimize options =
        { options with Optimize = optimize }
    
    /// <summary>
    /// Creates compilation options with the specified treat warnings as errors setting.
    /// </summary>
    /// <param name="treatWarningsAsErrors">A value indicating whether to treat warnings as errors.</param>
    /// <returns>Compilation options with the specified treat warnings as errors setting.</returns>
    let withTreatWarningsAsErrors treatWarningsAsErrors options =
        { options with TreatWarningsAsErrors = treatWarningsAsErrors }
    
    /// <summary>
    /// Creates compilation options with the specified warning level.
    /// </summary>
    /// <param name="warningLevel">The warning level.</param>
    /// <returns>Compilation options with the specified warning level.</returns>
    let withWarningLevel warningLevel options =
        { options with WarningLevel = warningLevel }
    
    /// <summary>
    /// Creates compilation options with the specified debug type.
    /// </summary>
    /// <param name="debugType">The debug type.</param>
    /// <returns>Compilation options with the specified debug type.</returns>
    let withDebugType debugType options =
        { options with DebugType = Some debugType }
    
    /// <summary>
    /// Creates compilation options with the specified target framework.
    /// </summary>
    /// <param name="targetFramework">The target framework.</param>
    /// <returns>Compilation options with the specified target framework.</returns>
    let withTargetFramework targetFramework options =
        { options with TargetFramework = Some targetFramework }
    
    /// <summary>
    /// Creates compilation options with the specified language version.
    /// </summary>
    /// <param name="languageVersion">The language version.</param>
    /// <returns>Compilation options with the specified language version.</returns>
    let withLanguageVersion languageVersion options =
        { options with LanguageVersion = Some languageVersion }
    
    /// <summary>
    /// Creates compilation options with the specified additional options.
    /// </summary>
    /// <param name="additionalOptions">The additional compiler options.</param>
    /// <returns>Compilation options with the specified additional options.</returns>
    let withAdditionalOptions additionalOptions options =
        { options with AdditionalOptions = additionalOptions }

/// <summary>
/// Module containing functions for working with script execution options.
/// </summary>
module ScriptExecutionOptions =
    /// <summary>
    /// Creates default script execution options.
    /// </summary>
    /// <returns>Default script execution options.</returns>
    let createDefault() =
        {
            References = []
            Arguments = []
            Timeout = None
            CaptureOutput = true
            WorkingDirectory = None
            EnvironmentVariables = None
        }
    
    /// <summary>
    /// Creates script execution options with the specified references.
    /// </summary>
    /// <param name="references">The references to include in the script execution.</param>
    /// <returns>Script execution options with the specified references.</returns>
    let withReferences references options =
        { options with References = references }
    
    /// <summary>
    /// Creates script execution options with the specified arguments.
    /// </summary>
    /// <param name="arguments">The arguments to pass to the script.</param>
    /// <returns>Script execution options with the specified arguments.</returns>
    let withArguments arguments options =
        { options with Arguments = arguments }
    
    /// <summary>
    /// Creates script execution options with the specified timeout.
    /// </summary>
    /// <param name="timeout">The timeout for the script execution.</param>
    /// <returns>Script execution options with the specified timeout.</returns>
    let withTimeout timeout options =
        { options with Timeout = Some timeout }
    
    /// <summary>
    /// Creates script execution options with the specified capture output setting.
    /// </summary>
    /// <param name="captureOutput">A value indicating whether to capture the output of the script.</param>
    /// <returns>Script execution options with the specified capture output setting.</returns>
    let withCaptureOutput captureOutput options =
        { options with CaptureOutput = captureOutput }
    
    /// <summary>
    /// Creates script execution options with the specified working directory.
    /// </summary>
    /// <param name="workingDirectory">The working directory for the script execution.</param>
    /// <returns>Script execution options with the specified working directory.</returns>
    let withWorkingDirectory workingDirectory options =
        { options with WorkingDirectory = Some workingDirectory }
    
    /// <summary>
    /// Creates script execution options with the specified environment variables.
    /// </summary>
    /// <param name="environmentVariables">The environment variables for the script execution.</param>
    /// <returns>Script execution options with the specified environment variables.</returns>
    let withEnvironmentVariables environmentVariables options =
        { options with EnvironmentVariables = Some environmentVariables }

/// <summary>
/// Module containing functions for working with script execution results.
/// </summary>
module ScriptExecutionResult =
    /// <summary>
    /// Creates a successful script execution result.
    /// </summary>
    /// <param name="output">The output of the script execution.</param>
    /// <param name="returnValue">The return value of the script execution.</param>
    /// <param name="executionTime">The execution time of the script.</param>
    /// <param name="warnings">The warnings that occurred during script execution.</param>
    /// <returns>A successful script execution result.</returns>
    let createSuccess output returnValue executionTime warnings =
        {
            Success = true
            Errors = []
            Warnings = warnings
            Output = output
            ReturnValue = returnValue
            ExecutionTime = executionTime
        }
    
    /// <summary>
    /// Creates a failed script execution result.
    /// </summary>
    /// <param name="errors">The errors that occurred during script execution.</param>
    /// <param name="warnings">The warnings that occurred during script execution.</param>
    /// <param name="output">The output of the script execution.</param>
    /// <param name="executionTime">The execution time of the script.</param>
    /// <returns>A failed script execution result.</returns>
    let createFailure errors warnings output executionTime =
        {
            Success = false
            Errors = errors
            Warnings = warnings
            Output = output
            ReturnValue = None
            ExecutionTime = executionTime
        }
