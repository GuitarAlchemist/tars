// StaticAnalysisRunner.fs - Wrapper for static analysis (e.g., FSharpLint)
module StaticAnalysisRunner
open System
open System.Diagnostics
open System.Text.Json

// Types representing analysis results
type AnalysisIssue = {
    File: string
    Line: int
    Column: int option
    Severity: string
    RuleId: string
    Message: string
}

type AnalysisResult = AnalysisIssue list

/// Executes the configured static analyzer and returns a structured result.
let runAnalysis (rootPath: string option) : Result<AnalysisResult, string> =
    let target = defaultArg rootPath "."
    // Command assumes `dotnet fsharplint` is installed and available on PATH.
    let command = sprintf "dotnet fsharplint --json %s" target
    try
        let psi = ProcessStartInfo()
        psi.FileName <- "cmd"
        psi.Arguments <- "/c " + command
        psi.RedirectStandardOutput <- true
        psi.RedirectStandardError <- true
        psi.UseShellExecute <- false
        use proc = Process.Start(psi)
        let output = proc.StandardOutput.ReadToEnd()
        proc.WaitForExit()
        if proc.ExitCode = 0 then
            // Parse JSON output into AnalysisResult
            try
                let result = JsonSerializer.Deserialize<AnalysisResult>(output)
                Ok result
            with ex -> Error (sprintf "JSON parsing error: %s" ex.Message)
        else
            let err = proc.StandardError.ReadToEnd()
            Error (sprintf "Analyzer failed (code %d): %s" proc.ExitCode err)
    with ex -> Error (sprintf "Exception while running analyzer: %s" ex.Message)
