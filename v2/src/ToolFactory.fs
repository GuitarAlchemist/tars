// ToolFactory.fs - Register custom tools including run_static_analysis
module ToolFactory
open System
open Types
open StaticAnalysisRunner

/// Generic tool definition
type Tool = {
    Name: string
    Description: string
    InputSchema: string option
    OutputSchema: string option
    Execute: obj -> Result<obj, string>
}

/// Factory to create tools by name
let createTool (toolName: string) : Result<Tool, string> =
    match toolName with
    | "run_static_analysis" ->
        // Input: optional root path (string). Output: JSON string of analysis issues.
        let exec (input: obj) =
            try
                let rootOpt =
                    match input with
                    | :? string as s when not (String.IsNullOrWhiteSpace s) -> Some s
                    | _ -> None
                match runAnalysis rootOpt with
                | Ok issues ->
                    // Serialize to JSON for downstream consumption
                    let json = JsonSerializer.Serialize(issues)
                    Ok (json :> obj)
                | Error err -> Error err
            with ex -> Error (ex.Message)
        Ok {
            Name = "run_static_analysis"
            Description = "Executes static analysis (FSharpLint) on the codebase and returns a JSON report of issues."
            InputSchema = Some "{ \"rootPath\": \"string (optional)\" }"
            OutputSchema = Some "JSON array of {File:string, Line:int, Column:int?, Severity:string, RuleId:string, Message:string}"
            Execute = exec
        }
    | _ ->
        Error (sprintf "Tool '%s' not recognized in ToolFactory." toolName)
}
