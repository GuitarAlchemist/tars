namespace Tars.Engine.Grammar

open System
open System.IO
open System.Diagnostics
open System.Text
open System.Text.RegularExpressions
open Microsoft.Data.Sqlite
open Tars.Engine.Grammar.GrammarSource

/// Language-specific code execution and compilation
module LanguageDispatcher =
    
    /// Execution result
    type ExecutionResult = {
        Success: bool
        Output: string
        Error: string option
        ExitCode: int
        ExecutionTime: TimeSpan
    }
    
    /// Language execution context
    type ExecutionContext = {
        WorkingDirectory: string
        EnvironmentVariables: Map<string, string>
        Timeout: TimeSpan option
        CaptureOutput: bool
    }
    
    /// Create default execution context
    let createDefaultContext () = {
        WorkingDirectory = Directory.GetCurrentDirectory()
        EnvironmentVariables = Map.empty
        Timeout = Some (TimeSpan.FromMinutes(5.0))
        CaptureOutput = true
    }
    
    /// Execute a process with given arguments
    let private executeProcess fileName arguments context =
        let startTime = DateTime.Now
        
        let psi = ProcessStartInfo()
        psi.FileName <- fileName
        psi.Arguments <- arguments
        psi.WorkingDirectory <- context.WorkingDirectory
        psi.UseShellExecute <- false
        psi.RedirectStandardOutput <- context.CaptureOutput
        psi.RedirectStandardError <- context.CaptureOutput
        psi.CreateNoWindow <- true
        
        // Add environment variables
        context.EnvironmentVariables |> Map.iter (fun key value ->
            psi.EnvironmentVariables.[key] <- value)
        
        try
            use proc = Process.Start(psi)

            let output =
                if context.CaptureOutput then
                    proc.StandardOutput.ReadToEnd()
                else ""

            let error =
                if context.CaptureOutput then
                    let err = proc.StandardError.ReadToEnd()
                    if String.IsNullOrWhiteSpace(err) then None else Some err
                else None

            // Handle timeout
            let completed =
                match context.Timeout with
                | Some timeout -> proc.WaitForExit(int timeout.TotalMilliseconds)
                | None -> proc.WaitForExit(); true

            if not completed then
                proc.Kill()
                {
                    Success = false
                    Output = output
                    Error = Some "Process timed out"
                    ExitCode = -1
                    ExecutionTime = DateTime.Now - startTime
                }
            else
                {
                    Success = proc.ExitCode = 0
                    Output = output
                    Error = error
                    ExitCode = proc.ExitCode
                    ExecutionTime = DateTime.Now - startTime
                }
        with
        | ex ->
            {
                Success = false
                Output = ""
                Error = Some ex.Message
                ExitCode = -1
                ExecutionTime = DateTime.Now - startTime
            }
    
    /// Generate temporary file with given extension
    let private generateTempFile extension (content: string) =
        let tempFile = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.{extension}")
        File.WriteAllText(tempFile, content)
        tempFile
    
    /// Clean up temporary file
    let private cleanupTempFile filePath =
        try
            if File.Exists(filePath) then
                File.Delete(filePath)
        with
        | _ -> () // Ignore cleanup errors
    
    /// Execute F# code
    let executeFSharp code context =
        let tempFile = generateTempFile "fsx" code
        try
            let arguments = $"\"{tempFile}\""
            executeProcess "dotnet" $"fsi {arguments}" context
        finally
            cleanupTempFile tempFile
    
    /// Execute C# code
    let executeCSharp code context =
        // Create a simple C# console application
        let programCode = $"""
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

public class Program
{{
    public static void Main(string[] args)
    {{
{code}
    }}
}}"""
        
        let tempFile = generateTempFile "cs" programCode
        let tempDir = Path.GetDirectoryName(tempFile)
        let projectName = Path.GetFileNameWithoutExtension(tempFile)
        
        try
            // Create a temporary project
            let projectFile = Path.Combine(tempDir, $"{projectName}.csproj")
            let projectContent = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
</Project>"""
            File.WriteAllText(projectFile, projectContent)
            
            // Build and run
            let buildResult = executeProcess "dotnet" $"build \"{projectFile}\"" context
            if buildResult.Success then
                executeProcess "dotnet" $"run --project \"{projectFile}\"" context
            else
                buildResult
        finally
            cleanupTempFile tempFile
    
    /// Execute Python code
    let executePython code context =
        let tempFile = generateTempFile "py" code
        try
            let arguments = $"\"{tempFile}\""
            executeProcess "python" arguments context
        finally
            cleanupTempFile tempFile
    
    /// Execute Rust code
    let executeRust code context =
        let mainCode = $"""
fn main() {{
{code}
}}"""
        
        let tempFile = generateTempFile "rs" mainCode
        let tempDir = Path.GetDirectoryName(tempFile)
        let executableName = Path.GetFileNameWithoutExtension(tempFile)
        
        try
            // Compile with rustc
            let compileResult = executeProcess "rustc" $"\"{tempFile}\" -o \"{Path.Combine(tempDir, executableName)}\"" context
            if compileResult.Success then
                // Execute the compiled binary
                let executablePath = Path.Combine(tempDir, executableName + if Environment.OSVersion.Platform = PlatformID.Win32NT then ".exe" else "")
                executeProcess executablePath "" context
            else
                compileResult
        finally
            cleanupTempFile tempFile
    
    /// Execute JavaScript code (using Node.js)
    let executeJavaScript code context =
        let tempFile = generateTempFile "js" code
        try
            let arguments = $"\"{tempFile}\""
            executeProcess "node" arguments context
        finally
            cleanupTempFile tempFile
    
    /// Execute TypeScript code
    let executeTypeScript code context =
        let tempFile = generateTempFile "ts" code
        try
            // First compile TypeScript to JavaScript
            let compileResult = executeProcess "tsc" $"\"{tempFile}\" --outDir \"{Path.GetDirectoryName(tempFile)}\"" context
            if compileResult.Success then
                let jsFile = Path.ChangeExtension(tempFile, ".js")
                executeProcess "node" $"\"{jsFile}\"" context
            else
                compileResult
        finally
            cleanupTempFile tempFile
    
    /// Execute PowerShell code
    let executePowerShell code context =
        let tempFile = generateTempFile "ps1" code
        try
            let arguments = $"-ExecutionPolicy Bypass -File \"{tempFile}\""
            executeProcess "powershell" arguments context
        finally
            cleanupTempFile tempFile
    
    /// Execute Bash code
    let executeBash code context =
        let tempFile = generateTempFile "sh" code
        try
            let arguments = $"\"{tempFile}\""
            executeProcess "bash" arguments context
        finally
            cleanupTempFile tempFile
    
    /// Split SQL script into executable statements while respecting quoted literals
    let private splitSqlStatements (sql: string) =
        let builder = StringBuilder()
        let statements = ResizeArray<string>()
        let mutable inString = false
        let mutable stringDelimiter = '\000'
        let mutable previousChar = '\000'

        for ch in sql do
            match ch with
            | '\'' | '"' as quote when not inString ->
                inString <- true
                stringDelimiter <- quote
                builder.Append(ch) |> ignore
            | _ when inString && ch = stringDelimiter && previousChar <> '\\' ->
                inString <- false
                builder.Append(ch) |> ignore
            | ';' when not inString ->
                let statement = builder.ToString().Trim()
                if not (String.IsNullOrWhiteSpace(statement)) then
                    statements.Add(statement)
                builder.Clear() |> ignore
            | _ ->
                builder.Append(ch) |> ignore

            previousChar <- ch

        let trailing = builder.ToString().Trim()
        if not (String.IsNullOrWhiteSpace(trailing)) then
            statements.Add(trailing)

        statements |> Seq.toList

    /// Execute SQL code using SQLite (configurable via metadata/environment)
    let executeSQL (block: LanguageBlock) context =
        let stopwatch = Stopwatch.StartNew()

        let connectionString =
            block.Metadata
            |> Map.tryFind "connectionString"
            |> Option.orElse (context.EnvironmentVariables |> Map.tryFind "TARS_SQL_CONNECTION")
            |> Option.defaultValue "Data Source=:memory:;Mode=Memory;Cache=Shared"

        let provider =
            block.Metadata
            |> Map.tryFind "provider"
            |> Option.defaultValue "sqlite"

        let statements = splitSqlStatements block.Code

        match provider.ToLowerInvariant() with
        | "sqlite" ->
            try
                use connection = new SqliteConnection(connectionString)
                connection.Open()

                let outputs = ResizeArray<string>()

                for statement in statements do
                    use command = connection.CreateCommand()
                    command.CommandText <- statement

                    let isQuery =
                        Regex.IsMatch(statement, @"^(SELECT|PRAGMA|WITH)\b", RegexOptions.IgnoreCase)

                    if isQuery then
                        use reader = command.ExecuteReader()

                        if reader.FieldCount = 0 then
                            outputs.Add("Query executed (no columns returned)")
                        else
                            let header =
                                [for i in 0 .. reader.FieldCount - 1 -> reader.GetName(i)]
                                |> String.concat "\t"
                            outputs.Add(header)

                            while reader.Read() do
                                let row =
                                    [for i in 0 .. reader.FieldCount - 1 ->
                                        if reader.IsDBNull(i) then
                                            "NULL"
                                        else
                                            reader.GetValue(i).ToString()]
                                    |> String.concat "\t"
                                outputs.Add(row)
                    else
                        let affected = command.ExecuteNonQuery()
                        outputs.Add $"Statement executed (%d{affected} rows affected)"

                {
                    Success = true
                    Output = String.concat Environment.NewLine outputs
                    Error = None
                    ExitCode = 0
                    ExecutionTime = stopwatch.Elapsed
                }
            with
            | ex ->
                {
                    Success = false
                    Output = ""
                    Error = Some ex.Message
                    ExitCode = -1
                    ExecutionTime = stopwatch.Elapsed
                }
        | unsupported ->
            {
                Success = false
                Output = ""
                Error = Some $"Unsupported SQL provider: {unsupported}"
                ExitCode = -1
                ExecutionTime = stopwatch.Elapsed
            }

    /// Execute Wolfram Language code
    let executeWolfram (code: string) context =
        let tempFile = generateTempFile "wl" code
        try
            // Try to execute using WolframScript (if available)
            let arguments = $"-file \"{tempFile}\""
            let result = executeProcess "wolframscript" arguments context

            if result.Success then
                result
            else
                let errorMessage =
                    match result.Error with
                    | Some err when not (String.IsNullOrWhiteSpace err) -> err
                    | _ -> "WolframScript runtime is unavailable or failed to execute the supplied code."

                {
                    result with
                        Success = false
                        Error = Some errorMessage
                }
        finally
            cleanupTempFile tempFile

    /// Execute Julia code
    let executeJulia (code: string) context =
        let tempFile = generateTempFile "jl" code
        try
            // Try to execute using Julia (if available)
            let arguments = $"\"{tempFile}\""
            let result = executeProcess "julia" arguments context

            if result.Success then
                result
            else
                let errorMessage =
                    match result.Error with
                    | Some err when not (String.IsNullOrWhiteSpace err) -> err
                    | _ -> "Julia runtime is unavailable or failed to execute the supplied code."

                {
                    result with
                        Success = false
                        Error = Some errorMessage
                }
        finally
            cleanupTempFile tempFile
    
    /// Dispatch language block execution
    let executeLanguageBlock (block: LanguageBlock) context =
        match block.Language.ToUpperInvariant() with
        | "FSHARP" -> executeFSharp block.Code context
        | "CSHARP" -> executeCSharp block.Code context
        | "PYTHON" -> executePython block.Code context
        | "RUST" -> executeRust block.Code context
        | "JAVASCRIPT" -> executeJavaScript block.Code context
        | "TYPESCRIPT" -> executeTypeScript block.Code context
        | "POWERSHELL" -> executePowerShell block.Code context
        | "BASH" -> executeBash block.Code context
        | "SQL" -> executeSQL block context
        | "WOLFRAM" -> executeWolfram block.Code context
        | "JULIA" -> executeJulia block.Code context
        | unsupported ->
            {
                Success = false
                Output = ""
                Error = Some $"Unsupported language: {unsupported}"
                ExitCode = -1
                ExecutionTime = TimeSpan.Zero
            }
    
    /// Check if language runtime is available
    let checkLanguageAvailability (language: string) =
        let context = createDefaultContext()
        match language.ToUpperInvariant() with
        | "FSHARP" ->
            let result = executeProcess "dotnet" "fsi --help" { context with CaptureOutput = false }
            result.Success
        | "CSHARP" ->
            let result = executeProcess "dotnet" "--version" { context with CaptureOutput = false }
            result.Success
        | "PYTHON" ->
            let result = executeProcess "python" "--version" { context with CaptureOutput = false }
            result.Success
        | "RUST" ->
            let result = executeProcess "rustc" "--version" { context with CaptureOutput = false }
            result.Success
        | "JAVASCRIPT" ->
            let result = executeProcess "node" "--version" { context with CaptureOutput = false }
            result.Success
        | "TYPESCRIPT" ->
            let result = executeProcess "tsc" "--version" { context with CaptureOutput = false }
            result.Success
        | "POWERSHELL" ->
            let result = executeProcess "powershell" "-Command \"Get-Host\"" { context with CaptureOutput = false }
            result.Success
        | "BASH" ->
            let result = executeProcess "bash" "--version" { context with CaptureOutput = false }
            result.Success
        | "WOLFRAM" ->
            let result = executeProcess "wolframscript" "--version" { context with CaptureOutput = false }
            result.Success
        | "JULIA" ->
            let result = executeProcess "julia" "--version" { context with CaptureOutput = false }
            result.Success
        | _ -> false
    
    /// Get available languages on the system
    let getAvailableLanguages () =
        ["FSHARP"; "CSHARP"; "PYTHON"; "RUST"; "JAVASCRIPT"; "TYPESCRIPT"; "POWERSHELL"; "BASH"; "SQL"; "WOLFRAM"; "JULIA"]
        |> List.filter checkLanguageAvailability
    
    /// Validate language block before execution
    let validateLanguageBlock (block: LanguageBlock) =
        let errors = ResizeArray<string>()
        
        if String.IsNullOrWhiteSpace(block.Code) then
            errors.Add("Code content is empty")
        
        if not (LanguageBlock.isSupported block.Language) then
            errors.Add($"Unsupported language: {block.Language}")
        
        if not (checkLanguageAvailability block.Language) then
            errors.Add($"Language runtime not available: {block.Language}")
        
        if errors.Count = 0 then
            Ok block
        else
            Error (errors |> Seq.toList)

