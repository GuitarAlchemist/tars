open System
open System.Text
open TarsEngine.FSharp.Cli.Core

/// <summary>
/// Main entry point for the TARS CLI application.
/// </summary>
[<EntryPoint>]
let main args =
    try
        // Fix console encoding for proper emoji and Unicode support
        Console.OutputEncoding <- Encoding.UTF8
        Console.InputEncoding <- Encoding.UTF8

        // Create and run the CLI application
        let app = CliApplication()
        let exitCode = app.RunAsync(args).Result
        
        // Return the exit code
        exitCode
    with
    | ex ->
        // Print the error and return a non-zero exit code
        Console.WriteLine($"Fatal error: {ex.Message}")
        1
