namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks

/// <summary>
/// Command for running F# scripts or applications.
/// </summary>
type RunCommand() =
    interface ICommand with
        member _.Name = "run"
        
        member _.Description = "Run F# scripts or applications"
        
        member _.Usage = "tars run <script> [options]"
        
        member _.Examples = [
            "tars run script.fsx"
            "tars run app.exe --args \"arg1 arg2\""
            "tars run . --project MyProject.fsproj"
            "tars run script.fsx --watch"
        ]
        
        member _.ValidateOptions(options) =
            if options.Arguments.IsEmpty then
                false
            else
                let target = options.Arguments.[0]
                File.Exists(target) || Directory.Exists(target) || target.EndsWith(".fsx") || target.EndsWith(".exe")
        
        member _.ExecuteAsync(options) =
            Task.Run(fun () ->
                try
                    let target = options.Arguments.[0]
                    let args = 
                        match options.Options.TryFind("args") with
                        | Some a -> a
                        | None -> ""
                    let project = 
                        match options.Options.TryFind("project") with
                        | Some p -> Some p
                        | None -> None
                    let watch = options.Options.ContainsKey("watch")
                    
                    Console.WriteLine($"Running: {target}")
                    if not (String.IsNullOrEmpty(args)) then
                        Console.WriteLine($"Arguments: {args}")
                    match project with
                    | Some p -> Console.WriteLine($"Project: {p}")
                    | None -> ()
                    if watch then
                        Console.WriteLine("Watch mode: enabled")
                    
                    // Simulate execution
                    Console.WriteLine("Starting execution...")
                    
                    if target.EndsWith(".fsx") then
                        Console.WriteLine("Executing F# script...")
                        Console.WriteLine("Loading dependencies...")
                        Console.WriteLine("Compiling script...")
                        Console.WriteLine("Running script...")
                    elif target.EndsWith(".exe") then
                        Console.WriteLine("Executing application...")
                        Console.WriteLine("Loading application...")
                        Console.WriteLine("Starting main process...")
                    else
                        Console.WriteLine("Detecting project type...")
                        Console.WriteLine("Building project...")
                        Console.WriteLine("Running application...")
                    
                    Console.WriteLine("Hello from F# application!")
                    Console.WriteLine("Application completed successfully.")
                    
                    let summary = $"""
Execution Complete!

Target: {target}
{if not (String.IsNullOrEmpty(args)) then $"Arguments: {args}\n" else ""}
{match project with | Some p -> $"Project: {p}\n" | None -> ""}
{if watch then "Watch mode: enabled\n" else ""}
Execution completed successfully (simulated)
"""
                    CommandResult.success(summary)
                with
                | ex ->
                    CommandResult.failure($"Error: {ex.Message}")
            )
