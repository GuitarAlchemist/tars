namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Text
open System.Threading.Tasks

/// <summary>
/// Command for displaying help information.
/// </summary>
type HelpCommand(commands: ICommand list) =
    interface ICommand with
        member _.Name = "help"
        
        member _.Description = "Display help information"
        
        member _.Usage = "tars help [command]"
        
        member _.Examples = [
            "tars help"
            "tars help improve"
        ]
        
        member _.ValidateOptions(_) = true
        
        member _.ExecuteAsync(options) =
            Task.Run(fun () ->
                let sb = StringBuilder()
                
                if options.Arguments.IsEmpty then
                    // Show general help
                    sb.AppendLine("TARS - The Autonomous Reasoning System") |> ignore
                    sb.AppendLine() |> ignore
                    sb.AppendLine("Usage: tars <command> [options]") |> ignore
                    sb.AppendLine() |> ignore
                    sb.AppendLine("Commands:") |> ignore
                    
                    for command in commands do
                        sb.AppendLine($"  {command.Name,-15} {command.Description}") |> ignore
                    
                    sb.AppendLine() |> ignore
                    sb.AppendLine("Use 'tars help <command>' for more information about a specific command.") |> ignore
                else
                    // Show help for specific command
                    let commandName = options.Arguments.[0]
                    match commands |> List.tryFind (fun c -> c.Name = commandName) with
                    | Some command ->
                        sb.AppendLine($"Command: {command.Name}") |> ignore
                        sb.AppendLine($"Description: {command.Description}") |> ignore
                        sb.AppendLine() |> ignore
                        sb.AppendLine($"Usage: {command.Usage}") |> ignore
                        sb.AppendLine() |> ignore
                        
                        if not command.Examples.IsEmpty then
                            sb.AppendLine("Examples:") |> ignore
                            for example in command.Examples do
                                sb.AppendLine($"  {example}") |> ignore
                    | None ->
                        sb.AppendLine($"Unknown command: {commandName}") |> ignore
                
                CommandResult.success(sb.ToString())
            )
