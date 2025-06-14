namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Reflection
open System.Threading.Tasks

/// <summary>
/// Command for displaying version information.
/// </summary>
type VersionCommand() =
    interface ICommand with
        member _.Name = "version"
        
        member _.Description = "Display version information"
        
        member self.Usage = "tars version"
        
        member self.Examples = [
            "tars version"
        ]
        
        member self.ValidateOptions(_) = true
        
        member self.ExecuteAsync(_) =
            Task.Run(fun () ->
                let assembly = Assembly.GetExecutingAssembly()
                let version = assembly.GetName().Version
                let informationalVersion = 
                    assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()
                    |> Option.ofObj
                    |> Option.map (fun attr -> attr.InformationalVersion)
                    |> Option.defaultValue (version.ToString())
                
                let message = $"TARS CLI v{informationalVersion}"
                CommandResult.success(message)
            )
