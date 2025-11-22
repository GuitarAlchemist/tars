namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Collections.Generic
open System.Threading.Tasks

/// <summary>
/// Command options.
/// </summary>
type CommandOptions = {
    /// <summary>
    /// Command arguments.
    /// </summary>
    Arguments: string list
    
    /// <summary>
    /// Command options.
    /// </summary>
    Options: Map<string, string>
    
    /// <summary>
    /// Whether help is requested.
    /// </summary>
    Help: bool
}

/// <summary>
/// Command result.
/// </summary>
type CommandResult = {
    /// <summary>
    /// Whether the command was successful.
    /// </summary>
    Success: bool
    
    /// <summary>
    /// The exit code.
    /// </summary>
    ExitCode: int
    
    /// <summary>
    /// The result message.
    /// </summary>
    Message: string
}

/// <summary>
/// Command options module.
/// </summary>
module CommandOptions =
    /// <summary>
    /// Creates default command options.
    /// </summary>
    let createDefault() = {
        Arguments = []
        Options = Map.empty
        Help = false
    }
    
    /// <summary>
    /// Adds arguments to command options.
    /// </summary>
    let withArguments args options = { options with Arguments = args }
    
    /// <summary>
    /// Adds options to command options.
    /// </summary>
    let withOptions opts options = { options with Options = opts }
    
    /// <summary>
    /// Sets help flag.
    /// </summary>
    let withHelp help options = { options with Help = help }

/// <summary>
/// Command result module.
/// </summary>
module CommandResult =
    /// <summary>
    /// Creates a successful command result.
    /// </summary>
    let success message = {
        Success = true
        ExitCode = 0
        Message = message
    }
    
    /// <summary>
    /// Creates a failed command result.
    /// </summary>
    let failure message = {
        Success = false
        ExitCode = 1
        Message = message
    }

/// <summary>
/// Interface for commands.
/// </summary>
type ICommand =
    /// <summary>
    /// The name of the command.
    /// </summary>
    abstract member Name: string
    
    /// <summary>
    /// The description of the command.
    /// </summary>
    abstract member Description: string
    
    /// <summary>
    /// The usage of the command.
    /// </summary>
    abstract member Usage: string
    
    /// <summary>
    /// Examples of the command.
    /// </summary>
    abstract member Examples: string list
    
    /// <summary>
    /// Validates the command options.
    /// </summary>
    /// <param name="options">The command options.</param>
    /// <returns>True if the options are valid, false otherwise.</returns>
    abstract member ValidateOptions: options: CommandOptions -> bool
    
    /// <summary>
    /// Executes the command asynchronously.
    /// </summary>
    /// <param name="options">The command options.</param>
    /// <returns>The command result.</returns>
    abstract member ExecuteAsync: options: CommandOptions -> Task<CommandResult>
