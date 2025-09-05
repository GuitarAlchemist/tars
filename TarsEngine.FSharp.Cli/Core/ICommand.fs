namespace TarsEngine.FSharp.Cli.Core

open System.Threading.Tasks

/// Interface for TARS CLI commands
type ICommand =
    /// Command name
    abstract member Name: string
    /// Command description
    abstract member Description: string
    /// Command usage information
    abstract member Usage: string
    /// Execute the command with given arguments and options
    abstract member ExecuteAsync: args: string[] -> options: CommandOptions -> Task<CommandResult>

/// Base implementation for TARS CLI commands
[<AbstractClass>]
type BaseCommand() =
    
    /// Default implementation of ICommand
    interface ICommand with
        member this.Name = this.GetName()
        member this.Description = this.GetDescription()
        member this.Usage = this.GetUsage()
        member this.ExecuteAsync args options = this.ExecuteAsyncImpl args options
    
    /// Get the command name (must be implemented by derived classes)
    abstract member GetName: unit -> string
    
    /// Get the command description (must be implemented by derived classes)
    abstract member GetDescription: unit -> string
    
    /// Get the command usage (can be overridden by derived classes)
    abstract member GetUsage: unit -> string
    default this.GetUsage() = $"tars {this.GetName()} [options]"
    
    /// Execute the command implementation (must be implemented by derived classes)
    abstract member ExecuteAsyncImpl: args: string[] -> options: CommandOptions -> Task<CommandResult>

/// Autonomous instruction command interface
type IAutonomousCommand =
    inherit ICommand
    /// Execute autonomous instruction from file
    abstract member ExecuteInstructionAsync: instructionFile: string -> options: CommandOptions -> Task<CommandResult>

module Command =
    
    /// Create a simple command from functions
    let create name description usage executeFunc =
        { new ICommand with
            member _.Name = name
            member _.Description = description
            member _.Usage = usage
            member _.ExecuteAsync args options = executeFunc args options
        }
    
    /// Create an autonomous command
    let createAutonomous name description usage executeFunc executeInstructionFunc =
        { new IAutonomousCommand with
            member _.Name = name
            member _.Description = description
            member _.Usage = usage
            member _.ExecuteAsync args options = executeFunc args options
            member _.ExecuteInstructionAsync instructionFile options = executeInstructionFunc instructionFile options
        }
