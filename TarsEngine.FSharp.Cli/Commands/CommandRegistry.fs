namespace TarsEngine.FSharp.Cli.Commands

open System.Collections.Generic
open TarsEngine.FSharp.Cli.Services
open TarsEngine.FSharp.Metascripts.Services

/// <summary>
/// Registry for commands with separate metascript engine.
/// </summary>
type CommandRegistry(
    metascriptService: IMetascriptService,
    intelligenceService: IntelligenceService,
    mlService: MLService) =
    
    let commands = Dictionary<string, ICommand>()
    
    /// <summary>
    /// Registers a command.
    /// </summary>
    member _.RegisterCommand(command: ICommand) =
        commands.[command.Name] <- command
    
    /// <summary>
    /// Gets a command by name.
    /// </summary>
    member _.GetCommand(name: string) =
        match commands.TryGetValue(name) with
        | true, command -> Some command
        | false, _ -> None
    
    /// <summary>
    /// Gets all registered commands.
    /// </summary>
    member _.GetAllCommands() =
        commands.Values |> Seq.toList
    
    /// <summary>
    /// Registers the default commands with separate engines.
    /// </summary>
    member this.RegisterDefaultCommands() =
        // Core commands
        let versionCommand = VersionCommand()
        let improveCommand = ImproveCommand()
        
        // Development commands
        let compileCommand = CompileCommand()
        let runCommand = RunCommand()
        let testCommand = TestCommand()
        let analyzeCommand = AnalyzeCommand()
        
        // Metascript commands (using separate engine)
        let metascriptListCommand = MetascriptListCommand(metascriptService)
        
        // Advanced commands (using CLI services)
        let intelligenceCommand = IntelligenceCommand(intelligenceService)
        let mlCommand = MLCommand(mlService)
        
        // Register all commands
        this.RegisterCommand(versionCommand)
        this.RegisterCommand(improveCommand)
        this.RegisterCommand(compileCommand)
        this.RegisterCommand(runCommand)
        this.RegisterCommand(testCommand)
        this.RegisterCommand(analyzeCommand)
        this.RegisterCommand(metascriptListCommand)
        this.RegisterCommand(intelligenceCommand)
        this.RegisterCommand(mlCommand)
        
        // Create help command with all commands (must be last)
        let allCommands = this.GetAllCommands()
        let helpCommand = HelpCommand(allCommands)
        this.RegisterCommand(helpCommand)
