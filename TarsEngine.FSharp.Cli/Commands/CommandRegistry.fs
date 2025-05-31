namespace TarsEngine.FSharp.Cli.Commands

open System.Collections.Generic
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Services

/// <summary>
/// Registry for commands with separate metascript engine.
/// </summary>
type CommandRegistry(
    intelligenceService: IntelligenceService,
    mlService: MLService,
    dockerService: DockerService,
    mixtralService: MixtralService,
    llmRouter: LLMRouter) =
    
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

        // Metascript commands (standalone implementation)
        let loggerFactory = LoggerFactory.Create(fun builder -> builder.AddConsole() |> ignore)
        let executeLogger = loggerFactory.CreateLogger<ExecuteCommand>()
        let yamlLogger = loggerFactory.CreateLogger<YamlProcessingService>()
        let fileLogger = loggerFactory.CreateLogger<FileOperationsService>()

        let yamlService = YamlProcessingService(yamlLogger)
        let fileService = FileOperationsService(fileLogger)
        let executeCommand = ExecuteCommand(executeLogger, yamlService, fileService)

        // Swarm command
        let swarmLogger = loggerFactory.CreateLogger<SwarmCommand>()
        let swarmCommand = SwarmCommand(swarmLogger, dockerService)

        // Mixtral command
        let mixtralLogger = loggerFactory.CreateLogger<MixtralCommand>()
        let mixtralCommand = MixtralCommand(mixtralLogger, mixtralService, llmRouter)

        // Simple Transformer command
        let transformerLogger = loggerFactory.CreateLogger<SimpleTransformerCommand>()
        let transformerCommand = SimpleTransformerCommand(transformerLogger)

        // Mixture of Experts command
        let moeLogger = loggerFactory.CreateLogger<MixtureOfExpertsCommand>()
        let moeCommand = MixtureOfExpertsCommand(moeLogger)

        // Chatbot command
        let chatbotLogger = loggerFactory.CreateLogger<ChatbotCommand>()
        let chatbotCommand = ChatbotCommand(chatbotLogger, moeCommand)

        // Register working commands
        this.RegisterCommand(versionCommand)
        this.RegisterCommand(executeCommand)
        this.RegisterCommand(swarmCommand)
        this.RegisterCommand(mixtralCommand)
        this.RegisterCommand(transformerCommand)
        this.RegisterCommand(moeCommand)
        this.RegisterCommand(chatbotCommand)
        
        // Create help command with all commands (must be last)
        let allCommands = this.GetAllCommands()
        let helpCommand = HelpCommand(allCommands)
        this.RegisterCommand(helpCommand)
