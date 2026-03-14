namespace TarsEngine.FSharp.Cli.Commands

open System.Collections.Generic
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Services
// open TarsEngine.FSharp.Agents


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
    member self.RegisterCommand(command: ICommand) =
        commands.[command.Name] <- command
    
    /// <summary>
    /// Gets a command by name.
    /// </summary>
    member self.GetCommand(name: string) =
        match commands.TryGetValue(name) with
        | true, command -> Some command
        | false, _ -> None
    
    /// <summary>
    /// Gets all registered commands.
    /// </summary>
    member self.GetAllCommands() =
        commands.Values |> Seq.toList
    
    /// <summary>
    /// Registers the default commands with separate engines.
    /// </summary>
    member self.RegisterDefaultCommands() =
        // Core commands
        let versionCommand = new VersionCommand()

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

        // CUDA command - GPU acceleration testing and management
        let cudaLogger = loggerFactory.CreateLogger<CudaCommand>()
        let cudaCommand = CudaCommand(cudaLogger)

        // CUDA DSL command - TARS DSL integration with CUDA computational expressions
        let cudaDslLogger = loggerFactory.CreateLogger<CudaDslCommand>()
        let cudaDslCommand = CudaDslCommand(cudaDslLogger)

        // AI Model command - Real transformer models with CUDA acceleration
        let aiModelLogger = loggerFactory.CreateLogger<AiModelCommand>()
        let aiModelCommand = AiModelCommand(aiModelLogger)

        // AI Agent command - Autonomous agents with GPU-accelerated reasoning
        let aiAgentLogger = loggerFactory.CreateLogger<AiAgentCommand>()
        let aiAgentCommand = AiAgentCommand(aiAgentLogger)

        // AI Metascript command - Natural language programming with AI code generation
        let aiMetascriptLogger = loggerFactory.CreateLogger<AiMetascriptCommand>()
        let aiMetascriptCommand = AiMetascriptCommand(aiMetascriptLogger)

        // AI IDE command - Complete AI-native development environment
        let aiIdeLogger = loggerFactory.CreateLogger<AiIdeCommand>()
        let aiIdeCommand = AiIdeCommand(aiIdeLogger)

        // NEXUS command - Self-Improving Multi-Modal AI
        let nexusLogger = loggerFactory.CreateLogger<SelfImprovingAiCommand>()
        let nexusCommand = SelfImprovingAiCommand(nexusLogger)

        // Advanced AI command - Next-generation AI capabilities
        let advancedLogger = loggerFactory.CreateLogger<AdvancedAiCommand>()
        let advancedCommand = AdvancedAiCommand(advancedLogger)

        // HTTP Server command - API for VS Code extension
        let serverLogger = loggerFactory.CreateLogger<HttpServerCommand>()
        let serverCommand = HttpServerCommand(serverLogger)

        // Register working commands
        self.RegisterCommand(versionCommand)
        self.RegisterCommand(executeCommand)
        self.RegisterCommand(swarmCommand)
        self.RegisterCommand(mixtralCommand)
        self.RegisterCommand(transformerCommand)
        self.RegisterCommand(moeCommand)
        self.RegisterCommand(cudaCommand)
        self.RegisterCommand(cudaDslCommand)
        self.RegisterCommand(aiModelCommand)
        self.RegisterCommand(aiAgentCommand)
        self.RegisterCommand(aiMetascriptCommand)
        self.RegisterCommand(aiIdeCommand)
        self.RegisterCommand(nexusCommand)
        self.RegisterCommand(advancedCommand)
        self.RegisterCommand(serverCommand)
        
        // Create help command with all commands (must be last)
        let allCommands = self.GetAllCommands()
        let helpCommand = HelpCommand(allCommands)
        self.RegisterCommand(helpCommand)
