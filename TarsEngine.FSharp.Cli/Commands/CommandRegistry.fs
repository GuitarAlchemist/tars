namespace TarsEngine.FSharp.Cli.Commands

open System.Collections.Generic
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Core
open TarsEngine.FSharp.Cli.Services
// open TarsEngine.FSharp.Agents  // Temporarily disabled


/// <summary>
/// Registry for commands with separate metascript engine.
/// </summary>
type CommandRegistry(
    intelligenceService: IntelligenceService,
    mlService: MLService,
    dockerService: DockerService) =
    
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

        // Temporarily disabled Mixtral command
        // let mixtralLogger = loggerFactory.CreateLogger<MixtralCommand>()
        // let mixtralCommand = MixtralCommand(mixtralLogger, mixtralService, llmRouter)

        // Simple Transformer command
        let transformerLogger = loggerFactory.CreateLogger<SimpleTransformerCommand>()
        let transformerCommand = SimpleTransformerCommand(transformerLogger)

        // Mixture of Experts command
        let moeLogger = loggerFactory.CreateLogger<MixtureOfExpertsCommand>()
        let moeCommand = MixtureOfExpertsCommand(moeLogger)

        // Chatbot command
        let chatbotLogger = loggerFactory.CreateLogger<ChatbotCommand>()
        let chatbotCommand = ChatbotCommand(chatbotLogger, moeCommand)

        // Temporarily disabled due to Agents dependency
        // Teams command
        // let teamsLogger = loggerFactory.CreateLogger<TeamsCommand>()
        // let teamsCommand = TeamsCommand(teamsLogger)

        // VM command
        // let vmCommand = new VMCommand()

        // Config command - Real configuration management
        // let configLogger = loggerFactory.CreateLogger<ConfigCommand>()
        // let configCommand = ConfigCommand(configLogger)

        // Evolve command - Real auto-evolution system
        // let evolveLogger = loggerFactory.CreateLogger<EvolveCommand>()
        // let evolveCommand = EvolveCommand(evolveLogger)

        // Self-Chat command - Real self-dialogue using MoE
        // let selfChatLogger = loggerFactory.CreateLogger<SelfChatCommand>()
        // let selfChatCommand = SelfChatCommand(selfChatLogger, mixtralService)

        // WebAPI command - REST endpoint and GraphQL generation
        // let webApiLogger = loggerFactory.CreateLogger<WebApiCommand>()
        // let webApiCommand = WebApiCommand(webApiLogger)

        // Live Endpoints command - On-the-fly endpoint creation
        // let liveEndpointsLogger = loggerFactory.CreateLogger<LiveEndpointsCommand>()
        // let liveEndpointsCommand = LiveEndpointsCommand(liveEndpointsLogger)

        // UI command - Autonomous UI generation and management (temporarily disabled)
        // let uiLogger = loggerFactory.CreateLogger<UICommand>()
        // let agentOrchestrator = AgentOrchestrator(loggerFactory.CreateLogger<AgentOrchestrator>())
        // let uiCommand = UICommand(uiLogger, agentOrchestrator)

        // Roadmap command - Roadmap and achievement management (temporarily disabled)
        // let roadmapLogger = loggerFactory.CreateLogger<RoadmapCommand>()
        // let roadmapCommand = RoadmapCommand(roadmapLogger)

        // Service command - Windows service management
        let serviceCommand = new ServiceCommand()

        // Notebook command - Jupyter notebook operations
        let notebookLogger = loggerFactory.CreateLogger<NotebookCommand>()
        let notebookCommand = NotebookCommand(notebookLogger)

        // Generic LLM command - Universal model interface
        let llmLogger = loggerFactory.CreateLogger<LlmCommand>()
        let llmServiceLogger = loggerFactory.CreateLogger<GenericLlmService>()
        let httpClient = new System.Net.Http.HttpClient()
        let llmService = GenericLlmService(llmServiceLogger, httpClient)
        let llmCommand = LlmCommand(llmLogger, llmService)

        // TARS-aware LLM command with knowledge base integration
        let tarsLlmLogger = loggerFactory.CreateLogger<TarsLlmCommand>()
        let vectorStoreLogger = loggerFactory.CreateLogger<CodebaseVectorStore>()
        let vectorStore = CodebaseVectorStore(vectorStoreLogger)
        let knowledgeServiceLogger = loggerFactory.CreateLogger<TarsKnowledgeService>()
        let knowledgeService = TarsKnowledgeService(knowledgeServiceLogger, vectorStore, llmService)
        let tarsLlmCommand = TarsLlmCommand(tarsLlmLogger, knowledgeService)

        // Diagnostics command - Comprehensive system validation
        let diagnosticsLogger = loggerFactory.CreateLogger<DiagnosticsCommand>()
        let diagnosticsCommand = DiagnosticsCommand(diagnosticsLogger)

        // let enhancedDiagnosticsLogger = loggerFactory.CreateLogger<EnhancedDiagnosticsCommand>()
        // let enhancedDiagnosticsCommand = EnhancedDiagnosticsCommand(enhancedDiagnosticsLogger)

        let elmishDiagnosticsLogger = loggerFactory.CreateLogger<ElmishDiagnosticsCommand>()
        let elmishDiagnosticsCommand = ElmishDiagnosticsCommand(elmishDiagnosticsLogger)

        let tarsElmishLogger = loggerFactory.CreateLogger<TarsElmishCommand>()
        let tarsElmishCommand = TarsElmishCommand(tarsElmishLogger)

        // TARS API LLM command - Function calling integration
        let tarsApiLlmLogger = loggerFactory.CreateLogger<TarsApiLlmCommand>()
        let tarsApiLlmCommand = TarsApiLlmCommand(tarsApiLlmLogger, llmService)

        // Generate UI command - AI-driven Elmish UI generation
        let generateUICommand = GenerateUICommand()

        // Self-modifying UI command - Revolutionary self-improving interface
        let selfModifyingUICommand = SelfModifyingUICommand()

        // Register working commands
        self.RegisterCommand(versionCommand)
        self.RegisterCommand(executeCommand)
        self.RegisterCommand(swarmCommand)
        // self.RegisterCommand(mixtralCommand)  // Temporarily disabled
        self.RegisterCommand(transformerCommand)
        self.RegisterCommand(moeCommand)
        self.RegisterCommand(chatbotCommand)
        // Temporarily disabled due to Agents dependency
        // self.RegisterCommand(teamsCommand)
        // self.RegisterCommand(vmCommand)
        // self.RegisterCommand(configCommand)
        // self.RegisterCommand(evolveCommand)
        // self.RegisterCommand(selfChatCommand)
        // self.RegisterCommand(webApiCommand)
        // self.RegisterCommand(liveEndpointsCommand)
        // self.RegisterCommand(uiCommand)
        // self.RegisterCommand(roadmapCommand)
        self.RegisterCommand(serviceCommand)
        self.RegisterCommand(notebookCommand)
        self.RegisterCommand(llmCommand)
        self.RegisterCommand(tarsLlmCommand)
        self.RegisterCommand(diagnosticsCommand)
        // self.RegisterCommand(enhancedDiagnosticsCommand)
        self.RegisterCommand(elmishDiagnosticsCommand)
        self.RegisterCommand(tarsElmishCommand)
        self.RegisterCommand(tarsApiLlmCommand)
        self.RegisterCommand(generateUICommand)
        self.RegisterCommand(selfModifyingUICommand)

        // Create help command with all commands (must be last)
        let allCommands = self.GetAllCommands()
        let helpCommand = HelpCommand(allCommands)
        self.RegisterCommand(helpCommand)
