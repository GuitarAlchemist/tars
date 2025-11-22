namespace TarsEngine.FSharp.Cli.Commands

open System.Collections.Generic
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Services
open TarsEngine.FSharp.SelfImprovement


/// <summary>
/// Registry for commands with separate metascript engine.
/// </summary>
type CommandRegistry(
    intelligenceService: IntelligenceService,
    mlService: MLService,
    dockerService: DockerService,
    mixtralService: MixtralService,
    llmRouter: LLMRouter,
    searchService: TarsEngine.FSharp.OnDemandSearch.IOnDemandSearchService,
    selfImprovementService: ISelfImprovementService,
    loggerFactory: ILoggerFactory) =
    
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



        // HTTP Server command - API for VS Code extension
        let serverLogger = loggerFactory.CreateLogger<HttpServerCommand>()
        let serverCommand = HttpServerCommand(serverLogger)

        // Chat command - Interactive TARS chatbot
        let chatLogger = loggerFactory.CreateLogger<ChatbotCommand>()
        let chatCommand = ChatbotCommand(chatLogger, moeCommand)

        // Diagnostics command - System health check
        let diagnosticsLogger = loggerFactory.CreateLogger<DiagnosticsCommand>()
        let diagnosticsCommand = DiagnosticsCommand(diagnosticsLogger)

        // Non-Euclidean Vector Store command - Real hyperbolic geometry
        let nonEuclideanLogger = loggerFactory.CreateLogger<NonEuclideanCommand>()
        let nonEuclideanCommand = NonEuclideanCommand(nonEuclideanLogger)

        // Demo command - Comprehensive TARS demonstrations
        let demoCommand = DemoCommand()

        // Web UI command - Autonomous software engineering web interface
        let webUICommand = WebUICommand()

        // Concept Analysis - Sparse concept decomposition for interpretable AI
        let conceptAnalysisCommand = ConceptAnalysisCommand()

        // Flux Codex workflow command
        let fluxLogger = loggerFactory.CreateLogger<FluxCommand>()
        let fluxCommand = FluxCommand(fluxLogger)

        // Enhanced Intelligence Command - Tier 6 & 7 capabilities
        let enhancedIntelligenceLogger = loggerFactory.CreateLogger<EnhancedIntelligenceCommand>()
        let enhancedIntelligenceCommand = EnhancedIntelligenceCommand(enhancedIntelligenceLogger)

        // Web Search Command - Advanced web search capabilities
        let webSearchLogger = loggerFactory.CreateLogger<WebSearchCommand>()
        let webSearchCommand = WebSearchCommand(webSearchLogger, searchService)

        // TODO: Implement real functionality
        let statusLogger = loggerFactory.CreateLogger<LiveDemoCommand>()
        let statusCommand = LiveDemoCommand(statusLogger, mixtralService)

        // Real AI Command - Genuine AI capabilities with honest reporting
        let realAILogger = loggerFactory.CreateLogger<RealAICommand>()
        let realAICommand = RealAICommand(realAILogger, mixtralService)

        // Advanced LLM Service and Superior AI Command
        let advancedLLMLogger = loggerFactory.CreateLogger<AdvancedLLMService>()
        let httpClient = new System.Net.Http.HttpClient()
        let advancedLLMService = AdvancedLLMService(advancedLLMLogger, httpClient)
        let superiorAILogger = loggerFactory.CreateLogger<SuperiorAICommand>()
        let superiorAICommand = SuperiorAICommand(superiorAILogger, advancedLLMService)

        // Real Autonomous Improvement Service and Command (temporarily disabled)
        // let realAutonomousLogger = loggerFactory.CreateLogger<RealAutonomousImprovementService>()
        // let realAutonomousService = RealAutonomousImprovementService(realAutonomousLogger, advancedLLMService)
        // let realAutoCommandLogger = loggerFactory.CreateLogger<RealAutonomousCommand>()
        // let realAutoCommand = RealAutonomousCommand(realAutoCommandLogger, realAutonomousService)



        // Unified Reasoning - Showcases all architectural improvements
        // let unifiedReasoningCommand = UnifiedReasoningCommand()

        // Ultimate Reasoning - Showcases ALL TARS capabilities
        // let ultimateReasoningCommand = UltimateReasoningCommand()

        // Interactive Multi-Agent Commands - Enhanced interactive capabilities
        // let interactiveMultiAgentCommand = InteractiveMultiAgentCommand()
        // let enhancedMultiAgentReasoningCommand = EnhancedMultiAgentReasoningCommand()
        // let scenarioReasoningCommand = ScenarioReasoningCommand()
        // let continuousReasoningCommand = ContinuousReasoningCommand()

        // Code Protection System
        let codeProtectionCommand = CodeProtectionCommand()
        let astAnalysisCommand = ASTAnalysisCommand()

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
        self.RegisterCommand(serverCommand)
        self.RegisterCommand(chatCommand)
        self.RegisterCommand(diagnosticsCommand)
        self.RegisterCommand(nonEuclideanCommand)
        self.RegisterCommand(conceptAnalysisCommand)
        self.RegisterCommand(fluxCommand)
        self.RegisterCommand(enhancedIntelligenceCommand)
        self.RegisterCommand(webSearchCommand)
        self.RegisterCommand(statusCommand)
        self.RegisterCommand(realAICommand)
        self.RegisterCommand(superiorAICommand)
        // self.RegisterCommand(realAutoCommand)  // Temporarily disabled
        // self.RegisterCommand(unifiedReasoningCommand)
        // self.RegisterCommand(ultimateReasoningCommand)
        self.RegisterCommand(demoCommand)
        self.RegisterCommand(webUICommand)
        self.RegisterCommand(codeProtectionCommand)
        self.RegisterCommand(astAnalysisCommand)
        let autoLoopLogger = loggerFactory.CreateLogger<AutoLoopCommand>()
        let autoLoopCommand = AutoLoopCommand(autoLoopLogger, selfImprovementService, loggerFactory)
        self.RegisterCommand(autoLoopCommand)

        // Create help command with all commands (must be last)
        let allCommands = self.GetAllCommands()
        let helpCommand = HelpCommand(allCommands)
        self.RegisterCommand(helpCommand)
