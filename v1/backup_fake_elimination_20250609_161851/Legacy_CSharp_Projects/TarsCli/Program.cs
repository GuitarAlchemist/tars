using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using NLog;
using NLog.Extensions.Logging;
using System.CommandLine.Builder;
using System.CommandLine.Parsing;
using TarsCli.Controllers;
using TarsCli.Extensions;
using TarsCli.Services;
using TarsCli.Services.Mcp;
using TarsEngine.Consciousness;
using TarsEngine.Extensions;
using TarsEngine.Interfaces.Compilation;

namespace TarsCli;

internal static class Program
{
    // Main method
    public static async Task<int> Main(string[] args)
    {
        // Setup NLog
        LogManager.Setup()
            .LoadConfigurationFromFile("nlog.config")
            .GetCurrentClassLogger();

        try
        {
            // Setup configuration
            var configuration = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: false)
                .AddEnvironmentVariables()
                .AddCommandLine(args)
                .Build();

            // Create logs directory if it doesn't exist
            Directory.CreateDirectory(Path.Combine(Directory.GetCurrentDirectory(), "logs"));
            Directory.CreateDirectory(Path.Combine(Directory.GetCurrentDirectory(), "logs", "archives"));

            // Setup DI
            var serviceProvider = new ServiceCollection()
                .AddLogging(builder =>
                {
                    builder.ClearProviders();
                    builder.SetMinimumLevel(Microsoft.Extensions.Logging.LogLevel.Trace);
                    builder.AddNLog(configuration);
                })
                .AddSingleton<IConfiguration>(configuration)
                .AddHttpClient()
                .AddSingleton<GpuService>()
                .AddSingleton<OllamaService>()
                .AddSingleton<OllamaSetupService>()
                .AddSingleton<RetroactionService>()
                .AddSingleton<RetroactionLoopService>()
                .AddSingleton<DiagnosticsService>()
                .AddSingleton<DiagnosticReportService>()
                .AddSingleton<KnowledgeApplicationService>()
                .AddSingleton<KnowledgeIntegrationService>()
                .AddSingleton<CompilationService>()
                .AddSingleton<TestRunnerService>()
                .AddSingleton<SmartKnowledgeApplicationService>()
                .AddSingleton<KnowledgePrioritizationService>()
                .AddSingleton<GitService>()
                .AddSingleton<DynamicFSharpCompilerService>()
                .AddSingleton<MetascriptEngine>()
                .AddSingleton<TransformationLearningService>()
                .AddSingleton<MultiAgentCollaborationService>()
                .AddSingleton<DistributedAgentService>()
                .AddSingleton<AiCodeUnderstandingService>()
                .AddSingleton<LearningService>()
                .AddSingleton<Commands.RetroactionCommand>()
                .AddSingleton<SelfImprovementService>()
                .AddSingleton<ScriptExecutionService>()
                .AddSingleton<TemplateService>()
                .AddSingleton<SessionService>()
                .AddSingleton<WorkflowCoordinationService>()
                .AddSingleton<HuggingFaceService>()
                .AddSingleton<LanguageSpecificationService>()
                .AddSingleton<DocumentationService>()
                .AddSingleton<DemoService>()
                .AddSingleton<SecretsService>()
                .AddSingleton<UserInteractionService>()
                .AddSingleton<AutoImprovementService>()
                .AddSingleton<SlackIntegrationService>()
                .AddSingleton<TarsSpeechService>()
                .AddSingleton<ConversationLoggingService>()
                .AddSingleton<ChatBotService>()
                .AddSingleton<ChatWebSocketService>()
                .AddSingleton<ExplorationReflectionService>()
                .AddSingleton<ExplorationReflectionService2>()
                .AddSingleton<DeepThinkingService>()
                .AddSingleton<McpService>()
                .AddSingleton<TarsMcpService>()
                .AddSingleton<ReplicaCommunicationProtocol>()
                .AddSingleton<ResponseProcessor>()
                .AddSingleton<IMcpActionHandler, AnalyzerReplicaActionHandler>()
                .AddSingleton<IMcpActionHandler, GeneratorReplicaActionHandler>()
                .AddSingleton<IMcpActionHandler, TesterReplicaActionHandler>()
                .AddSingleton<IMcpActionHandler, CoordinatorReplicaActionHandler>()
                .AddTarsServices()
                .AddSingleton<TarsReplicaManager>()
                .AddSingleton<SelfCodingWorkflow>()
                .AddSingleton<TarsA2AService>()

                // Code Analysis services
                .AddSingleton<Services.CodeAnalysis.SecurityVulnerabilityAnalyzer>()
                .AddSingleton<Services.CodeAnalysis.ICodeAnalyzer, Services.CodeAnalysis.CSharpCodeAnalyzer>()
                .AddSingleton<Services.CodeAnalysis.ICodeAnalyzer, Services.CodeAnalysis.FSharpCodeAnalyzer>()
                .AddSingleton<Services.CodeAnalysis.CodeAnalyzerService>()
                .AddSingleton<Services.CodeAnalysis.ImprovementSuggestionGenerator>()

                // Code Generation services
                .AddSingleton<Services.CodeGeneration.ICodeGenerator, Services.CodeGeneration.CSharpCodeGenerator>()
                .AddSingleton<Services.CodeGeneration.ICodeGenerator, Services.CodeGeneration.FSharpCodeGenerator>()
                .AddSingleton<Services.CodeGeneration.CodeGeneratorService>()

                // Testing services
                .AddSingleton<Services.Testing.ITestPatternRepository, Services.Testing.TestPatternRepository>()
                .AddSingleton<Services.Testing.ITestGenerator, Services.Testing.CSharpTestGenerator>()
                .AddSingleton<Services.Testing.ITestGenerator, Services.Testing.ImprovedCSharpTestGenerator>()
                .AddSingleton<Services.Testing.ITestGenerator, Services.Testing.FSharpTestGenerator>()
                .AddSingleton<Services.Testing.TestGeneratorService>()
                .AddSingleton<Services.Testing.TestRunnerService>()
                .AddSingleton<Services.Testing.TestResultAnalyzer>()

                // Workflow services
                .AddSingleton<Services.Workflow.IWorkflowDefinition, Services.Workflow.SelfCodingWorkflowDefinition>()
                .AddSingleton<Services.Workflow.IWorkflowCoordinator, Services.Workflow.WorkflowCoordinator>()
                .AddSingleton<Services.Workflow.TaskPrioritizer>()
                .AddSingleton<Services.Workflow.ProgressTracker>()

                // Self-coding services
                .AddSingleton<Services.SelfCoding.FileProcessor>()
                .AddSingleton<Services.SelfCoding.AnalysisProcessor>()
                .AddSingleton<Services.SelfCoding.CodeGenerationProcessor>()
                .AddSingleton<Services.SelfCoding.TestProcessor>()
                .AddSingleton<Services.SelfCoding.SelfCodingWorkflow>()
                .AddSingleton<Commands.A2ACommands>()
                .AddSingleton<Commands.McpSwarmCommand>()
                .AddSingleton<Commands.SwarmSelfImprovementCommand>()
                .AddSingleton<Commands.SelfCodingCommand>()
                .AddSingleton<Commands.TestGeneratorCommand>()
                .AddSingleton<LearningPlanService>()
                .AddSingleton<CourseGeneratorService>()
                .AddSingleton<TutorialOrganizerService>()
                .AddSingleton<ConsoleCaptureService>()
                .AddSingleton<DslService>()
                .AddSingleton<DslDebuggerService>()
                .AddSingleton<DockerModelRunnerService>()
                .AddSingleton<DockerAIAgentService>()
                .AddSingleton<ModelProviderFactory>()
                .AddSingleton<ConsoleService>()
                .AddSingleton<ConfigurationService>()
                .AddSingleton<DocumentationKnowledgeService>()
                .AddSingleton<OperationSummaryService>()
                .AddSingleton<TarsEngine.Consciousness.Intelligence.IntelligenceSpark>()
                .AddSingleton<TarsEngine.ML.Core.IntelligenceMeasurement>()
                .AddSingleton<TarsEngine.Consciousness.Intelligence.Pattern.ImplicitPatternRecognition>()
                .AddSingleton<TarsEngine.Consciousness.Intelligence.Heuristic.HeuristicReasoning>()
                .AddSingleton<TarsEngine.Consciousness.Intelligence.IntuitiveReasoning>()
                .AddSingleton<TarsEngine.Consciousness.Intelligence.SpontaneousThought>()
                .AddSingleton<TarsEngine.Consciousness.Intelligence.CuriosityDrive>()
                .AddSingleton<TarsEngine.Consciousness.Intelligence.InsightGeneration>()
                .AddSingleton<TarsEngine.Consciousness.Intelligence.CreativeThinking>()
                .AddSingleton<KnowledgeVisualizationService>()
                .AddSingleton<KnowledgeTestGenerationService>()
                .AddSingleton<AutonomousImprovementService>()
                .AddSingleton<TarsEngine.Services.Interfaces.ICodeAnalysisService, TarsEngine.Services.CodeAnalysisService>()
                .AddSingleton<TarsEngine.Services.Interfaces.IProjectAnalysisService, TarsEngine.Services.SimpleProjectAnalysisService>()
                .AddSingleton<TarsEngine.Services.Interfaces.ILlmService, TarsEngine.Services.LlmService>()
                .AddSingleton<TarsEngine.Services.Interfaces.ICodeGenerationService, TarsEngine.Services.CodeGenerationService>()
                .AddSingleton<TarsEngine.Services.CodeExecutionService>()
                .AddSingleton<TarsEngine.Services.LearningService>()
                .AddSingleton<TarsEngine.Services.Interfaces.ISelfImprovementService, TarsEngine.Services.SelfImprovementService>()
                .AddSingleton<CollaborationService>()
                .AddSingleton<TarsEngine.Services.Interfaces.IMetascriptService, TarsEngine.Services.MetascriptService>()
                .AddSingleton<TarsEngine.Services.Interfaces.ITestGenerationService, TarsEngine.Services.TestGenerationService>()
                .AddSingleton<TarsEngine.Services.Interfaces.ITestValidationService, TarsEngine.Services.TestValidationService>()
                .AddSingleton<TarsEngine.Services.Interfaces.IRegressionTestingService, TarsEngine.Services.RegressionTestingService>()
                .AddSingleton<TarsEngine.Services.Interfaces.ICodeQualityService, TarsEngine.Services.CodeQualityService>()
                .AddSingleton<TarsEngine.Services.Interfaces.IComplexityAnalysisService, TarsEngine.Services.ComplexityAnalysisService>()
                .AddSingleton<TarsEngine.Services.Interfaces.IReadabilityService, TarsEngine.Services.ReadabilityService>()
                .AddSingleton<TarsEngine.Services.Interfaces.IDocumentParserService, TarsEngine.Services.DocumentParserService>()
                .AddSingleton<TarsEngine.Services.Interfaces.IContentClassifierService, TarsEngine.Services.ContentClassifierService>()
                .AddSingleton<TarsEngine.Services.Interfaces.IKnowledgeExtractorService, TarsEngine.Services.KnowledgeExtractorService>()
                .AddSingleton<TarsEngine.Services.Interfaces.IKnowledgeRepository, TarsEngine.Services.KnowledgeRepository>()
                .AddSingleton<TarsEngine.Data.KnowledgeDbContext>()

                // Improvement Generation System services
                .AddSingleton<TarsEngine.Services.Interfaces.ICodeAnalyzerService, TarsEngine.Services.CodeAnalyzerService>()
                .AddSingleton<TarsEngine.Services.CodeSmellDetector>()
                .AddSingleton<TarsEngine.Services.ComplexityAnalyzer>()
                .AddSingleton<TarsEngine.Services.PerformanceAnalyzer>()
                .AddSingleton<TarsEngine.Services.GenericAnalyzer>()
                .AddSingleton<TarsEngine.Services.CSharpAnalyzer>()
                .AddSingleton<TarsEngine.Services.FSharpAnalyzer>()
                .AddSingleton<TarsEngine.Services.ImprovementSuggestionGenerator>()
                .AddSingleton<TarsEngine.Services.SecurityVulnerabilityAnalyzer>()
                .AddSingleton<TarsEngine.Services.ImprovementPrioritizer>()

                .AddSingleton<TarsEngine.Services.Interfaces.IPatternMatcherService, TarsEngine.Services.PatternMatcherService>()
                .AddSingleton<TarsEngine.Services.PatternLanguage>()
                .AddSingleton<TarsEngine.Services.PatternParser>()
                .AddSingleton<TarsEngine.Services.PatternMatcher>()
                .AddSingleton<TarsEngine.Services.FuzzyMatcher>()
                .AddSingleton<TarsEngine.Data.PatternLibrary>(sp => new TarsEngine.Data.PatternLibrary(
                    sp.GetRequiredService<ILogger<TarsEngine.Data.PatternLibrary>>(),
                    sp.GetRequiredService<TarsEngine.Services.PatternParser>(),
                    Path.Combine(Directory.GetCurrentDirectory(), "TarsEngine", "Data", "Patterns")))

                .AddSingleton<TarsEngine.Services.Interfaces.IMetascriptGeneratorService, TarsEngine.Services.MetascriptGeneratorService>()
                .AddSingleton<TarsEngine.Services.MetascriptTemplateService>(sp => new TarsEngine.Services.MetascriptTemplateService(
                    sp.GetRequiredService<ILogger<TarsEngine.Services.MetascriptTemplateService>>(),
                    Path.Combine(Directory.GetCurrentDirectory(), "TarsEngine", "Data", "Templates")))
                .AddSingleton<TarsEngine.Services.TemplateFiller>()
                .AddSingleton<TarsEngine.Services.ParameterOptimizer>()
                // REAL IMPLEMENTATION NEEDED
                .AddSingleton<TarsEngine.Interfaces.Compilation.IFSharpCompiler>(sp => new MockFSharpCompiler())
                .AddSingleton<TarsEngine.Services.MetascriptSandbox>(sp => new TarsEngine.Services.MetascriptSandbox(
                    sp.GetRequiredService<ILogger<TarsEngine.Services.MetascriptSandbox>>(),
                    sp.GetRequiredService<IConfiguration>(),
                    sp.GetRequiredService<TarsEngine.Interfaces.Compilation.IFSharpCompiler>()))

                .AddSingleton<TarsEngine.Services.Interfaces.IImprovementPrioritizerService, TarsEngine.Services.ImprovementPrioritizerService>()
                .AddSingleton<TarsEngine.Services.ImprovementScorer>()
                .AddSingleton<TarsEngine.Services.StrategicAlignmentService>(sp => new TarsEngine.Services.StrategicAlignmentService(
                    sp.GetRequiredService<ILogger<TarsEngine.Services.StrategicAlignmentService>>(),
                    Path.Combine(Directory.GetCurrentDirectory(), "TarsEngine", "Data", "Goals")))
                .AddSingleton<TarsEngine.Services.DependencyGraphService>()
                .AddSingleton<TarsEngine.Services.ImprovementQueue>((sp) => new TarsEngine.Services.ImprovementQueue(
                    sp.GetRequiredService<ILogger<TarsEngine.Services.ImprovementQueue>>(),
                    Path.Combine(Directory.GetCurrentDirectory(), "TarsEngine", "Data", "Improvements"),
                    sp.GetRequiredService<TarsEngine.Services.DependencyGraphService>()))

                // Improvement Generation System CLI commands
                .AddSingleton<Commands.ImprovementGenerationCommand>()
                .AddSingleton<Commands.ImprovementWorkflowCommand>()
                .AddSingleton<SelfImprovementController>()
                .AddSingleton<Mcp.McpController>(sp => new Mcp.McpController(
                    sp.GetRequiredService<ILogger<Mcp.McpController>>(),
                    configuration))

                // Consciousness services
                .AddConsciousnessServices()

                // Intelligence progression measurement services
                .AddIntelligenceProgressionMeasurement()

                // Code complexity command
                .AddSingleton<Commands.CodeComplexityCommand>()

                // Operation Summary Service
                .AddSingleton<OperationSummaryService>()

                // VS Code Control Command
                .AddSingleton<Commands.VSCodeControlCommand>()

                // Augment VS Code Demo Command
                .AddSingleton<Commands.AugmentVSCodeDemoCommand>()

                // Agent implementations
                .AddSingleton<Services.Agents.CodeAnalyzerAgent>()
                .AddSingleton<Services.Agents.ProjectManagerAgent>()

                .BuildServiceProvider();

            // Get services
            var logger = serviceProvider.GetRequiredService<ILoggerFactory>().CreateLogger("TarsCli");
            var retroactionService = serviceProvider.GetRequiredService<RetroactionService>();
            var diagnosticsService = serviceProvider.GetRequiredService<DiagnosticsService>();
            var setupService = serviceProvider.GetRequiredService<OllamaSetupService>();

            // Run initial diagnostics
            logger.LogInformation("Running initial diagnostics...");
            var diagnosticsResult = await diagnosticsService.RunInitialDiagnosticsAsync();

            // Skip confirmation prompt for diagnostics command
            var skipConfirmation = args.Length > 0 &&
                                   (args[0].Equals("diagnostics", StringComparison.OrdinalIgnoreCase) ||
                                    args[0].Equals("setup", StringComparison.OrdinalIgnoreCase));

            if (!diagnosticsResult.IsReady && !skipConfirmation)
            {
                logger.LogWarning("System is not fully ready for TARS operations");
                logger.LogInformation("You can still proceed, but some features may not work as expected");

                // Prompt user to continue
                Console.Write("Do you want to continue anyway? (y/n): ");
                var response = Console.ReadLine()?.ToLower();
                if (response != "y" && response != "yes")
                {
                    logger.LogInformation("Operation cancelled by user");
                    return 1;
                }
            }
            else if (!diagnosticsResult.IsReady)
            {
                logger.LogWarning("System is not fully ready for TARS operations");
                logger.LogInformation("Proceeding with command execution anyway...");
            }
            else
            {
                logger.LogInformation("System is ready for TARS operations");
            }

            // Check if we should use Docker for Ollama
            var useDocker = configuration.GetValue<bool>("Ollama:UseDocker", false);
            var skipOllamaCheck = configuration.GetValue<bool>("OLLAMA_SKIP_SETUP", false) ||
                                  Environment.GetEnvironmentVariable("OLLAMA_SKIP_SETUP") == "true";

            if (useDocker || skipOllamaCheck)
            {
                logger.LogInformation("Using Docker for Ollama operations - skipping Ollama setup check");
                Console.WriteLine("Using Docker for Ollama operations - skipping Ollama setup check");
            }
            else
            {
                // Check Ollama setup
                if (!await setupService.CheckOllamaSetupAsync())
                {
                    logger.LogError("Failed to set up Ollama. Please run the Install-Prerequisites.ps1 script or set up Ollama manually.");
                    logger.LogInformation("You can find the script in the Scripts directory.");
                    return 1;
                }
            }

            // Setup and run command line
            Console.WriteLine("Setting up command line...");
            var rootCommand = CliSupport.SetupCommandLine(
                configuration,
                logger,
                serviceProvider.GetRequiredService<ILoggerFactory>(),
                diagnosticsService,
                retroactionService,
                setupService,
                serviceProvider);  // setupService is the OllamaSetupService instance

            Console.WriteLine($"Invoking command: {string.Join(" ", args)}");

            // Check if we need to use the triple-quoted parser
            if (args.Any(arg => arg.Contains("\"\"\"") || args.Contains("-triple-quoted")))
            {
                // Create the parser
                var parser = new CommandLineBuilder(rootCommand)
                    .UseDefaults()
                    .Build();

                // Use our custom parser for triple-quoted strings
                var parseResult = Parsing.TripleQuotedArgumentParser.ParseCommandLine(parser, args);
                var result = await parseResult.InvokeAsync();
                return result;
            }
            else
            {
                // Use the standard parser for regular commands
                var result = await rootCommand.InvokeAsync(args);
                Console.WriteLine($"Command completed with result: {result}");
                return result;
            }
        }
        catch (Exception ex)
        {
            // Log any startup exceptions
            Console.WriteLine($"CRITICAL ERROR: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
            LogManager.GetCurrentClassLogger().Error(ex, "Stopped program because of exception");
            return 1;
        }
        finally
        {
            // Ensure to flush and stop internal timers/threads before application-exit
            LogManager.Shutdown();
        }
    }
}
