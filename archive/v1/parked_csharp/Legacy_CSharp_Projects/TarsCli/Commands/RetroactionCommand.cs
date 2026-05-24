using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for retroaction coding using metascripts and F#
/// </summary>
public class RetroactionCommand
{
    private readonly ILogger<RetroactionCommand> _logger;
    private readonly MetascriptEngine _metascriptEngine;
    private readonly DynamicFSharpCompilerService _fsharpCompiler;
    private readonly TransformationLearningService _learningService;
    private readonly MultiAgentCollaborationService _collaborationService;

    public RetroactionCommand(
        ILogger<RetroactionCommand> logger,
        MetascriptEngine metascriptEngine,
        DynamicFSharpCompilerService fsharpCompiler,
        TransformationLearningService learningService,
        MultiAgentCollaborationService collaborationService)
    {
        _logger = logger;
        _metascriptEngine = metascriptEngine;
        _fsharpCompiler = fsharpCompiler;
        _learningService = learningService;
        _collaborationService = collaborationService;

        // Register sample agents
        RegisterSampleAgents();
    }

    /// <summary>
    /// Registers the retroaction commands
    /// </summary>
    public Command RegisterCommands()
    {
        var retroactionCommand = new Command("retroaction", "Retroaction coding commands");

        // Add transform command
        var transformCommand = new Command("transform", "Transform code using metascript rules");
        var transformPath = new Argument<string>("path", "Path to the file or directory to transform");
        var transformRules = new Option<string>(["--rules", "-r"], "Path to the metascript rules file");
        var transformOutput = new Option<string>(["--output", "-o"], "Output directory for transformed files");
        var transformRecursive = new Option<bool>(["--recursive", "-R"], () => false, "Process directories recursively");
        var transformPreview = new Option<bool>(["--preview", "-p"], () => false, "Preview changes without applying them");

        transformCommand.AddArgument(transformPath);
        transformCommand.AddOption(transformRules);
        transformCommand.AddOption(transformOutput);
        transformCommand.AddOption(transformRecursive);
        transformCommand.AddOption(transformPreview);
        transformCommand.SetHandler(TransformAsync, transformPath, transformRules, transformOutput, transformRecursive, transformPreview);
        retroactionCommand.AddCommand(transformCommand);

        // Add compile command
        var compileCommand = new Command("compile", "Compile F# code dynamically");
        var compilePath = new Argument<string>("path", "Path to the F# file to compile");
        var compileOutput = new Option<string>(["--output", "-o"], "Output assembly name");

        compileCommand.AddArgument(compilePath);
        compileCommand.AddOption(compileOutput);
        compileCommand.SetHandler(CompileAsync, compilePath, compileOutput);
        retroactionCommand.AddCommand(compileCommand);

        // Add generate command
        var generateCommand = new Command("generate", "Generate F# code from metascript rules");
        var generateRules = new Argument<string>("rules", "Path to the metascript rules file");
        var generateOutput = new Option<string>(["--output", "-o"], "Output F# file path");

        generateCommand.AddArgument(generateRules);
        generateCommand.AddOption(generateOutput);
        generateCommand.SetHandler(GenerateAsync, generateRules, generateOutput);
        retroactionCommand.AddCommand(generateCommand);

        // Add multi-agent commands
        var agentCommand = new Command("agent", "Multi-agent collaboration commands");
        var agentListCommand = new Command("list", "List registered agents");
        var agentAnalyzeCommand = new Command("analyze", "Analyze code using multiple agents");
        var agentTransformCommand = new Command("transform", "Transform code using multiple agents");
        var agentLearnCommand = new Command("learn", "Learn from code transformations");

        var agentAnalyzePath = new Argument<string>("path", "Path to the file to analyze");
        var agentTransformPath = new Argument<string>("path", "Path to the file to transform");
        var agentTransformOutput = new Option<string>(["--output", "-o"], "Output path for transformed file");
        var agentLearnOriginal = new Argument<string>("original", "Path to the original file");
        var agentLearnTransformed = new Argument<string>("transformed", "Path to the transformed file");
        var agentLearnAccepted = new Option<bool>(["--accepted", "-a"], () => true, "Whether the transformation was accepted");

        agentListCommand.SetHandler(ListAgentsAsync);
        agentAnalyzeCommand.AddArgument(agentAnalyzePath);
        agentAnalyzeCommand.SetHandler(AnalyzeWithAgentsAsync, agentAnalyzePath);
        agentTransformCommand.AddArgument(agentTransformPath);
        agentTransformCommand.AddOption(agentTransformOutput);
        agentTransformCommand.SetHandler(TransformWithAgentsAsync, agentTransformPath, agentTransformOutput);
        agentLearnCommand.AddArgument(agentLearnOriginal);
        agentLearnCommand.AddArgument(agentLearnTransformed);
        agentLearnCommand.AddOption(agentLearnAccepted);
        agentLearnCommand.SetHandler(LearnFromTransformationAsync, agentLearnOriginal, agentLearnTransformed, agentLearnAccepted);

        agentCommand.AddCommand(agentListCommand);
        agentCommand.AddCommand(agentAnalyzeCommand);
        agentCommand.AddCommand(agentTransformCommand);
        agentCommand.AddCommand(agentLearnCommand);
        retroactionCommand.AddCommand(agentCommand);

        return retroactionCommand;
    }

    /// <summary>
    /// Transforms code using metascript rules
    /// </summary>
    private async Task<int> TransformAsync(
        string path,
        string rulesPath,
        string outputPath,
        bool recursive,
        bool preview)
    {
        try
        {
            Console.WriteLine($"Transforming {path} using rules from {rulesPath}");

            if (!File.Exists(rulesPath))
            {
                Console.WriteLine($"Rules file not found: {rulesPath}");
                return 1;
            }

            // Parse the metascript rules
            var metascriptContent = await File.ReadAllTextAsync(rulesPath);
            var rules = _metascriptEngine.ParseMetascript(metascriptContent);

            Console.WriteLine($"Loaded {rules.Count} transformation rules:");
            foreach (var rule in rules)
            {
                Console.WriteLine($"  - {rule.Name}: {rule.Description}");
            }

            // Process files
            if (File.Exists(path))
            {
                // Transform a single file
                await TransformFileAsync(path, rules, outputPath, preview);
            }
            else if (Directory.Exists(path))
            {
                // Transform a directory
                var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
                var files = Directory.GetFiles(path, "*.cs", searchOption);

                Console.WriteLine($"Found {files.Length} C# files to transform");

                foreach (var file in files)
                {
                    await TransformFileAsync(file, rules, outputPath, preview);
                }
            }
            else
            {
                Console.WriteLine($"Path not found: {path}");
                return 1;
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error transforming code: {ex.Message}");
            _logger.LogError(ex, "Error in TransformAsync");
            return 1;
        }
    }

    /// <summary>
    /// Transforms a single file using metascript rules
    /// </summary>
    private async Task TransformFileAsync(
        string filePath,
        List<MetascriptEngine.TransformationRule> rules,
        string outputPath,
        bool preview)
    {
        try
        {
            Console.WriteLine($"Transforming file: {filePath}");

            // Read the file
            var code = await File.ReadAllTextAsync(filePath);

            // Apply transformations
            var transformedCode = await _metascriptEngine.ApplyTransformationsAsync(code, rules);

            // Check if any changes were made
            var hasChanges = code != transformedCode;

            if (!hasChanges)
            {
                Console.WriteLine("  No changes needed");
                return;
            }

            // Determine output path
            string outputFilePath;
            if (!string.IsNullOrEmpty(outputPath))
            {
                // If outputPath is a directory, use it as the base directory
                if (Directory.Exists(outputPath))
                {
                    var relativePath = Path.GetRelativePath(Directory.GetCurrentDirectory(), filePath);
                    outputFilePath = Path.Combine(outputPath, relativePath);

                    // Ensure directory exists
                    Directory.CreateDirectory(Path.GetDirectoryName(outputFilePath));
                }
                else
                {
                    // If outputPath is a file path, use it directly
                    outputFilePath = outputPath;
                }
            }
            else
            {
                // If no output path is specified, use the original file path
                outputFilePath = filePath;
            }

            if (preview)
            {
                // Show a preview of the changes
                Console.WriteLine("  Changes preview:");
                ShowDiff(code, transformedCode);
            }
            else
            {
                // Apply the changes
                await File.WriteAllTextAsync(outputFilePath, transformedCode);
                Console.WriteLine($"  Transformed file saved to: {outputFilePath}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Error transforming file {filePath}: {ex.Message}");
            _logger.LogError(ex, $"Error transforming file {filePath}");
        }
    }

    /// <summary>
    /// Compiles F# code dynamically
    /// </summary>
    private async Task<int> CompileAsync(string path, string outputName)
    {
        try
        {
            Console.WriteLine($"Compiling F# code from {path}");

            if (!File.Exists(path))
            {
                Console.WriteLine($"File not found: {path}");
                return 1;
            }

            // Read the F# code
            var fsharpCode = await File.ReadAllTextAsync(path);

            // Use the file name without extension as the assembly name if not specified
            if (string.IsNullOrEmpty(outputName))
            {
                outputName = Path.GetFileNameWithoutExtension(path);
            }

            // Compile the F# code
            var assembly = await _fsharpCompiler.CompileFSharpCodeAsync(fsharpCode, outputName);

            Console.WriteLine($"Successfully compiled to assembly: {assembly.FullName}");
            Console.WriteLine("Available types:");

            // Show the types in the assembly
            foreach (var type in assembly.GetExportedTypes())
            {
                Console.WriteLine($"  - {type.FullName}");

                // Show public static methods
                var methods = type.GetMethods(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
                if (methods.Length > 0)
                {
                    Console.WriteLine("    Static methods:");
                    foreach (var method in methods)
                    {
                        Console.WriteLine($"      - {method.Name}");
                    }
                }
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error compiling F# code: {ex.Message}");
            _logger.LogError(ex, "Error in CompileAsync");
            return 1;
        }
    }

    /// <summary>
    /// Generates F# code from metascript rules
    /// </summary>
    private async Task<int> GenerateAsync(string rulesPath, string outputPath)
    {
        try
        {
            Console.WriteLine($"Generating F# code from rules in {rulesPath}");

            if (!File.Exists(rulesPath))
            {
                Console.WriteLine($"Rules file not found: {rulesPath}");
                return 1;
            }

            // Parse the metascript rules
            var metascriptContent = await File.ReadAllTextAsync(rulesPath);
            var rules = _metascriptEngine.ParseMetascript(metascriptContent);

            // Generate F# code
            var fsharpCode = _metascriptEngine.GenerateFSharpCode(rules);

            // Determine output path
            if (string.IsNullOrEmpty(outputPath))
            {
                outputPath = Path.ChangeExtension(rulesPath, ".fs");
            }

            // Save the F# code
            await File.WriteAllTextAsync(outputPath, fsharpCode);
            Console.WriteLine($"Generated F# code saved to: {outputPath}");

            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error generating F# code: {ex.Message}");
            _logger.LogError(ex, "Error in GenerateAsync");
            return 1;
        }
    }

    /// <summary>
    /// Shows a simple diff between two strings
    /// </summary>
    private void ShowDiff(string original, string modified)
    {
        var originalLines = original.Split('\n');
        var modifiedLines = modified.Split('\n');

        // Use a simple line-by-line comparison
        var diff = new List<(int LineNumber, string OriginalLine, string ModifiedLine)>();

        for (var i = 0; i < Math.Max(originalLines.Length, modifiedLines.Length); i++)
        {
            var originalLine = i < originalLines.Length ? originalLines[i] : "";
            var modifiedLine = i < modifiedLines.Length ? modifiedLines[i] : "";

            if (originalLine != modifiedLine)
            {
                diff.Add((i + 1, originalLine, modifiedLine));
            }
        }

        // Show the diff
        foreach (var (lineNumber, originalLine, modifiedLine) in diff)
        {
            Console.WriteLine($"    Line {lineNumber}:");
            Console.WriteLine($"      - {originalLine}");
            Console.WriteLine($"      + {modifiedLine}");
            Console.WriteLine();
        }
    }

    /// <summary>
    /// Registers sample agents for demonstration
    /// </summary>
    // Adapter classes to bridge between TarsEngineFSharp.SampleAgents and TarsCli.Services.ICodeAgent
    private class NullReferenceAnalysisAgentAdapter : ICodeAgent, IAnalysisAgent
    {
        public string Name => "NullReferenceAnalyzer";
        public string Description => "Analyzes code for potential null reference exceptions";
        public AgentRole Role => AgentRole.Analysis;

        public async Task<AnalysisResult> AnalyzeAsync(string filePath, string code)
        {
            // Create a simple analysis result
            var issues = new List<CodeIssue>
            {
                new()
                {
                    Type = IssueType.MissingExceptionHandling,
                    Location = "Line 10",
                    Description = "Potential null reference exception"
                }
            };

            return new AnalysisResult
            {
                Issues = issues
            };
        }
    }

    private class IneffectiveLoopAnalysisAgentAdapter : ICodeAgent, IAnalysisAgent
    {
        public string Name => "IneffectiveLoopAnalyzer";
        public string Description => "Analyzes code for loops that could be replaced with LINQ";
        public AgentRole Role => AgentRole.Analysis;

        public async Task<AnalysisResult> AnalyzeAsync(string filePath, string code)
        {
            // Create a simple analysis result
            var issues = new List<CodeIssue>
            {
                new()
                {
                    Type = IssueType.IneffectiveCode,
                    Location = "Line 20",
                    Description = "Loop could be replaced with LINQ"
                }
            };

            return new AnalysisResult
            {
                Issues = issues
            };
        }
    }

    private class NullCheckTransformationAgentAdapter : ICodeAgent, ITransformationAgent
    {
        public string Name => "NullCheckTransformer";
        public string Description => "Transforms code to add null checks";
        public AgentRole Role => AgentRole.Transformation;

        public async Task<TransformationResult> TransformAsync(string filePath, string code, CollaborativeAnalysisResult analysisResult)
        {
            // Simple transformation - add a null check
            var transformedCode = code;
            if (!code.Contains("if (x == null) throw new ArgumentNullException(nameof(x));"))
            {
                transformedCode = code.Replace("void Process(string x)", "void Process(string x)\n    {\n        if (x == null) throw new ArgumentNullException(nameof(x));");
            }

            return new TransformationResult
            {
                TransformedCode = transformedCode,
                Success = true,
                TransformationType = "NullCheck",
                Location = filePath
            };
        }
    }

    private class LinqTransformationAgentAdapter : ICodeAgent, ITransformationAgent
    {
        public string Name => "LinqTransformer";
        public string Description => "Transforms code to replace loops with LINQ";
        public AgentRole Role => AgentRole.Transformation;

        public async Task<TransformationResult> TransformAsync(string filePath, string code, CollaborativeAnalysisResult analysisResult)
        {
            // Simple transformation - replace a loop with LINQ
            var transformedCode = code;
            if (code.Contains("for (int i = 0; i < items.Count; i++)"))
            {
                transformedCode = code.Replace(
                    "for (int i = 0; i < items.Count; i++)\n    {\n        sum += items[i];\n    }",
                    "sum = items.Sum();");
            }

            return new TransformationResult
            {
                TransformedCode = transformedCode,
                Success = true,
                TransformationType = "LinqTransformation",
                Location = filePath
            };
        }
    }

    private void RegisterSampleAgents()
    {
        try
        {
            // Create and register sample agents using adapters
            var nullAnalyzer = new NullReferenceAnalysisAgentAdapter();
            var loopAnalyzer = new IneffectiveLoopAnalysisAgentAdapter();
            var nullTransformer = new NullCheckTransformationAgentAdapter();
            var linqTransformer = new LinqTransformationAgentAdapter();

            _collaborationService.RegisterAgent(nullAnalyzer);
            _collaborationService.RegisterAgent(loopAnalyzer);
            _collaborationService.RegisterAgent(nullTransformer);
            _collaborationService.RegisterAgent(linqTransformer);

            _logger.LogInformation("Registered sample agents");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error registering sample agents");
        }
    }

    /// <summary>
    /// Lists all registered agents
    /// </summary>
    private async Task<int> ListAgentsAsync()
    {
        try
        {
            Console.WriteLine("Registered agents:");

            var agents = _collaborationService.GetRegisteredAgents();

            if (agents.Count == 0)
            {
                Console.WriteLine("  No agents registered");
                return 0;
            }

            // Group agents by role
            var agentsByRole = agents.GroupBy(a => a.Role);

            foreach (var group in agentsByRole)
            {
                Console.WriteLine($"\n{group.Key} Agents:");

                foreach (var agent in group)
                {
                    Console.WriteLine($"  - {agent.Name}: {agent.Description}");
                }
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error listing agents: {ex.Message}");
            _logger.LogError(ex, "Error in ListAgentsAsync");
            return 1;
        }
    }

    /// <summary>
    /// Analyzes code using multiple agents
    /// </summary>
    private async Task<int> AnalyzeWithAgentsAsync(string path)
    {
        try
        {
            Console.WriteLine($"Analyzing {path} using multiple agents...");

            if (!File.Exists(path))
            {
                Console.WriteLine($"File not found: {path}");
                return 1;
            }

            // Analyze the file
            var result = await _collaborationService.AnalyzeFileAsync(path);

            // Display results
            Console.WriteLine($"\nAnalysis results for {result.FilePath}:");

            if (result.Issues.Count == 0)
            {
                Console.WriteLine("  No issues found");
            }
            else
            {
                Console.WriteLine($"\nIssues ({result.Issues.Count}):");

                // Group issues by type
                var issuesByType = result.Issues.GroupBy(i => i.Type);

                foreach (var group in issuesByType)
                {
                    Console.WriteLine($"\n  {group.Key}:");

                    foreach (var issue in group)
                    {
                        Console.WriteLine($"    - {issue.Description}");
                        Console.WriteLine($"      Location: {issue.Location}");
                        // Severity is not available in this version
                        // Console.WriteLine($"      Severity: {issue.Severity}");
                    }
                }
            }

            if (result.Suggestions.Count > 0)
            {
                Console.WriteLine($"\nSuggestions ({result.Suggestions.Count}):");

                // Group suggestions by type
                var suggestionsByType = result.Suggestions.GroupBy(s => s.Type);

                foreach (var group in suggestionsByType)
                {
                    Console.WriteLine($"\n  {group.Key}:");

                    foreach (var suggestion in group)
                    {
                        Console.WriteLine($"    - {suggestion.Description}");
                        Console.WriteLine($"      Location: {suggestion.Location}");
                        Console.WriteLine($"      Confidence: {suggestion.Confidence:P0}");
                    }
                }
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error analyzing code: {ex.Message}");
            _logger.LogError(ex, "Error in AnalyzeWithAgentsAsync");
            return 1;
        }
    }

    /// <summary>
    /// Transforms code using multiple agents
    /// </summary>
    private async Task<int> TransformWithAgentsAsync(string path, string outputPath)
    {
        try
        {
            Console.WriteLine($"Transforming {path} using multiple agents...");

            if (!File.Exists(path))
            {
                Console.WriteLine($"File not found: {path}");
                return 1;
            }

            // Transform the file
            var result = await _collaborationService.TransformFileAsync(path, outputPath);

            if (!result.Success)
            {
                Console.WriteLine("No transformations were applied");
                return 0;
            }

            Console.WriteLine($"\nTransformation results:");
            Console.WriteLine($"  Applied {result.AppliedTransformations.Count} transformations");

            foreach (var transformation in result.AppliedTransformations)
            {
                Console.WriteLine($"    - {transformation.TransformationType} by {transformation.AgentName}");
                Console.WriteLine($"      Location: {transformation.Location}");
            }

            Console.WriteLine($"\nTransformed file saved to: {result.TransformedFilePath}");

            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error transforming code: {ex.Message}");
            _logger.LogError(ex, "Error in TransformWithAgentsAsync");
            return 1;
        }
    }

    /// <summary>
    /// Learns from a code transformation
    /// </summary>
    private async Task<int> LearnFromTransformationAsync(string originalPath, string transformedPath, bool accepted)
    {
        try
        {
            Console.WriteLine($"Learning from transformation: {originalPath} -> {transformedPath} (accepted: {accepted})");

            if (!File.Exists(originalPath))
            {
                Console.WriteLine($"Original file not found: {originalPath}");
                return 1;
            }

            if (!File.Exists(transformedPath))
            {
                Console.WriteLine($"Transformed file not found: {transformedPath}");
                return 1;
            }

            // Read the files
            var originalCode = await File.ReadAllTextAsync(originalPath);
            var transformedCode = await File.ReadAllTextAsync(transformedPath);

            // Find learning agents
            var learningAgents = _collaborationService.GetAgentsByRole(AgentRole.Learning).ToList();

            if (learningAgents.Count == 0)
            {
                Console.WriteLine("No learning agents registered");
                return 1;
            }

            // Learn from the transformation
            foreach (var agent in learningAgents)
            {
                if (agent is ILearningAgent learningAgent)
                {
                    await learningAgent.LearnFromTransformationAsync(originalCode, transformedCode, accepted);
                    Console.WriteLine($"  Agent {agent.Name} learned from the transformation");
                }
            }

            // Get rule statistics
            var statistics = _learningService.GetRuleStatistics();

            Console.WriteLine("\nRule statistics:");

            foreach (var stat in statistics)
            {
                var successRate = stat.Value.TotalApplications > 0
                    ? (double)stat.Value.SuccessfulApplications / stat.Value.TotalApplications
                    : 0;

                Console.WriteLine($"  - {stat.Key}: {stat.Value.SuccessfulApplications}/{stat.Value.TotalApplications} ({successRate:P0})");
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error learning from transformation: {ex.Message}");
            _logger.LogError(ex, "Error in LearnFromTransformationAsync");
            return 1;
        }
    }
}