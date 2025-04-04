using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for generating tests from the knowledge base
/// </summary>
public class KnowledgeTestGenerationCommand : Command
{
    private readonly ILogger<KnowledgeTestGenerationCommand> _logger;
    private readonly KnowledgeTestGenerationService _knowledgeTestGenerationService;
    private readonly ConsoleService _consoleService;

    /// <summary>
    /// Initializes a new instance of the <see cref="KnowledgeTestGenerationCommand"/> class.
    /// </summary>
    public KnowledgeTestGenerationCommand(
        ILogger<KnowledgeTestGenerationCommand> logger,
        KnowledgeTestGenerationService knowledgeTestGenerationService,
        ConsoleService consoleService)
        : base("knowledge-test", "Generate tests from the knowledge base")
    {
        _logger = logger;
        _knowledgeTestGenerationService = knowledgeTestGenerationService;
        _consoleService = consoleService;

        // Add options
        var projectOption = new Option<string>(
            ["--project", "-p"],
            "The target project for which to generate tests");
        projectOption.IsRequired = true;

        var maxTestsOption = new Option<int>(
            ["--max-tests", "-m"],
            () => 5,
            "Maximum number of tests to generate");

        // Add options to command
        AddOption(projectOption);
        AddOption(maxTestsOption);

        // Set the handler
        this.SetHandler(async (project, maxTests) =>
        {
            await HandleCommandAsync(project, maxTests);
        }, projectOption, maxTestsOption);
    }

    /// <summary>
    /// Handles the command execution
    /// </summary>
    private async Task HandleCommandAsync(string project, int maxTests)
    {
        try
        {
            int testsGenerated = await _knowledgeTestGenerationService.GenerateTestsAsync(project, maxTests);
            
            if (testsGenerated > 0)
            {
                _consoleService.WriteSuccess($"Successfully generated {testsGenerated} tests for project {project}");
                _consoleService.WriteInfo("Tests are saved in the 'generated_tests' directory");
            }
            else
            {
                _consoleService.WriteWarning($"No tests were generated for project {project}");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing knowledge test generation command");
            _consoleService.WriteError($"Error: {ex.Message}");
        }
    }
}
