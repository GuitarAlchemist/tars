using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TarsCli.Models;

namespace TarsCli.Services;

/// <summary>
/// Service for integrating knowledge application with other TARS systems
/// </summary>
public class KnowledgeIntegrationService
{
    private readonly ILogger<KnowledgeIntegrationService> _logger;
    private readonly KnowledgeApplicationService _knowledgeApplicationService;
    private readonly DocumentationService _documentationService;
    private readonly DslService _dslService;
    private readonly OllamaService _ollamaService;

    public KnowledgeIntegrationService(
        ILogger<KnowledgeIntegrationService> logger,
        KnowledgeApplicationService knowledgeApplicationService,
        DocumentationService documentationService,
        DslService dslService,
        OllamaService ollamaService)
    {
        _logger = logger;
        _knowledgeApplicationService = knowledgeApplicationService;
        _documentationService = documentationService;
        _dslService = dslService;
        _ollamaService = ollamaService;
    }

    /// <summary>
    /// Generate a metascript based on extracted knowledge
    /// </summary>
    /// <param name="targetDirectory">The directory to target with the metascript</param>
    /// <param name="filePattern">The file pattern to match</param>
    /// <param name="model">The model to use</param>
    /// <returns>Path to the generated metascript</returns>
    public async Task<string> GenerateKnowledgeMetascriptAsync(string targetDirectory, string filePattern = "*.cs", string model = "llama3")
    {
        _logger.LogInformation($"Generating knowledge metascript for {targetDirectory} with pattern {filePattern}");

        try
        {
            // Get all knowledge from the knowledge base
            var allKnowledge = await _knowledgeApplicationService.GetAllKnowledgeAsync();
            if (allKnowledge.Count == 0)
            {
                _logger.LogWarning("No knowledge available in the knowledge base");
                return null;
            }

            // Create a summary of the knowledge
            var knowledgeSummary = new System.Text.StringBuilder();
            knowledgeSummary.AppendLine("# Knowledge Base Summary");
            knowledgeSummary.AppendLine();

            // Add key concepts
            knowledgeSummary.AppendLine("## Key Concepts");
            knowledgeSummary.AppendLine();
            foreach (var knowledge in allKnowledge.Take(5))
            {
                if (knowledge.KeyConcepts != null)
                {
                    foreach (var concept in knowledge.KeyConcepts.Take(3))
                    {
                        knowledgeSummary.AppendLine($"- **{concept.Name}**: {concept.Definition}");
                    }
                }
            }
            knowledgeSummary.AppendLine();

            // Add insights
            knowledgeSummary.AppendLine("## Key Insights");
            knowledgeSummary.AppendLine();
            foreach (var knowledge in allKnowledge.Take(5))
            {
                if (knowledge.Insights != null)
                {
                    foreach (var insight in knowledge.Insights.Take(3))
                    {
                        knowledgeSummary.AppendLine($"- {insight.Description}");
                    }
                }
            }
            knowledgeSummary.AppendLine();

            // Add design decisions
            knowledgeSummary.AppendLine("## Design Decisions");
            knowledgeSummary.AppendLine();
            foreach (var knowledge in allKnowledge.Take(5))
            {
                if (knowledge.DesignDecisions != null)
                {
                    foreach (var decision in knowledge.DesignDecisions.Take(3))
                    {
                        knowledgeSummary.AppendLine($"- **{decision.Decision}**: {decision.Rationale}");
                    }
                }
            }

            // Generate the metascript
            var metascript = $@"DESCRIBE {{
    name: ""Knowledge-Based Code Improvement""
    version: ""1.0""
    author: ""TARS""
    description: ""A metascript for improving code based on knowledge extracted from documentation""
    tags: [""knowledge"", ""code-improvement"", ""documentation""]
}}

CONFIG {{
    model: ""{model}""
    temperature: 0.3
    max_tokens: 4000
}}

// Define paths to important directories
VARIABLE target_dir {{
    value: ""{targetDirectory.Replace("\\", "\\\\")}""
}}

VARIABLE file_pattern {{
    value: ""{filePattern}""
}}

// Define the knowledge base summary
VARIABLE knowledge_summary {{
    value: `{knowledgeSummary.ToString().Replace("`", "\\`")}`
}}

// Get all files matching the pattern
FUNCTION get_files {{
    ACTION {{
        type: ""get_files""
        directory: ""${{target_dir}}""
        pattern: ""${{file_pattern}}""
        output_variable: ""files""
    }}

    RETURN {{
        value: ""${{files}}""
    }}
}}

// Improve a file based on knowledge
FUNCTION improve_file {{
    parameters: [""file_path""]

    ACTION {{
        type: ""log""
        message: ""Improving file: ${{file_path}}""
    }}

    // Read the file content
    ACTION {{
        type: ""file_read""
        path: ""${{file_path}}""
        output_variable: ""file_content""
    }}

    // Create a prompt for improving the file
    VARIABLE prompt_text {{
        value: ""You are an expert at improving code by applying knowledge from a knowledge base.

I'll provide you with:
1. The content of a file from the TARS project
2. A summary of knowledge extracted from TARS documentation

Your task is to improve the file by applying relevant knowledge from the knowledge base. This could involve:
- Adding comments to explain concepts or design decisions
- Improving variable or function names based on standard terminology
- Adding documentation that references key concepts
- Restructuring code to better align with design principles
- Adding references to related concepts or components

Here's the file content:

```
${{file_content}}
```

Here's the knowledge base summary:

${{knowledge_summary}}

Please provide your improved version of the file. If you make changes, explain each change and how it applies knowledge from the knowledge base.

If the file doesn't need improvement or the knowledge isn't relevant, just say so.""
    }}

    // Get improvement suggestions from the LLM
    PROMPT {{
        text: ""${{prompt_text}}""
        model: ""{model}""
        temperature: 0.3
        max_tokens: 4000
        output_variable: ""response""
    }}

    // Extract the improved content
    VARIABLE improved_content {{
        value: ""${{response.match(/```(?:[\\w]*)\s*([\\s\\S]*?)```/)?.[1]?.trim() || file_content}}""
    }}

    // Check if the content was actually improved
    IF {{
        condition: ""${{improved_content !== file_content}}""
        then: {{
            // Create a backup of the original file
            VARIABLE backup_path {{
                value: ""${{file_path}}.bak""
            }}

            ACTION {{
                type: ""file_write""
                path: ""${{backup_path}}""
                content: ""${{file_content}}""
            }}

            // Write the improved content back to the file
            ACTION {{
                type: ""file_write""
                path: ""${{file_path}}""
                content: ""${{improved_content}}""
            }}

            ACTION {{
                type: ""log""
                message: ""File improved: ${{file_path}}""
            }}

            RETURN {{
                value: true
            }}
        }}
        else: {{
            ACTION {{
                type: ""log""
                message: ""No improvements needed for: ${{file_path}}""
            }}

            RETURN {{
                value: false
            }}
        }}
    }}
}}

// Main workflow
TARS {{
    // Initialize the workflow
    ACTION {{
        type: ""log""
        message: ""Starting Knowledge-Based Code Improvement""
    }}

    // Get all files matching the pattern
    ACTION {{
        type: ""call_function""
        function: ""get_files""
        output_variable: ""files""
    }}

    ACTION {{
        type: ""log""
        message: ""Found ${{files.length}} files to process""
    }}

    // Process each file
    VARIABLE improved_files {{
        value: []
    }}

    FOREACH {{
        items: ""${{files}}""
        item_variable: ""file""
        
        // Improve the file
        ACTION {{
            type: ""call_function""
            function: ""improve_file""
            parameters: {{
                file_path: ""${{file}}""
            }}
            output_variable: ""improved""
        }}

        // Track improved files
        IF {{
            condition: ""${{improved}}""
            then: {{
                VARIABLE improved_files {{
                    value: ""${{[...improved_files, file]}}""
                }}
            }}
        }}
    }}

    // Generate a summary report
    VARIABLE summary {{
        value: ""# Knowledge-Based Code Improvement Report\n\n"" +
               ""## Summary\n\n"" +
               ""- **Files Processed:** ${{files.length}}\n"" +
               ""- **Files Improved:** ${{improved_files.length}}\n\n"" +
               ""## Improved Files\n\n"" +
               ""${{improved_files.map(file => `- ${{file}}`).join('\n')}}\n\n"" +
               ""## Knowledge Applied\n\n"" +
               ""${{knowledge_summary}}\n\n"" +
               ""## Next Steps\n\n"" +
               ""1. Review the improved files to ensure the changes are appropriate\n"" +
               ""2. Extract additional knowledge from documentation to further improve the codebase\n"" +
               ""3. Run the improvement process again with new knowledge\n""
    }}

    ACTION {{
        type: ""file_write""
        path: ""knowledge_improvement_report.md""
        content: ""${{summary}}""
    }}

    ACTION {{
        type: ""log""
        message: ""Knowledge-Based Code Improvement completed. Report generated: knowledge_improvement_report.md""
    }}
}}";

            // Save the metascript
            var metascriptPath = Path.Combine(Path.GetTempPath(), "knowledge_improvement.tars");
            await File.WriteAllTextAsync(metascriptPath, metascript);

            _logger.LogInformation($"Knowledge metascript generated: {Path.GetFullPath(metascriptPath)}");
            return metascriptPath;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating knowledge metascript");
            return null;
        }
    }

    /// <summary>
    /// Run a knowledge-based improvement cycle
    /// </summary>
    /// <param name="explorationDirectory">The directory containing exploration files to extract knowledge from</param>
    /// <param name="targetDirectory">The directory to target with improvements</param>
    /// <param name="filePattern">The file pattern to match</param>
    /// <param name="model">The model to use</param>
    /// <returns>Path to the generated report</returns>
    public async Task<string> RunKnowledgeImprovementCycleAsync(string explorationDirectory, string targetDirectory, string filePattern = "*.cs", string model = "llama3")
    {
        _logger.LogInformation($"Running knowledge improvement cycle for {explorationDirectory} -> {targetDirectory}");

        try
        {
            // Step 1: Extract knowledge from exploration files
            _logger.LogInformation("Step 1: Extracting knowledge from exploration files");
            var files = Directory.GetFiles(explorationDirectory, "*.md", SearchOption.AllDirectories);
            _logger.LogInformation($"Found {files.Length} exploration files");

            foreach (var file in files)
            {
                _logger.LogInformation($"Extracting knowledge from: {Path.GetFileName(file)}");
                await _knowledgeApplicationService.ExtractKnowledgeAsync(file, model);
            }

            // Step 2: Generate a knowledge report
            _logger.LogInformation("Step 2: Generating knowledge report");
            var reportPath = await _knowledgeApplicationService.GenerateKnowledgeReportAsync();

            // Step 3: Generate a knowledge metascript
            _logger.LogInformation("Step 3: Generating knowledge metascript");
            var metascriptPath = await GenerateKnowledgeMetascriptAsync(targetDirectory, filePattern, model);

            // Step 4: Run the knowledge metascript
            _logger.LogInformation("Step 4: Running knowledge metascript");
            await _dslService.RunDslFileAsync(metascriptPath, true);

            _logger.LogInformation("Knowledge improvement cycle completed");
            return reportPath;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running knowledge improvement cycle");
            return null;
        }
    }

    /// <summary>
    /// Generate a retroaction report based on knowledge extraction and application
    /// </summary>
    /// <param name="explorationDirectory">The directory containing exploration files</param>
    /// <param name="targetDirectory">The directory containing target files</param>
    /// <param name="model">The model to use</param>
    /// <returns>Path to the generated report</returns>
    public async Task<string> GenerateRetroactionReportAsync(string explorationDirectory, string targetDirectory, string model = "llama3")
    {
        _logger.LogInformation($"Generating retroaction report for {explorationDirectory} -> {targetDirectory}");

        try
        {
            // Get all knowledge from the knowledge base
            var allKnowledge = await _knowledgeApplicationService.GetAllKnowledgeAsync();
            if (allKnowledge.Count == 0)
            {
                _logger.LogWarning("No knowledge available in the knowledge base");
                return null;
            }

            // Get all exploration files
            var explorationFiles = Directory.GetFiles(explorationDirectory, "*.md", SearchOption.AllDirectories);
            
            // Get all target files
            var targetFiles = Directory.GetFiles(targetDirectory, "*.cs", SearchOption.AllDirectories);

            // Create a prompt for generating the retroaction report
            var prompt = $@"You are an expert at analyzing software development processes and knowledge flow.

I'll provide you with information about:
1. A set of exploration documents that contain knowledge and insights
2. A set of code files that have been improved based on that knowledge
3. A summary of the extracted knowledge

Your task is to generate a retroaction report that analyzes:
- How effectively knowledge from explorations was extracted and applied
- What patterns or insights emerged from this process
- What improvements could be made to the knowledge extraction and application process
- What next steps should be taken to further improve the codebase

Exploration files ({explorationFiles.Length}):
{string.Join("\n", explorationFiles.Select(f => $"- {Path.GetFileName(f)}"))}

Target files ({targetFiles.Length}):
{string.Join("\n", targetFiles.Select(f => $"- {Path.GetFileName(f)}"))}

Knowledge summary:
{string.Join("\n", allKnowledge.Select(k => $"- {k.Title}: {k.Summary}"))}

Please provide a comprehensive retroaction report in markdown format.";

            // Generate the retroaction report
            var response = await _ollamaService.GenerateAsync(prompt, model);

            // Save the report
            var reportPath = Path.Combine(Path.GetTempPath(), $"retroaction_report_{DateTime.Now:yyyyMMdd_HHmmss}.md");
            await File.WriteAllTextAsync(reportPath, response);

            _logger.LogInformation($"Retroaction report generated: {Path.GetFullPath(reportPath)}");
            return reportPath;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating retroaction report");
            return null;
        }
    }
}
