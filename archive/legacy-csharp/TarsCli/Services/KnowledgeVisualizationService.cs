using System.Text;
using System.Text.Json;

namespace TarsCli.Services;

/// <summary>
/// Service for visualizing the knowledge base
/// </summary>
public class KnowledgeVisualizationService
{
    private readonly ILogger<KnowledgeVisualizationService> _logger;
    private readonly ConsoleService _consoleService;
    private readonly string _knowledgeBaseFile = "knowledge_base.json";
    private readonly string _visualizationDir = "visualizations";

    /// <summary>
    /// Initializes a new instance of the <see cref="KnowledgeVisualizationService"/> class.
    /// </summary>
    public KnowledgeVisualizationService(
        ILogger<KnowledgeVisualizationService> logger,
        ConsoleService consoleService)
    {
        _logger = logger;
        _consoleService = consoleService;
    }

    /// <summary>
    /// Generates a visualization of the knowledge base
    /// </summary>
    /// <returns>True if the visualization was generated successfully</returns>
    public async Task<bool> GenerateVisualizationAsync()
    {
        try
        {
            _consoleService.WriteHeader("TARS Knowledge Base Visualization");
            _consoleService.WriteInfo("Generating visualization...");

            // Ensure the visualization directory exists
            Directory.CreateDirectory(_visualizationDir);

            // Read the knowledge base
            if (!File.Exists(_knowledgeBaseFile))
            {
                _consoleService.WriteError($"Knowledge base file not found: {_knowledgeBaseFile}");
                return false;
            }

            var kbJson = await File.ReadAllTextAsync(_knowledgeBaseFile);
            var kb = JsonSerializer.Deserialize<JsonElement>(kbJson);

            // Generate HTML visualization
            var htmlPath = Path.Combine(_visualizationDir, "knowledge_base.html");
            await GenerateHtmlVisualizationAsync(kb, htmlPath);

            // Generate Markdown summary
            var mdPath = Path.Combine(_visualizationDir, "knowledge_base_summary.md");
            await GenerateMarkdownSummaryAsync(kb, mdPath);

            // Generate JSON statistics
            var statsPath = Path.Combine(_visualizationDir, "knowledge_base_stats.json");
            await GenerateJsonStatisticsAsync(kb, statsPath);

            _consoleService.WriteSuccess("Visualization generated successfully");
            _consoleService.WriteInfo($"HTML visualization: {Path.GetFullPath(htmlPath)}");
            _consoleService.WriteInfo($"Markdown summary: {Path.GetFullPath(mdPath)}");
            _consoleService.WriteInfo($"JSON statistics: {Path.GetFullPath(statsPath)}");

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating visualization");
            _consoleService.WriteError($"Error: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Generates an HTML visualization of the knowledge base
    /// </summary>
    private async Task<bool> GenerateHtmlVisualizationAsync(JsonElement kb, string outputPath)
    {
        try
        {
            var patterns = kb.GetProperty("patterns").EnumerateArray().ToList();
            var bestPractices = kb.GetProperty("best_practices").EnumerateArray().ToList();
            var codeExamples = kb.GetProperty("code_examples").EnumerateArray().ToList();
            var improvementStrategies = kb.GetProperty("improvement_strategies").EnumerateArray().ToList();
            var architectureInsights = kb.GetProperty("architecture_insights").EnumerateArray().ToList();

            var html = new StringBuilder();
            html.AppendLine("<!DOCTYPE html>");
            html.AppendLine("<html lang=\"en\">");
            html.AppendLine("<head>");
            html.AppendLine("  <meta charset=\"UTF-8\">");
            html.AppendLine("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">");
            html.AppendLine("  <title>TARS Knowledge Base Visualization</title>");
            html.AppendLine("  <style>");
            html.AppendLine("    body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }");
            html.AppendLine("    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }");
            html.AppendLine("    h2 { color: #2980b9; margin-top: 30px; }");
            html.AppendLine("    h3 { color: #3498db; }");
            html.AppendLine("    .container { max-width: 1200px; margin: 0 auto; }");
            html.AppendLine("    .card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: #f9f9f9; }");
            html.AppendLine("    .card h4 { margin-top: 0; color: #2c3e50; }");
            html.AppendLine("    .card p { margin-bottom: 10px; }");
            html.AppendLine("    .card .meta { font-size: 0.8em; color: #7f8c8d; }");
            html.AppendLine("    .code { background-color: #f0f0f0; padding: 10px; border-radius: 5px; font-family: monospace; white-space: pre-wrap; }");
            html.AppendLine("    .stats { display: flex; justify-content: space-around; flex-wrap: wrap; margin: 20px 0; }");
            html.AppendLine("    .stat-card { background-color: #ecf0f1; border-radius: 8px; padding: 15px; margin: 10px; min-width: 150px; text-align: center; }");
            html.AppendLine("    .stat-value { font-size: 2em; font-weight: bold; color: #2980b9; }");
            html.AppendLine("    .stat-label { font-size: 0.9em; color: #7f8c8d; }");
            html.AppendLine("    .tabs { display: flex; margin-bottom: 20px; }");
            html.AppendLine("    .tab { padding: 10px 20px; cursor: pointer; background-color: #f0f0f0; margin-right: 5px; border-radius: 5px 5px 0 0; }");
            html.AppendLine("    .tab.active { background-color: #3498db; color: white; }");
            html.AppendLine("    .tab-content { display: none; }");
            html.AppendLine("    .tab-content.active { display: block; }");
            html.AppendLine("  </style>");
            html.AppendLine("</head>");
            html.AppendLine("<body>");
            html.AppendLine("  <div class=\"container\">");
            html.AppendLine("    <h1>TARS Knowledge Base Visualization</h1>");
            
            // Statistics section
            html.AppendLine("    <div class=\"stats\">");
            html.AppendLine($"      <div class=\"stat-card\"><div class=\"stat-value\">{patterns.Count}</div><div class=\"stat-label\">Patterns</div></div>");
            html.AppendLine($"      <div class=\"stat-card\"><div class=\"stat-value\">{bestPractices.Count}</div><div class=\"stat-label\">Best Practices</div></div>");
            html.AppendLine($"      <div class=\"stat-card\"><div class=\"stat-value\">{codeExamples.Count}</div><div class=\"stat-label\">Code Examples</div></div>");
            html.AppendLine($"      <div class=\"stat-card\"><div class=\"stat-value\">{improvementStrategies.Count}</div><div class=\"stat-label\">Improvement Strategies</div></div>");
            html.AppendLine($"      <div class=\"stat-card\"><div class=\"stat-value\">{architectureInsights.Count}</div><div class=\"stat-label\">Architecture Insights</div></div>");
            html.AppendLine("    </div>");
            
            // Tabs
            html.AppendLine("    <div class=\"tabs\">");
            html.AppendLine("      <div class=\"tab active\" onclick=\"showTab('patterns')\">Patterns</div>");
            html.AppendLine("      <div class=\"tab\" onclick=\"showTab('best-practices')\">Best Practices</div>");
            html.AppendLine("      <div class=\"tab\" onclick=\"showTab('code-examples')\">Code Examples</div>");
            html.AppendLine("      <div class=\"tab\" onclick=\"showTab('improvement-strategies')\">Improvement Strategies</div>");
            html.AppendLine("      <div class=\"tab\" onclick=\"showTab('architecture-insights')\">Architecture Insights</div>");
            html.AppendLine("    </div>");
            
            // Patterns tab
            html.AppendLine("    <div id=\"patterns\" class=\"tab-content active\">");
            html.AppendLine("      <h2>Patterns</h2>");
            if (patterns.Count > 0)
            {
                foreach (var pattern in patterns)
                {
                    html.AppendLine("      <div class=\"card\">");
                    html.AppendLine($"        <h4>{GetPropertyStringValue(pattern, "name")}</h4>");
                    html.AppendLine($"        <p><strong>Description:</strong> {GetPropertyStringValue(pattern, "description")}</p>");
                    html.AppendLine($"        <p><strong>Context:</strong> {GetPropertyStringValue(pattern, "context")}</p>");
                    
                    var example = GetPropertyStringValue(pattern, "example");
                    if (!string.IsNullOrEmpty(example))
                    {
                        html.AppendLine($"        <p><strong>Example:</strong></p>");
                        html.AppendLine($"        <div class=\"code\">{example}</div>");
                    }
                    
                    var source = GetPropertyStringValue(pattern, "source");
                    if (!string.IsNullOrEmpty(source))
                    {
                        html.AppendLine($"        <p class=\"meta\">Source: {source}</p>");
                    }
                    
                    html.AppendLine("      </div>");
                }
            }
            else
            {
                html.AppendLine("      <p>No patterns found in the knowledge base.</p>");
            }
            html.AppendLine("    </div>");
            
            // Best Practices tab
            html.AppendLine("    <div id=\"best-practices\" class=\"tab-content\">");
            html.AppendLine("      <h2>Best Practices</h2>");
            if (bestPractices.Count > 0)
            {
                foreach (var practice in bestPractices)
                {
                    html.AppendLine("      <div class=\"card\">");
                    html.AppendLine($"        <h4>{GetPropertyStringValue(practice, "name")}</h4>");
                    html.AppendLine($"        <p><strong>Description:</strong> {GetPropertyStringValue(practice, "description")}</p>");
                    html.AppendLine($"        <p><strong>Context:</strong> {GetPropertyStringValue(practice, "context")}</p>");
                    
                    var source = GetPropertyStringValue(practice, "source");
                    if (!string.IsNullOrEmpty(source))
                    {
                        html.AppendLine($"        <p class=\"meta\">Source: {source}</p>");
                    }
                    
                    html.AppendLine("      </div>");
                }
            }
            else
            {
                html.AppendLine("      <p>No best practices found in the knowledge base.</p>");
            }
            html.AppendLine("    </div>");
            
            // Code Examples tab
            html.AppendLine("    <div id=\"code-examples\" class=\"tab-content\">");
            html.AppendLine("      <h2>Code Examples</h2>");
            if (codeExamples.Count > 0)
            {
                foreach (var example in codeExamples)
                {
                    html.AppendLine("      <div class=\"card\">");
                    html.AppendLine($"        <h4>{GetPropertyStringValue(example, "description")}</h4>");
                    html.AppendLine($"        <p><strong>Language:</strong> {GetPropertyStringValue(example, "language")}</p>");
                    
                    var code = GetPropertyStringValue(example, "code");
                    if (!string.IsNullOrEmpty(code))
                    {
                        html.AppendLine($"        <div class=\"code\">{code}</div>");
                    }
                    
                    var source = GetPropertyStringValue(example, "source");
                    if (!string.IsNullOrEmpty(source))
                    {
                        html.AppendLine($"        <p class=\"meta\">Source: {source}</p>");
                    }
                    
                    html.AppendLine("      </div>");
                }
            }
            else
            {
                html.AppendLine("      <p>No code examples found in the knowledge base.</p>");
            }
            html.AppendLine("    </div>");
            
            // Improvement Strategies tab
            html.AppendLine("    <div id=\"improvement-strategies\" class=\"tab-content\">");
            html.AppendLine("      <h2>Improvement Strategies</h2>");
            if (improvementStrategies.Count > 0)
            {
                foreach (var strategy in improvementStrategies)
                {
                    html.AppendLine("      <div class=\"card\">");
                    html.AppendLine($"        <h4>{GetPropertyStringValue(strategy, "name")}</h4>");
                    html.AppendLine($"        <p><strong>Description:</strong> {GetPropertyStringValue(strategy, "description")}</p>");
                    html.AppendLine($"        <p><strong>Applicability:</strong> {GetPropertyStringValue(strategy, "applicability")}</p>");
                    
                    var source = GetPropertyStringValue(strategy, "source");
                    if (!string.IsNullOrEmpty(source))
                    {
                        html.AppendLine($"        <p class=\"meta\">Source: {source}</p>");
                    }
                    
                    html.AppendLine("      </div>");
                }
            }
            else
            {
                html.AppendLine("      <p>No improvement strategies found in the knowledge base.</p>");
            }
            html.AppendLine("    </div>");
            
            // Architecture Insights tab
            html.AppendLine("    <div id=\"architecture-insights\" class=\"tab-content\">");
            html.AppendLine("      <h2>Architecture Insights</h2>");
            if (architectureInsights.Count > 0)
            {
                foreach (var insight in architectureInsights)
                {
                    html.AppendLine("      <div class=\"card\">");
                    html.AppendLine($"        <h4>{GetPropertyStringValue(insight, "name")}</h4>");
                    html.AppendLine($"        <p><strong>Description:</strong> {GetPropertyStringValue(insight, "description")}</p>");
                    html.AppendLine($"        <p><strong>Context:</strong> {GetPropertyStringValue(insight, "context")}</p>");
                    
                    var source = GetPropertyStringValue(insight, "source");
                    if (!string.IsNullOrEmpty(source))
                    {
                        html.AppendLine($"        <p class=\"meta\">Source: {source}</p>");
                    }
                    
                    html.AppendLine("      </div>");
                }
            }
            else
            {
                html.AppendLine("      <p>No architecture insights found in the knowledge base.</p>");
            }
            html.AppendLine("    </div>");
            
            // JavaScript for tab switching
            html.AppendLine("    <script>");
            html.AppendLine("      function showTab(tabId) {");
            html.AppendLine("        // Hide all tab contents");
            html.AppendLine("        document.querySelectorAll('.tab-content').forEach(content => {");
            html.AppendLine("          content.classList.remove('active');");
            html.AppendLine("        });");
            html.AppendLine("        // Show the selected tab content");
            html.AppendLine("        document.getElementById(tabId).classList.add('active');");
            html.AppendLine("        // Update tab styling");
            html.AppendLine("        document.querySelectorAll('.tab').forEach(tab => {");
            html.AppendLine("          tab.classList.remove('active');");
            html.AppendLine("        });");
            html.AppendLine("        document.querySelector(`.tab[onclick=\"showTab('${tabId}')\"]`).classList.add('active');");
            html.AppendLine("      }");
            html.AppendLine("    </script>");
            
            html.AppendLine("  </div>");
            html.AppendLine("</body>");
            html.AppendLine("</html>");

            await File.WriteAllTextAsync(outputPath, html.ToString());
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating HTML visualization");
            return false;
        }
    }

    /// <summary>
    /// Generates a Markdown summary of the knowledge base
    /// </summary>
    private async Task<bool> GenerateMarkdownSummaryAsync(JsonElement kb, string outputPath)
    {
        try
        {
            var patterns = kb.GetProperty("patterns").EnumerateArray().ToList();
            var bestPractices = kb.GetProperty("best_practices").EnumerateArray().ToList();
            var codeExamples = kb.GetProperty("code_examples").EnumerateArray().ToList();
            var improvementStrategies = kb.GetProperty("improvement_strategies").EnumerateArray().ToList();
            var architectureInsights = kb.GetProperty("architecture_insights").EnumerateArray().ToList();

            var md = new StringBuilder();
            md.AppendLine("# TARS Knowledge Base Summary");
            md.AppendLine();
            md.AppendLine("## Statistics");
            md.AppendLine();
            md.AppendLine("| Category | Count |");
            md.AppendLine("|----------|-------|");
            md.AppendLine($"| Patterns | {patterns.Count} |");
            md.AppendLine($"| Best Practices | {bestPractices.Count} |");
            md.AppendLine($"| Code Examples | {codeExamples.Count} |");
            md.AppendLine($"| Improvement Strategies | {improvementStrategies.Count} |");
            md.AppendLine($"| Architecture Insights | {architectureInsights.Count} |");
            md.AppendLine($"| **Total** | **{patterns.Count + bestPractices.Count + codeExamples.Count + improvementStrategies.Count + architectureInsights.Count}** |");
            md.AppendLine();

            // Patterns
            md.AppendLine("## Patterns");
            md.AppendLine();
            if (patterns.Count > 0)
            {
                foreach (var pattern in patterns.Take(10)) // Limit to 10 for brevity
                {
                    md.AppendLine($"### {GetPropertyStringValue(pattern, "name")}");
                    md.AppendLine();
                    md.AppendLine($"**Description:** {GetPropertyStringValue(pattern, "description")}");
                    md.AppendLine();
                    md.AppendLine($"**Context:** {GetPropertyStringValue(pattern, "context")}");
                    md.AppendLine();
                    
                    var example = GetPropertyStringValue(pattern, "example");
                    if (!string.IsNullOrEmpty(example))
                    {
                        md.AppendLine("**Example:**");
                        md.AppendLine();
                        md.AppendLine("```");
                        md.AppendLine(example);
                        md.AppendLine("```");
                        md.AppendLine();
                    }
                    
                    var source = GetPropertyStringValue(pattern, "source");
                    if (!string.IsNullOrEmpty(source))
                    {
                        md.AppendLine($"*Source: {source}*");
                        md.AppendLine();
                    }
                }
                
                if (patterns.Count > 10)
                {
                    md.AppendLine($"*... and {patterns.Count - 10} more patterns*");
                    md.AppendLine();
                }
            }
            else
            {
                md.AppendLine("No patterns found in the knowledge base.");
                md.AppendLine();
            }

            // Best Practices (abbreviated)
            md.AppendLine("## Best Practices");
            md.AppendLine();
            if (bestPractices.Count > 0)
            {
                md.AppendLine("| Name | Context |");
                md.AppendLine("|------|---------|");
                foreach (var practice in bestPractices)
                {
                    md.AppendLine($"| {GetPropertyStringValue(practice, "name")} | {GetPropertyStringValue(practice, "context")} |");
                }
                md.AppendLine();
            }
            else
            {
                md.AppendLine("No best practices found in the knowledge base.");
                md.AppendLine();
            }

            // Code Examples (abbreviated)
            md.AppendLine("## Code Examples");
            md.AppendLine();
            if (codeExamples.Count > 0)
            {
                md.AppendLine("| Description | Language |");
                md.AppendLine("|-------------|----------|");
                foreach (var example in codeExamples)
                {
                    md.AppendLine($"| {GetPropertyStringValue(example, "description")} | {GetPropertyStringValue(example, "language")} |");
                }
                md.AppendLine();
            }
            else
            {
                md.AppendLine("No code examples found in the knowledge base.");
                md.AppendLine();
            }

            // Improvement Strategies (abbreviated)
            md.AppendLine("## Improvement Strategies");
            md.AppendLine();
            if (improvementStrategies.Count > 0)
            {
                md.AppendLine("| Name | Applicability |");
                md.AppendLine("|------|--------------|");
                foreach (var strategy in improvementStrategies)
                {
                    md.AppendLine($"| {GetPropertyStringValue(strategy, "name")} | {GetPropertyStringValue(strategy, "applicability")} |");
                }
                md.AppendLine();
            }
            else
            {
                md.AppendLine("No improvement strategies found in the knowledge base.");
                md.AppendLine();
            }

            // Architecture Insights (abbreviated)
            md.AppendLine("## Architecture Insights");
            md.AppendLine();
            if (architectureInsights.Count > 0)
            {
                md.AppendLine("| Name | Context |");
                md.AppendLine("|------|---------|");
                foreach (var insight in architectureInsights)
                {
                    md.AppendLine($"| {GetPropertyStringValue(insight, "name")} | {GetPropertyStringValue(insight, "context")} |");
                }
                md.AppendLine();
            }
            else
            {
                md.AppendLine("No architecture insights found in the knowledge base.");
                md.AppendLine();
            }

            await File.WriteAllTextAsync(outputPath, md.ToString());
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating Markdown summary");
            return false;
        }
    }

    /// <summary>
    /// Generates JSON statistics for the knowledge base
    /// </summary>
    private async Task<bool> GenerateJsonStatisticsAsync(JsonElement kb, string outputPath)
    {
        try
        {
            var patterns = kb.GetProperty("patterns").EnumerateArray().ToList();
            var bestPractices = kb.GetProperty("best_practices").EnumerateArray().ToList();
            var codeExamples = kb.GetProperty("code_examples").EnumerateArray().ToList();
            var improvementStrategies = kb.GetProperty("improvement_strategies").EnumerateArray().ToList();
            var architectureInsights = kb.GetProperty("architecture_insights").EnumerateArray().ToList();

            // Count contexts
            var contexts = new Dictionary<string, int>();
            foreach (var pattern in patterns)
            {
                var context = GetPropertyStringValue(pattern, "context");
                if (!string.IsNullOrEmpty(context))
                {
                    if (contexts.ContainsKey(context))
                        contexts[context]++;
                    else
                        contexts[context] = 1;
                }
            }

            // Count languages
            var languages = new Dictionary<string, int>();
            foreach (var example in codeExamples)
            {
                var language = GetPropertyStringValue(example, "language");
                if (!string.IsNullOrEmpty(language))
                {
                    if (languages.ContainsKey(language))
                        languages[language]++;
                    else
                        languages[language] = 1;
                }
            }

            // Count sources
            var sources = new Dictionary<string, int>();
            CountSourcesFromArray(patterns, sources);
            CountSourcesFromArray(bestPractices, sources);
            CountSourcesFromArray(codeExamples, sources);
            CountSourcesFromArray(improvementStrategies, sources);
            CountSourcesFromArray(architectureInsights, sources);

            // Create statistics object
            var stats = new
            {
                total_entries = patterns.Count + bestPractices.Count + codeExamples.Count + improvementStrategies.Count + architectureInsights.Count,
                categories = new
                {
                    patterns = patterns.Count,
                    best_practices = bestPractices.Count,
                    code_examples = codeExamples.Count,
                    improvement_strategies = improvementStrategies.Count,
                    architecture_insights = architectureInsights.Count
                },
                contexts = contexts.OrderByDescending(c => c.Value).ToDictionary(c => c.Key, c => c.Value),
                languages = languages.OrderByDescending(l => l.Value).ToDictionary(l => l.Key, l => l.Value),
                sources = sources.OrderByDescending(s => s.Value).ToDictionary(s => s.Key, s => s.Value),
                timestamp = DateTime.UtcNow
            };

            var json = JsonSerializer.Serialize(stats, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(outputPath, json);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating JSON statistics");
            return false;
        }
    }

    /// <summary>
    /// Gets a string value from a JSON property, or an empty string if the property doesn't exist
    /// </summary>
    private string GetPropertyStringValue(JsonElement element, string propertyName)
    {
        if (element.TryGetProperty(propertyName, out var property))
        {
            return property.ValueKind == JsonValueKind.String ? property.GetString() ?? "" : "";
        }
        return "";
    }

    /// <summary>
    /// Counts sources from an array of JSON elements
    /// </summary>
    private void CountSourcesFromArray(List<JsonElement> elements, Dictionary<string, int> sources)
    {
        foreach (var element in elements)
        {
            var source = GetPropertyStringValue(element, "source");
            if (!string.IsNullOrEmpty(source))
            {
                if (sources.ContainsKey(source))
                    sources[source]++;
                else
                    sources[source] = 1;
            }
        }
    }
}
