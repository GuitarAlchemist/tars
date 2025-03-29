using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Text;
using System.Text.Json;

namespace TarsCli.Services;

public class TemplateService
{
    private readonly ILogger<TemplateService> _logger;
    private readonly IConfiguration _configuration;
    private readonly string _projectRoot;
    private readonly string _templatesDir;

    public TemplateService(
        ILogger<TemplateService> logger,
        IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;
        _projectRoot = _configuration["Tars:ProjectRoot"] ?? 
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), 
                "source", "repos", "tars");
        _templatesDir = Path.Combine(_projectRoot, "templates");
    }

    public async Task<bool> InitializeTemplatesDirectory()
    {
        try
        {
            if (!Directory.Exists(_templatesDir))
            {
                _logger.LogInformation($"Creating templates directory: {_templatesDir}");
                Directory.CreateDirectory(_templatesDir);
            }

            // Create default templates if they don't exist
            await CreateDefaultTemplates();
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing templates directory");
            return false;
        }
    }

    public async Task<bool> CreateDefaultTemplates()
    {
        try
        {
            // Create default session template
            var sessionTemplatePath = Path.Combine(_templatesDir, "default_session.json");
            if (!File.Exists(sessionTemplatePath))
            {
                var sessionTemplate = new
                {
                    session = new
                    {
                        name = "{{SESSION_NAME}}",
                        description = "A TARS session",
                        created = "{{TIMESTAMP}}",
                        version = "1.0.0"
                    },
                    agents = new
                    {
                        planner = new
                        {
                            model = "llama3",
                            temperature = 0.7,
                            description = "Plans the overall approach and breaks down tasks"
                        },
                        coder = new
                        {
                            model = "codellama:13b-code",
                            temperature = 0.2,
                            description = "Writes and refines code based on the plan"
                        },
                        critic = new
                        {
                            model = "llama3",
                            temperature = 0.5,
                            description = "Reviews and critiques code and plans"
                        },
                        executor = new
                        {
                            model = "llama3",
                            temperature = 0.3,
                            description = "Executes plans and reports results"
                        }
                    },
                    ollama = new
                    {
                        baseUrl = "http://localhost:11434",
                        defaultModel = "llama3"
                    }
                };

                var json = JsonSerializer.Serialize(sessionTemplate, new JsonSerializerOptions { WriteIndented = true });
                await File.WriteAllTextAsync(sessionTemplatePath, json);
                _logger.LogInformation($"Created default session template: {sessionTemplatePath}");
            }

            // Create default plan template
            var planTemplatePath = Path.Combine(_templatesDir, "default_plan.fsx");
            if (!File.Exists(planTemplatePath))
            {
                var planTemplate = @"// TARS Plan Template
// This is a sample F# script that defines a TARS workflow

#r ""nuget: FSharp.Data""

open System
open FSharp.Data

// Define the workflow
let workflow = async {
    printfn ""Starting TARS workflow...""
    
    // Example: Call the Planner agent
    let plannerPrompt = ""Create a plan for: {{TASK_DESCRIPTION}}""
    printfn ""Sending prompt to Planner: %s"" plannerPrompt
    
    // In a real implementation, this would call the actual agent
    let planResult = ""1. Analyze requirements\n2. Design solution\n3. Implement code\n4. Test solution""
    printfn ""Plan created: %s"" planResult
    
    // Example: Call the Coder agent
    let coderPrompt = sprintf ""Implement this plan:\n%s"" planResult
    printfn ""Sending prompt to Coder: %s"" coderPrompt
    
    // In a real implementation, this would call the actual agent
    let codeResult = ""// Implementation code\npublic class Solution {\n    public void Execute() {\n        Console.WriteLine(""Solution executed"");\n    }\n}""
    printfn ""Code generated: %s"" codeResult
    
    // Example: Call the Critic agent
    let criticPrompt = sprintf ""Review this code:\n%s"" codeResult
    printfn ""Sending prompt to Critic: %s"" criticPrompt
    
    // In a real implementation, this would call the actual agent
    let criticResult = ""The implementation looks good, but could use better error handling.""
    printfn ""Code reviewed: %s"" criticResult
    
    return ""Workflow completed successfully""
}

// Run the workflow
let result = Async.RunSynchronously(workflow)
printfn ""Result: %s"" result
";
                await File.WriteAllTextAsync(planTemplatePath, planTemplate);
                _logger.LogInformation($"Created default plan template: {planTemplatePath}");
            }

            // Create default workflow template
            var workflowTemplatePath = Path.Combine(_templatesDir, "default_workflow.json");
            if (!File.Exists(workflowTemplatePath))
            {
                var workflowTemplate = new
                {
                    name = "{{WORKFLOW_NAME}}",
                    description = "{{WORKFLOW_DESCRIPTION}}",
                    version = "1.0.0",
                    tasks = new[]
                    {
                        new
                        {
                            id = "1",
                            description = "Create a plan for: {{TASK_DESCRIPTION}}",
                            assignedTo = "planner",
                            dependencies = Array.Empty<string>()
                        },
                        new
                        {
                            id = "2",
                            description = "Implement the plan created in the previous step",
                            assignedTo = "coder",
                            dependencies = new[] { "1" }
                        },
                        new
                        {
                            id = "3",
                            description = "Review the implementation from the previous step",
                            assignedTo = "critic",
                            dependencies = new[] { "2" }
                        },
                        new
                        {
                            id = "4",
                            description = "Execute the implementation and report results",
                            assignedTo = "executor",
                            dependencies = new[] { "3" }
                        }
                    }
                };

                var json = JsonSerializer.Serialize(workflowTemplate, new JsonSerializerOptions { WriteIndented = true });
                await File.WriteAllTextAsync(workflowTemplatePath, json);
                _logger.LogInformation($"Created default workflow template: {workflowTemplatePath}");
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating default templates");
            return false;
        }
    }

    public async Task<string> GetTemplateContent(string templateName)
    {
        try
        {
            var templatePath = Path.Combine(_templatesDir, templateName);
            if (!File.Exists(templatePath))
            {
                _logger.LogError($"Template not found: {templateName}");
                return string.Empty;
            }

            return await File.ReadAllTextAsync(templatePath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error reading template: {templateName}");
            return string.Empty;
        }
    }

    public async Task<bool> CreateTemplate(string templateName, string content)
    {
        try
        {
            var templatePath = Path.Combine(_templatesDir, templateName);
            
            // Check if template already exists
            if (File.Exists(templatePath))
            {
                _logger.LogWarning($"Template already exists: {templateName}");
                return false;
            }

            await File.WriteAllTextAsync(templatePath, content);
            _logger.LogInformation($"Created template: {templatePath}");
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error creating template: {templateName}");
            return false;
        }
    }

    public List<string> ListTemplates()
    {
        try
        {
            if (!Directory.Exists(_templatesDir))
            {
                return new List<string>();
            }

            return Directory.GetFiles(_templatesDir)
                .Select(Path.GetFileName)
                .ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error listing templates");
            return new List<string>();
        }
    }

    public async Task<string> ApplyTemplateVariables(string templateContent, Dictionary<string, string> variables)
    {
        try
        {
            var result = templateContent;
            
            foreach (var variable in variables)
            {
                result = result.Replace($"{{{{{variable.Key}}}}}", variable.Value);
            }
            
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error applying template variables");
            return templateContent;
        }
    }
}
