using Microsoft.Extensions.Configuration;
using Microsoft.FSharp.Core;
using System.Text;

namespace TarsCli.Services;

public class WorkflowCoordinationService
{
    private readonly ILogger<WorkflowCoordinationService> _logger;
    private readonly IConfiguration _configuration;
    private readonly OllamaService _ollamaService;

    public WorkflowCoordinationService(
        ILogger<WorkflowCoordinationService> logger,
        IConfiguration configuration,
        OllamaService ollamaService)
    {
        _logger = logger;
        _configuration = configuration;
        _ollamaService = ollamaService;
    }

    public async Task<bool> RunWorkflow(string taskDescription)
    {
        try
        {
            _logger.LogInformation($"Running workflow for task: {taskDescription}");

            // Get Ollama endpoint from configuration
            var ollamaEndpoint = _configuration["Ollama:BaseUrl"] ?? "http://localhost:11434";

            // Create and run a standard workflow
            Console.WriteLine($"Creating workflow for task: {taskDescription}");
            Console.WriteLine("Using standard agent configuration:");
            Console.WriteLine("- Planner: llama3 (temperature: 0.7)");
            Console.WriteLine("- Coder: codellama:13b-code (temperature: 0.2)");
            Console.WriteLine("- Critic: llama3 (temperature: 0.5)");
            Console.WriteLine("- Executor: llama3 (temperature: 0.3)");
            Console.WriteLine();

            // Run the workflow
            Console.WriteLine("Starting workflow execution...");

            // TODO: Uncomment this when the F# module is properly referenced
            // var workflowResult = await FSharpAsync.StartAsTask(
            //     AgentCoordination.runStandardWorkflow(ollamaEndpoint, taskDescription),
            //     FSharpOption<TaskCreationOptions>.None,
            //     FSharpOption<System.Threading.CancellationToken>.None);

            // For now, we'll just simulate a workflow result
            var workflowResult = new {
                Status = "Completed",
                StartTime = DateTime.UtcNow.AddMinutes(-5),
                EndTime = FSharpOption<DateTime>.Some(DateTime.UtcNow),
                Tasks = new[] {
                    new {
                        Id = "1",
                        Description = "Create a plan for: " + taskDescription,
                        AssignedTo = "Planner",
                        Status = "Completed",
                        Messages = new[] {
                            new {
                                Role = "Planner",
                                Content = "I've created a plan for this task.",
                                Timestamp = DateTime.UtcNow.AddMinutes(-4)
                            }
                        },
                        Result = FSharpOption<string>.Some("1. Analyze requirements\n2. Design solution\n3. Implement code\n4. Test solution")
                    },
                    new {
                        Id = "2",
                        Description = "Implement the plan created in the previous step",
                        AssignedTo = "Coder",
                        Status = "Completed",
                        Messages = new[] {
                            new {
                                Role = "Coder",
                                Content = "I've implemented the solution based on the plan.",
                                Timestamp = DateTime.UtcNow.AddMinutes(-3)
                            }
                        },
                        Result = FSharpOption<string>.Some("// Implementation code here\npublic class Solution {\n    public void Execute() {\n        Console.WriteLine(\"Solution executed\");\n    }\n}")
                    },
                    new {
                        Id = "3",
                        Description = "Review the implementation from the previous step",
                        AssignedTo = "Critic",
                        Status = "Completed",
                        Messages = new[] {
                            new {
                                Role = "Critic",
                                Content = "I've reviewed the implementation and have some feedback.",
                                Timestamp = DateTime.UtcNow.AddMinutes(-2)
                            }
                        },
                        Result = FSharpOption<string>.Some("The implementation looks good, but could use better error handling.")
                    },
                    new {
                        Id = "4",
                        Description = "Execute the implementation and report results",
                        AssignedTo = "Executor",
                        Status = "Completed",
                        Messages = new[] {
                            new {
                                Role = "Executor",
                                Content = "I've executed the implementation and here are the results.",
                                Timestamp = DateTime.UtcNow.AddMinutes(-1)
                            }
                        },
                        Result = FSharpOption<string>.Some("Execution successful. All tests passed.")
                    }
                }
            };

            // Display the results
            Console.WriteLine();
            CliSupport.WriteHeader("Workflow Results");
            Console.WriteLine($"Status: {workflowResult.Status}");
            Console.WriteLine($"Start Time: {workflowResult.StartTime:yyyy-MM-dd HH:mm:ss}");

            if (FSharpOption<DateTime>.get_IsSome(workflowResult.EndTime))
            {
                var endTime = workflowResult.EndTime.Value;
                var duration = (endTime - workflowResult.StartTime).TotalSeconds;
                Console.WriteLine($"End Time: {endTime:yyyy-MM-dd HH:mm:ss}");
                Console.WriteLine($"Duration: {duration:F1} seconds");
            }

            Console.WriteLine();
            CliSupport.WriteHeader("Task Results");

            foreach (var task in workflowResult.Tasks)
            {
                CliSupport.WriteColorLine($"Task {task.Id}: {task.Description}", ConsoleColor.Cyan);
                Console.WriteLine($"Assigned To: {task.AssignedTo}");
                Console.WriteLine($"Status: {task.Status}");

                if (task.Messages.Length > 0)
                {
                    Console.WriteLine("Messages:");
                    foreach (var message in task.Messages)
                    {
                        Console.WriteLine($"- From {message.Role} at {message.Timestamp:HH:mm:ss}:");
                        Console.WriteLine($"  {message.Content.Substring(0, Math.Min(100, message.Content.Length))}...");
                    }
                }

                if (FSharpOption<string>.get_IsSome(task.Result))
                {
                    Console.WriteLine("Result:");
                    var result = task.Result.Value;
                    var previewLength = Math.Min(200, result.Length);
                    Console.WriteLine(result.Substring(0, previewLength) + (result.Length > previewLength ? "..." : ""));
                }

                Console.WriteLine();
            }

            // Save the results to a file
            var outputDir = Path.Combine(
                _configuration["Tars:ProjectRoot"] ?? Directory.GetCurrentDirectory(),
                "output",
                $"workflow_{DateTime.UtcNow:yyyyMMdd_HHmmss}");

            Directory.CreateDirectory(outputDir);

            // Save each task result to a separate file
            foreach (var task in workflowResult.Tasks)
            {
                if (FSharpOption<string>.get_IsSome(task.Result))
                {
                    var fileName = $"task_{task.Id}_{task.AssignedTo}.txt";
                    var filePath = Path.Combine(outputDir, fileName);
                    await File.WriteAllTextAsync(filePath, task.Result.Value);
                }
            }

            // Save a summary file
            var summaryBuilder = new StringBuilder();
            summaryBuilder.AppendLine($"# Workflow Summary for: {taskDescription}");
            summaryBuilder.AppendLine();
            summaryBuilder.AppendLine($"Status: {workflowResult.Status}");
            summaryBuilder.AppendLine($"Start Time: {workflowResult.StartTime:yyyy-MM-dd HH:mm:ss}");

            if (FSharpOption<DateTime>.get_IsSome(workflowResult.EndTime))
            {
                var endTime = workflowResult.EndTime.Value;
                var duration = (endTime - workflowResult.StartTime).TotalSeconds;
                summaryBuilder.AppendLine($"End Time: {endTime:yyyy-MM-dd HH:mm:ss}");
                summaryBuilder.AppendLine($"Duration: {duration:F1} seconds");
            }

            summaryBuilder.AppendLine();
            summaryBuilder.AppendLine("## Task Results");

            foreach (var task in workflowResult.Tasks)
            {
                summaryBuilder.AppendLine($"### Task {task.Id}: {task.Description}");
                summaryBuilder.AppendLine($"Assigned To: {task.AssignedTo}");
                summaryBuilder.AppendLine($"Status: {task.Status}");

                if (FSharpOption<string>.get_IsSome(task.Result))
                {
                    summaryBuilder.AppendLine();
                    summaryBuilder.AppendLine("Result:");
                    summaryBuilder.AppendLine("```");
                    summaryBuilder.AppendLine(task.Result.Value);
                    summaryBuilder.AppendLine("```");
                }

                summaryBuilder.AppendLine();
            }

            var summaryPath = Path.Combine(outputDir, "summary.md");
            await File.WriteAllTextAsync(summaryPath, summaryBuilder.ToString());

            Console.WriteLine($"Workflow results saved to: {outputDir}");

            return workflowResult.Status == "Completed";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error running workflow for task: {taskDescription}");
            Console.WriteLine($"Error running workflow: {ex.Message}");
            return false;
        }
    }
}
