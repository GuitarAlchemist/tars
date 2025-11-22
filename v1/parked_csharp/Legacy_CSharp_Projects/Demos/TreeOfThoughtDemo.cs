using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TarsEngine.FSharp.Adapters.TreeOfThought;
using TarsEngine.Services.Abstractions.TreeOfThought;

namespace TarsEngine.Demos
{
    /// <summary>
    /// Demo for the Tree-of-Thought implementation.
    /// </summary>
    public class TreeOfThoughtDemo
    {
        private readonly ITreeOfThoughtService _totService;
        private readonly ILogger<TreeOfThoughtDemo> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="TreeOfThoughtDemo"/> class.
        /// </summary>
        /// <param name="totService">The Tree-of-Thought service.</param>
        /// <param name="logger">The logger.</param>
        public TreeOfThoughtDemo(ITreeOfThoughtService totService, ILogger<TreeOfThoughtDemo> logger)
        {
            _totService = totService ?? throw new ArgumentNullException(nameof(totService));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Runs the demo.
        /// </summary>
        public async void Run()
        {
            _logger.LogInformation("Starting Tree-of-Thought demo...");

            try
            {
                // Create a thought tree for a problem
                var problem = "How to improve code quality?";
                var options = new TreeCreationOptions
                {
                    Approaches = new List<string>
                    {
                        "Static Analysis",
                        "Code Reviews",
                        "Automated Testing",
                        "Refactoring"
                    },
                    ApproachEvaluations = new Dictionary<string, EvaluationMetrics>
                    {
                        ["Static Analysis"] = new EvaluationMetrics(0.7, 0.8, 0.6, 0.7, 0.7),
                        ["Code Reviews"] = new EvaluationMetrics(0.9, 0.7, 0.8, 0.9, 0.825),
                        ["Automated Testing"] = new EvaluationMetrics(0.8, 0.9, 0.9, 0.7, 0.825),
                        ["Refactoring"] = new EvaluationMetrics(0.7, 0.6, 0.8, 0.9, 0.75)
                    }
                };

                _logger.LogInformation("Creating thought tree for problem: {Problem}", problem);
                var root = await _totService.CreateThoughtTreeAsync(problem, options);

                // Add child nodes to the best approach
                var bestNode = await _totService.SelectBestNodeAsync(root);
                _logger.LogInformation("Best approach: {Approach}", bestNode.Thought);

                if (bestNode.Thought == "Code Reviews")
                {
                    await _totService.AddChildAsync(bestNode, "Pair Programming");
                    await _totService.AddChildAsync(bestNode, "Pull Request Reviews");
                    await _totService.AddChildAsync(bestNode, "Code Walkthroughs");
                }
                else if (bestNode.Thought == "Automated Testing")
                {
                    await _totService.AddChildAsync(bestNode, "Unit Testing");
                    await _totService.AddChildAsync(bestNode, "Integration Testing");
                    await _totService.AddChildAsync(bestNode, "End-to-End Testing");
                }

                // Generate a report
                _logger.LogInformation("Generating report...");
                var report = await _totService.GenerateReportAsync(root, "Code Quality Improvement");

                // Save the report
                var reportPath = Path.Combine("Demos", "CodeQualityReport.md");
                await _totService.SaveReportAsync(root, "Code Quality Improvement", reportPath);
                _logger.LogInformation("Report saved to: {ReportPath}", reportPath);

                // Print the report
                Console.WriteLine("\nReport:");
                Console.WriteLine(report);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running Tree-of-Thought demo");
            }
        }

        /// <summary>
        /// Creates a service provider with the necessary services.
        /// </summary>
        /// <returns>The service provider.</returns>
        public static ServiceProvider CreateServiceProvider()
        {
            var services = new ServiceCollection();

            // Add logging
            services.AddLogging(builder =>
            {
                builder.AddConsole();
                builder.SetMinimumLevel(LogLevel.Information);
            });

            // Add the Tree-of-Thought service
            services.AddSingleton<ITreeOfThoughtService, FSharpTreeOfThoughtService>();

            // Add the demo
            services.AddSingleton<TreeOfThoughtDemo>();

            return services.BuildServiceProvider();
        }

        /// <summary>
        /// Entry point for the demo.
        /// </summary>
        public static void Main()
        {
            // Create the service provider
            using var serviceProvider = CreateServiceProvider();

            // Get the demo
            var demo = serviceProvider.GetRequiredService<TreeOfThoughtDemo>();

            // Run the demo
            demo.Run();

            // Wait for user input
            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }
    }
}
