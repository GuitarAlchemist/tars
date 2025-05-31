using Microsoft.Extensions.DependencyInjection;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for demonstrating the self-coding capabilities
/// </summary>
public class SelfCodingDemoCommand : Command
{
    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// Initializes a new instance of the SelfCodingDemoCommand class
    /// </summary>
    /// <param name="serviceProvider">Service provider</param>
    public SelfCodingDemoCommand(IServiceProvider serviceProvider) : base("self-code-demo", "Demonstrate self-coding capabilities")
    {
        _serviceProvider = serviceProvider;

        this.SetHandler(async () =>
        {
            var logger = _serviceProvider.GetRequiredService<ILogger<SelfCodingDemoCommand>>();
            var replicaManager = _serviceProvider.GetRequiredService<TarsReplicaManager>();
            var dockerService = _serviceProvider.GetRequiredService<DockerService>();
            var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

            try
            {
                // Display demo header
                consoleService.WriteHeader("TARS SELF-CODING DEMONSTRATION");
                consoleService.WriteInfo("This demo showcases the self-coding capabilities of TARS.");
                consoleService.WriteInfo("TARS can analyze, improve, and test its own code using a swarm of specialized replicas.");
                Console.WriteLine();

                // Check if Docker is running
                consoleService.WriteSubHeader("Checking Docker");
                var isDockerRunning = await dockerService.IsDockerRunning();
                if (!isDockerRunning)
                {
                    consoleService.WriteError("Docker is not running. Please start Docker first.");
                    // Failure
                    return;
                }
                consoleService.WriteSuccess("Docker is running");
                Console.WriteLine();

                // Create Docker network
                consoleService.WriteSubHeader("Creating Docker Network");
                consoleService.WriteInfo("Creating Docker network for TARS replicas...");
                await Task.Delay(1000);
                consoleService.WriteSuccess("Docker network created successfully (simulated)");
                Console.WriteLine();

                // Create replicas
                consoleService.WriteSubHeader("Creating TARS Replicas");
                consoleService.WriteInfo("Creating analyzer replica...");
                var analyzerReplica = await replicaManager.CreateReplicaAsync(
                    "CodeAnalyzer",
                    "analyzer",
                    ["analyze_code", "detect_issues", "suggest_improvements"]);
                consoleService.WriteSuccess($"Analyzer replica created with ID: {analyzerReplica.Id}");

                consoleService.WriteInfo("Creating generator replica...");
                var generatorReplica = await replicaManager.CreateReplicaAsync(
                    "CodeGenerator",
                    "generator",
                    ["generate_code", "refactor_code", "optimize_code"]);
                consoleService.WriteSuccess($"Generator replica created with ID: {generatorReplica.Id}");

                consoleService.WriteInfo("Creating tester replica...");
                var testerReplica = await replicaManager.CreateReplicaAsync(
                    "TestGenerator",
                    "tester",
                    ["generate_tests", "run_tests", "analyze_test_results"]);
                consoleService.WriteSuccess($"Tester replica created with ID: {testerReplica.Id}");

                consoleService.WriteInfo("Creating coordinator replica...");
                var coordinatorReplica = await replicaManager.CreateReplicaAsync(
                    "Coordinator",
                    "coordinator",
                    ["coordinate_workflow", "prioritize_tasks", "track_progress"]);
                consoleService.WriteSuccess($"Coordinator replica created with ID: {coordinatorReplica.Id}");
                Console.WriteLine();

                // Wait for replicas to start
                consoleService.WriteSubHeader("Waiting for Replicas to Start");
                consoleService.WriteInfo("Waiting for replicas to start...");
                await Task.Delay(5000);
                await replicaManager.UpdateReplicaStatusesAsync();
                var replicas = replicaManager.GetAllReplicas();
                foreach (var replica in replicas)
                {
                    consoleService.WriteInfo($"Replica {replica.Name} (ID: {replica.Id}) status: {replica.Status}");
                }
                Console.WriteLine();

                // Simulate self-coding workflow
                consoleService.WriteSubHeader("Simulating Self-Coding Workflow");

                // Step 1: File Selection
                consoleService.WriteInfo("Step 1: File Selection");
                consoleService.WriteInfo("Coordinator selects a file to improve: TarsCli/Program.cs");
                await Task.Delay(1000);
                consoleService.WriteSuccess("File selected");
                Console.WriteLine();

                // Step 2: Code Analysis
                consoleService.WriteInfo("Step 2: Code Analysis");
                consoleService.WriteInfo("Analyzer examines the file for improvement opportunities");
                await Task.Delay(2000);
                consoleService.WriteSuccess("Analysis completed");
                consoleService.WriteInfo("Found 3 potential improvements (simulated):");
                consoleService.WriteInfo("  1. Missing XML documentation on public methods");
                consoleService.WriteInfo("  2. Unused variable in ConfigureServices method");
                consoleService.WriteInfo("  3. Opportunity to use pattern matching in Main method");
                Console.WriteLine();

                // Step 3: Code Generation
                consoleService.WriteInfo("Step 3: Code Generation");
                consoleService.WriteInfo("Generator creates improved code based on analysis");
                await Task.Delay(2000);
                consoleService.WriteSuccess("Code generation completed");
                consoleService.WriteInfo("Generated improvements for all 3 issues (simulated)");
                Console.WriteLine();

                // Step 4: Testing
                consoleService.WriteInfo("Step 4: Testing");
                consoleService.WriteInfo("Tester validates the improved code");
                await Task.Delay(2000);
                consoleService.WriteSuccess("Testing completed");
                consoleService.WriteInfo("All tests passed (simulated)");
                Console.WriteLine();

                // Step 5: Code Application
                consoleService.WriteInfo("Step 5: Code Application");
                consoleService.WriteInfo("Coordinator applies the improved code");
                await Task.Delay(1000);
                consoleService.WriteSuccess("Code application completed");
                consoleService.WriteInfo("File successfully improved (simulated)");
                Console.WriteLine();

                // Step 6: Learning
                consoleService.WriteInfo("Step 6: Learning");
                consoleService.WriteInfo("Coordinator records successful patterns for future use");
                await Task.Delay(1000);
                consoleService.WriteSuccess("Learning completed");
                consoleService.WriteInfo("3 new patterns added to the knowledge base (simulated)");
                Console.WriteLine();

                // Stop and remove replicas
                consoleService.WriteSubHeader("Cleaning Up");
                consoleService.WriteInfo("Stopping and removing replicas...");
                foreach (var replica in replicas)
                {
                    await replicaManager.RemoveReplicaAsync(replica.Id);
                    consoleService.WriteInfo($"Replica {replica.Name} (ID: {replica.Id}) removed");
                }
                Console.WriteLine();

                // Remove Docker network
                consoleService.WriteInfo("Removing Docker network...");
                await Task.Delay(1000);
                consoleService.WriteSuccess("Docker network removed successfully (simulated)");
                Console.WriteLine();

                // Display demo footer
                consoleService.WriteHeader("DEMO COMPLETE");
                consoleService.WriteInfo("This concludes the TARS self-coding demonstration.");
                consoleService.WriteInfo("You can use the following commands to manage the self-coding process:");
                consoleService.WriteInfo("  self-code setup - Set up the self-coding environment");
                consoleService.WriteInfo("  self-code start --target <directories> --auto-apply - Start the self-coding process");
                consoleService.WriteInfo("  self-code status - Check the status of the self-coding process");
                consoleService.WriteInfo("  self-code stop - Stop the self-coding process");
                Console.WriteLine();

                // Success
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Error running self-coding demo");
                consoleService.WriteError($"Error running self-coding demo: {ex.Message}");
                // Failure
            }
        });
    }
}