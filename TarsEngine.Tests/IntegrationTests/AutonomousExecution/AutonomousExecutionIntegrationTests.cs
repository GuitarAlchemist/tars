using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;
using System.Reflection;
using TarsEngine.Models;
using TarsEngine.Services;
using TarsEngine.Services.Interfaces;
using Xunit;

// Use alias to avoid ambiguity with System.Threading.ExecutionContext
using ExecutionContextModel = TarsEngine.Models.ExecutionContext;
using MsLogLevel = Microsoft.Extensions.Logging.LogLevel;

namespace TarsEngine.Tests.IntegrationTests.AutonomousExecution;

public class AutonomousExecutionIntegrationTests
{
    private IServiceProvider _serviceProvider;
    private IExecutionPlannerService _executionPlannerService;
    private SafeExecutionEnvironment _safeExecutionEnvironment;
    private RollbackManager _rollbackManager;
    private IImprovementPrioritizerService _improvementPrioritizerService;

    public AutonomousExecutionIntegrationTests()
    {
        // Create service collection
        var services = new ServiceCollection();

        // Add logging
        services.AddLogging(builder =>
        {
            builder.AddSimpleConsole(options =>
            {
                options.IncludeScopes = true;
                options.SingleLine = true;
                options.TimestampFormat = "HH:mm:ss ";
            });
            builder.SetMinimumLevel(MsLogLevel.Debug);
        });

        // Add services
        services.AddSingleton<IExecutionPlannerService, ExecutionPlannerService>();
        services.AddSingleton<SafeExecutionEnvironment>();
        services.AddSingleton<ChangeValidator>();
        services.AddSingleton<RollbackManager>();
        services.AddSingleton<IImprovementPrioritizerService, ImprovementPrioritizerService>();
        services.AddSingleton<SyntaxValidator>();
        services.AddSingleton<SemanticValidator>();
        services.AddSingleton<TestExecutor>();
        services.AddSingleton<FileBackupService>();
        services.AddSingleton<TransactionManager>();
        services.AddSingleton<AuditTrailService>();
        services.AddSingleton<PermissionManager>();
        services.AddSingleton<VirtualFileSystem>();

        // Build service provider
        _serviceProvider = services.BuildServiceProvider();

        // Get services
        _executionPlannerService = _serviceProvider.GetRequiredService<IExecutionPlannerService>();
        _safeExecutionEnvironment = _serviceProvider.GetRequiredService<SafeExecutionEnvironment>();
        _serviceProvider.GetRequiredService<ChangeValidator>();
        _rollbackManager = _serviceProvider.GetRequiredService<RollbackManager>();
        _improvementPrioritizerService = _serviceProvider.GetRequiredService<IImprovementPrioritizerService>();
    }

    [Fact]
    public async Task TestBasicExecutionFlow()
    {
        // Create test improvement
        var improvement = CreateTestImprovement();
        await _improvementPrioritizerService.SaveImprovementAsync(improvement);

        // Create execution plan
        var executionPlan = await _executionPlannerService.CreateExecutionPlanAsync(improvement);
        Assert.NotNull(executionPlan);

        // Validate execution plan
        var isValid = await _executionPlannerService.ValidateExecutionPlanAsync(executionPlan);
        Assert.True(isValid);

        // Create execution context
        var executionContext = await _executionPlannerService.CreateExecutionContextAsync(executionPlan, new Dictionary<string, string>
        {
            { "Mode", ExecutionMode.DryRun.ToString() },
            { "Environment", ExecutionEnvironment.Development.ToString() },
            { "OutputDirectory", Path.Combine(Path.GetTempPath(), "TARS", "Output", executionPlan.Id) }
        });
        Assert.NotNull(executionContext);

        // Create rollback context
        var backupDirectory = Path.Combine(Path.GetTempPath(), "TARS", "Backup", executionPlan.Id);
        var result = _rollbackManager.CreateRollbackContext(executionContext.Id, "TARS", $"Execution of improvement: {improvement.Id}", backupDirectory);
        Assert.True(result);

        // Execute plan
        var executionResult = await _executionPlannerService.ExecuteExecutionPlanAsync(executionPlan, new Dictionary<string, string>
        {
            { "Validate", "true" },
            { "AutoRollback", "true" }
        });
        Assert.NotNull(executionResult);
        Assert.True(executionResult.IsSuccessful);

        // Get execution context
        var updatedContext = await _executionPlannerService.GetExecutionContextAsync(executionContext.Id);
        Assert.NotNull(updatedContext);
        Assert.Equal(executionContext.Id, updatedContext.Id);

        // Clean up
        await _executionPlannerService.RemoveExecutionPlanAsync(executionPlan.Id);
        _rollbackManager.RemoveRollbackContext(executionContext.Id, true);
        await _improvementPrioritizerService.RemoveImprovementAsync(improvement.Id);
    }

    [Fact]
    public async Task TestExecutionWithRollback()
    {
        // Create test improvement
        var improvement = CreateTestImprovement();
        await _improvementPrioritizerService.SaveImprovementAsync(improvement);

        // Create execution plan
        var executionPlan = await _executionPlannerService.CreateExecutionPlanAsync(improvement);
        Assert.NotNull(executionPlan);

        // Validate execution plan
        var isValid = await _executionPlannerService.ValidateExecutionPlanAsync(executionPlan);
        Assert.True(isValid);

        // Create execution context
        var executionContext = await _executionPlannerService.CreateExecutionContextAsync(executionPlan, new Dictionary<string, string>
        {
            { "Mode", ExecutionMode.Real.ToString() },
            { "Environment", ExecutionEnvironment.Development.ToString() },
            { "OutputDirectory", Path.Combine(Path.GetTempPath(), "TARS", "Output", executionPlan.Id) }
        });
        Assert.NotNull(executionContext);

        // Create rollback context
        var backupDirectory = Path.Combine(Path.GetTempPath(), "TARS", "Backup", executionPlan.Id);
        var result = _rollbackManager.CreateRollbackContext(executionContext.Id, "TARS", $"Execution of improvement: {improvement.Id}", backupDirectory);
        Assert.True(result);

        // Create test file
        var testFilePath = Path.Combine(Path.GetTempPath(), "TARS", "Test", "test.txt");
        Directory.CreateDirectory(Path.GetDirectoryName(testFilePath));
        File.WriteAllText(testFilePath, "Original content");

        // Add file to affected files
        executionContext.AddAffectedFile(testFilePath);
        await _executionPlannerService.SaveExecutionContextAsync(executionContext);

        // Backup file
        var backupFilePath = await _rollbackManager.BackupFileAsync(executionContext.Id, string.Empty, testFilePath);
        Assert.NotNull(backupFilePath);

        // Modify file
        File.WriteAllText(testFilePath, "Modified content");

        // Roll back changes
        var rollbackResult = await _rollbackManager.RestoreFileAsync(executionContext.Id, string.Empty, testFilePath);
        Assert.True(rollbackResult);

        // Verify file content
        var content = File.ReadAllText(testFilePath);
        Assert.Equal("Original content", content);

        // Clean up
        File.Delete(testFilePath);
        await _executionPlannerService.RemoveExecutionPlanAsync(executionPlan.Id);
        _rollbackManager.RemoveRollbackContext(executionContext.Id, true);
        await _improvementPrioritizerService.RemoveImprovementAsync(improvement.Id);
    }

    private PrioritizedImprovement CreateTestImprovement()
    {
        return new PrioritizedImprovement
        {
            Id = Guid.NewGuid().ToString(),
            Name = "Test Improvement",
            Description = "A test improvement for integration testing",
            Category = ImprovementCategory.Other,
            Impact = ImprovementImpact.Medium,
            Effort = ImprovementEffort.Medium,
            Risk = ImprovementRisk.Low,
            PriorityScore = 0.75,
            PriorityRank = 1,
            Status = ImprovementStatus.Approved,
            CreatedAt = DateTime.UtcNow,
            PrioritizedAt = DateTime.UtcNow
        };
    }
}

// Mock implementation of ExecutionPlannerService for testing
public class ExecutionPlannerService : IExecutionPlannerService
{
    private readonly Dictionary<string, ExecutionPlan> _executionPlans = new();
    private readonly Dictionary<string, ExecutionContextModel> _executionContexts = new();
    private readonly Dictionary<string, ExecutionPlanResult> _executionResults = new();

    public async Task<ExecutionPlan> CreateExecutionPlanAsync(PrioritizedImprovement improvement, Dictionary<string, string>? options = null)
    {
        var plan = new ExecutionPlan
        {
            Id = Guid.NewGuid().ToString(),
            Name = $"Execution Plan for {improvement.Name}",
            Description = improvement.Description,
            ImprovementId = improvement.Id,
            CreatedAt = DateTime.UtcNow
        };

        // Add a simple step
        plan.Steps.Add(new ExecutionStep
        {
            Id = Guid.NewGuid().ToString(),
            Name = "Test Step",
            Description = "A test step",
            Type = ExecutionStepType.Other,
            Order = 1
        });

        _executionPlans[plan.Id] = plan;
        return plan;
    }

    public async Task<ExecutionPlan> CreateExecutionPlanAsync(GeneratedMetascript metascript, Dictionary<string, string>? options = null)
    {
        var plan = new ExecutionPlan
        {
            Id = Guid.NewGuid().ToString(),
            Name = $"Execution Plan for Metascript",
            Description = "Metascript execution plan",
            MetascriptId = metascript.Id,
            CreatedAt = DateTime.UtcNow
        };

        // Add a simple step
        plan.Steps.Add(new ExecutionStep
        {
            Id = Guid.NewGuid().ToString(),
            Name = "Test Step",
            Description = "A test step",
            Type = ExecutionStepType.Other,
            Order = 1
        });

        _executionPlans[plan.Id] = plan;
        return plan;
    }

    public async Task<bool> ValidateExecutionPlanAsync(ExecutionPlan plan, Dictionary<string, string>? options = null)
    {
        // Simple validation: check if plan has steps
        return plan.Steps.Count > 0;
    }

    public async Task<ExecutionPlanResult> ExecuteExecutionPlanAsync(ExecutionPlan plan, Dictionary<string, string>? options = null)
    {
        var result = new ExecutionPlanResult
        {
            ExecutionPlanId = plan.Id,
            Status = ExecutionPlanStatus.Completed,
            IsSuccessful = true,
            StartedAt = DateTime.UtcNow,
            CompletedAt = DateTime.UtcNow,
            DurationMs = 100
        };

        _executionResults[plan.Id] = result;
        return result;
    }

    public async Task<ExecutionPlan?> GetExecutionPlanAsync(string planId)
    {
        return _executionPlans.TryGetValue(planId, out var plan) ? plan : null;
    }

    public async Task<List<ExecutionPlan>> GetExecutionPlansAsync(Dictionary<string, string>? options = null)
    {
        return [.._executionPlans.Values];
    }

    public async Task<bool> SaveExecutionPlanAsync(ExecutionPlan plan)
    {
        _executionPlans[plan.Id] = plan;
        return true;
    }

    public async Task<bool> RemoveExecutionPlanAsync(string planId)
    {
        return _executionPlans.Remove(planId);
    }

    public async Task<ExecutionContextModel> CreateExecutionContextAsync(ExecutionPlan plan, Dictionary<string, string>? options = null)
    {
        var context = new ExecutionContextModel
        {
            Id = Guid.NewGuid().ToString(),
            ExecutionPlanId = plan.Id,
            ImprovementId = plan.ImprovementId,
            MetascriptId = plan.MetascriptId,
            Mode = options?.TryGetValue("Mode", out var mode) == true && Enum.TryParse<ExecutionMode>(mode, out var modeEnum)
                ? modeEnum
                : ExecutionMode.DryRun,
            Environment = options?.TryGetValue("Environment", out var env) == true && Enum.TryParse<ExecutionEnvironment>(env, out var envEnum)
                ? envEnum
                : ExecutionEnvironment.Development,
            CreatedAt = DateTime.UtcNow
        };

        if (options?.TryGetValue("OutputDirectory", out var outputDir) == true)
        {
            context.OutputDirectory = outputDir;
        }

        _executionContexts[context.Id] = context;
        return context;
    }

    public async Task<ExecutionContextModel?> GetExecutionContextAsync(string contextId)
    {
        return _executionContexts.TryGetValue(contextId, out var context) ? context : null;
    }

    public async Task<bool> SaveExecutionContextAsync(ExecutionContextModel context)
    {
        _executionContexts[context.Id] = context;
        return true;
    }

    public async Task<ExecutionPlanResult?> GetExecutionResultAsync(string planId)
    {
        return _executionResults.TryGetValue(planId, out var result) ? result : null;
    }

    public async Task<bool> SaveExecutionResultAsync(ExecutionPlanResult result)
    {
        _executionResults[result.ExecutionPlanId] = result;
        return true;
    }

    public async Task<Dictionary<string, string>> GetAvailablePlanningOptionsAsync()
    {
        return new Dictionary<string, string>
        {
            { "ValidateSteps", "Whether to validate steps (true, false)" },
            { "IncludeDependencies", "Whether to include dependencies (true, false)" }
        };
    }

    public async Task<Dictionary<string, string>> GetAvailableExecutionOptionsAsync()
    {
        return new Dictionary<string, string>
        {
            { "Validate", "Whether to validate changes (true, false)" },
            { "AutoRollback", "Whether to automatically roll back on failure (true, false)" }
        };
    }
}

// Mock implementation of ImprovementPrioritizerService for testing
public class ImprovementPrioritizerService : IImprovementPrioritizerService
{
    private readonly Dictionary<string, PrioritizedImprovement> _improvements = new();
    private readonly Dictionary<string, StrategicGoal> _goals = new();

    // This method is not part of the interface but is used by the test
    public async Task<List<PrioritizedImprovement>> PrioritizeImprovementsAsync(List<GeneratedImprovement> improvements, Dictionary<string, string>? options = null)
    {
        var prioritizedImprovements = new List<PrioritizedImprovement>();
        foreach (var improvement in improvements)
        {
            var prioritizedImprovement = new PrioritizedImprovement
            {
                Id = improvement.Id,
                Name = improvement.Name,
                Description = improvement.Description,
                Category = improvement.Category,
                Status = improvement.Status,
                CreatedAt = improvement.CreatedAt,
                UpdatedAt = improvement.UpdatedAt,
                MetascriptId = improvement.MetascriptId,
                PatternMatchId = improvement.PatternMatchId,
                AffectedFiles = improvement.AffectedFiles,
                Tags = improvement.Tags,
                Metadata = improvement.Metadata,
                Impact = ImprovementImpact.Medium,
                Effort = ImprovementEffort.Medium,
                Risk = ImprovementRisk.Medium,
                PriorityScore = 0.5,
                PriorityRank = prioritizedImprovements.Count + 1,
                PrioritizedAt = DateTime.UtcNow
            };
            prioritizedImprovements.Add(prioritizedImprovement);
            _improvements[prioritizedImprovement.Id] = prioritizedImprovement;
        }
        return prioritizedImprovements;
    }

    public async Task<PrioritizedImprovement?> GetImprovementAsync(string improvementId)
    {
        return _improvements.TryGetValue(improvementId, out var improvement) ? improvement : null;
    }

    public async Task<List<PrioritizedImprovement>> GetImprovementsAsync(Dictionary<string, string>? options = null)
    {
        return [.._improvements.Values];
    }

    public async Task<bool> SaveImprovementAsync(PrioritizedImprovement improvement)
    {
        _improvements[improvement.Id] = improvement;
        return true;
    }

    public async Task<bool> RemoveImprovementAsync(string improvementId)
    {
        return _improvements.Remove(improvementId);
    }

    public async Task<Dictionary<string, string>> GetAvailablePrioritizationOptionsAsync()
    {
        return new Dictionary<string, string>
        {
            { "ImpactWeight", "Weight for impact (0.0-1.0)" },
            { "EffortWeight", "Weight for effort (0.0-1.0)" },
            { "RiskWeight", "Weight for risk (0.0-1.0)" }
        };
    }

    // Implement the missing methods from the interface
    public async Task<PrioritizedImprovement> PrioritizeImprovementAsync(PrioritizedImprovement improvement, Dictionary<string, string>? options = null)
    {
        improvement.PriorityScore = 0.5; // Default priority score
        improvement.PrioritizedAt = DateTime.UtcNow;
        return improvement;
    }

    public async Task<List<PrioritizedImprovement>> PrioritizeImprovementsAsync(List<PrioritizedImprovement> improvements, Dictionary<string, string>? options = null)
    {
        foreach (var improvement in improvements)
        {
            await PrioritizeImprovementAsync(improvement, options);
        }
        return improvements;
    }

    public async Task<PrioritizedImprovement> CreateImprovementFromMetascriptAsync(GeneratedMetascript metascript, Dictionary<string, string>? options = null)
    {
        var improvement = new PrioritizedImprovement
        {
            Id = Guid.NewGuid().ToString(),
            Name = metascript.Name,
            Description = metascript.Description,
            MetascriptId = metascript.Id,
            CreatedAt = DateTime.UtcNow,
            Status = ImprovementStatus.Pending
        };
        _improvements[improvement.Id] = improvement;
        return improvement;
    }

    public async Task<PrioritizedImprovement> CreateImprovementFromPatternMatchAsync(PatternMatch patternMatch, Dictionary<string, string>? options = null)
    {
        var improvement = new PrioritizedImprovement
        {
            Id = Guid.NewGuid().ToString(),
            Name = $"Pattern match: {patternMatch.PatternName}",
            Description = patternMatch.ExpectedImprovement ?? $"Improvement based on pattern: {patternMatch.PatternName}",
            PatternMatchId = patternMatch.Id,
            CreatedAt = DateTime.UtcNow,
            Status = ImprovementStatus.Pending
        };
        _improvements[improvement.Id] = improvement;
        return improvement;
    }

    public async Task<bool> UpdateImprovementAsync(PrioritizedImprovement improvement)
    {
        improvement.UpdatedAt = DateTime.UtcNow;
        _improvements[improvement.Id] = improvement;
        return true;
    }

    public async Task<List<PrioritizedImprovement>> GetNextImprovementsAsync(int count, Dictionary<string, string>? options = null)
    {
        return _improvements.Values.OrderByDescending(i => i.PriorityScore).Take(count).ToList();
    }

    public async Task<List<StrategicGoal>> GetStrategicGoalsAsync(Dictionary<string, string>? options = null)
    {
        return [.._goals.Values];
    }

    public async Task<StrategicGoal?> GetStrategicGoalAsync(string goalId)
    {
        return _goals.TryGetValue(goalId, out var goal) ? goal : null;
    }

    public async Task<bool> AddStrategicGoalAsync(StrategicGoal goal)
    {
        _goals[goal.Id] = goal;
        return true;
    }

    public async Task<bool> UpdateStrategicGoalAsync(StrategicGoal goal)
    {
        goal.UpdatedAt = DateTime.UtcNow;
        _goals[goal.Id] = goal;
        return true;
    }

    public async Task<bool> RemoveStrategicGoalAsync(string goalId)
    {
        return _goals.Remove(goalId);
    }

    public async Task<ImprovementDependencyGraph> GetDependencyGraphAsync(Dictionary<string, string>? options = null)
    {
        return new ImprovementDependencyGraph
        {
            Nodes = _improvements.Values.Select(i => new ImprovementNode { Id = i.Id, Name = i.Name }).ToList(),
            Edges = []
        };
    }

    public async Task<Dictionary<string, string>> GetAvailableOptionsAsync()
    {
        return new Dictionary<string, string>
        {
            { "SortBy", "Priority, Impact, Effort, Risk" },
            { "SortDirection", "Ascending, Descending" },
            { "FilterByStatus", "Pending, InProgress, Completed, etc." },
            { "FilterByCategory", "Performance, Security, Maintainability, etc." }
        };
    }
}
