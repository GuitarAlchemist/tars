using TarsEngine.Models;
using TarsEngine.Services.Interfaces;
using MsLogLevel = Microsoft.Extensions.Logging.LogLevel;

namespace TarsApp.Services;

public class MockExecutionService : IExecutionService
{
    private readonly ILogger<MockExecutionService> _logger;
    private readonly List<ExecutionPlan> _executionPlans = [];

    public MockExecutionService(ILogger<MockExecutionService> logger)
    {
        _logger = logger;

        // Add some sample execution plans
        var plan1 = new ExecutionPlan
        {
            Id = Guid.NewGuid().ToString(),
            Name = "Sample Execution Plan 1",
            Description = "This is a sample execution plan for demonstration purposes.",
            Status = ExecutionPlanStatus.Completed,
            CreatedAt = DateTime.UtcNow.AddHours(-3),
            UpdatedAt = DateTime.UtcNow.AddHours(-1),
            ExecutedAt = DateTime.UtcNow.AddHours(-2),
            Metadata = new Dictionary<string, string>
            {
                { "Tags", "sample,demo" }
            },
            Result = new ExecutionPlanResult
            {
                IsSuccessful = true,
                StartedAt = DateTime.UtcNow.AddHours(-2),
                CompletedAt = DateTime.UtcNow.AddHours(-1),
                DurationMs = 3600000, // 1 hour
                Output = "Execution completed successfully"
            },
            Context = new TarsEngine.Models.ExecutionContext
            {
                Id = Guid.NewGuid().ToString(),
                Mode = ExecutionMode.DryRun,
                Environment = ExecutionEnvironment.Development,
                CreatedAt = DateTime.UtcNow.AddHours(-2),
                UpdatedAt = DateTime.UtcNow.AddHours(-1),
                Logs =
                [
                    new ExecutionLog
                    {
                        Timestamp = DateTime.UtcNow.AddHours(-2),
                        Level = TarsEngine.Models.LogLevel.Information,
                        Message = "Execution started",
                        Source = "MockExecutionService"
                    },

                    new ExecutionLog
                    {
                        Timestamp = DateTime.UtcNow.AddHours(-1),
                        Level = TarsEngine.Models.LogLevel.Information,
                        Message = "Execution completed",
                        Source = "MockExecutionService"
                    }
                ]
            },
            Steps =
            [
                new ExecutionStep
                {
                    Id = Guid.NewGuid().ToString(),
                    Name = "Step 1",
                    Description = "First step of the execution plan",
                    Status = ExecutionStepStatus.Completed,
                    StartedAt = DateTime.UtcNow.AddHours(-2),
                    CompletedAt = DateTime.UtcNow.AddHours(-1.5),
                    DurationMs = 1800000, // 30 minutes
                    Result = new ExecutionStepResult
                    {
                        IsSuccessful = true,
                        StartedAt = DateTime.UtcNow.AddHours(-2),
                        CompletedAt = DateTime.UtcNow.AddHours(-1.5),
                        DurationMs = 1800000, // 30 minutes
                        Output = "Step 1 completed successfully"
                    }
                },

                new ExecutionStep
                {
                    Id = Guid.NewGuid().ToString(),
                    Name = "Step 2",
                    Description = "Second step of the execution plan",
                    Status = ExecutionStepStatus.Completed,
                    StartedAt = DateTime.UtcNow.AddHours(-1.5),
                    CompletedAt = DateTime.UtcNow.AddHours(-1),
                    DurationMs = 1800000, // 30 minutes
                    Result = new ExecutionStepResult
                    {
                        IsSuccessful = true,
                        StartedAt = DateTime.UtcNow.AddHours(-1.5),
                        CompletedAt = DateTime.UtcNow.AddHours(-1),
                        DurationMs = 1800000, // 30 minutes
                        Output = "Step 2 completed successfully"
                    }
                }
            ]
        };
        _executionPlans.Add(plan1);

        var plan2 = new ExecutionPlan
        {
            Id = Guid.NewGuid().ToString(),
            Name = "Sample Execution Plan 2",
            Description = "This is another sample execution plan for demonstration purposes.",
            Status = ExecutionPlanStatus.InProgress,
            CreatedAt = DateTime.UtcNow.AddMinutes(-45),
            UpdatedAt = DateTime.UtcNow.AddMinutes(-15),
            ExecutedAt = DateTime.UtcNow.AddMinutes(-30),
            Metadata = new Dictionary<string, string>
            {
                { "Tags", "sample,demo,in-progress" }
            },
            Result = new ExecutionPlanResult
            {
                IsSuccessful = false,
                StartedAt = DateTime.UtcNow.AddMinutes(-30),
                CompletedAt = null,
                DurationMs = null,
                Output = "Execution in progress"
            },
            Context = new TarsEngine.Models.ExecutionContext
            {
                Id = Guid.NewGuid().ToString(),
                Mode = ExecutionMode.DryRun,
                Environment = ExecutionEnvironment.Development,
                CreatedAt = DateTime.UtcNow.AddMinutes(-30),
                UpdatedAt = DateTime.UtcNow.AddMinutes(-15),
                Logs =
                [
                    new ExecutionLog
                    {
                        Timestamp = DateTime.UtcNow.AddMinutes(-30),
                        Level = TarsEngine.Models.LogLevel.Information,
                        Message = "Execution started",
                        Source = "MockExecutionService"
                    },

                    new ExecutionLog
                    {
                        Timestamp = DateTime.UtcNow.AddMinutes(-15),
                        Level = TarsEngine.Models.LogLevel.Information,
                        Message = "Step 1 completed",
                        Source = "MockExecutionService"
                    }
                ]
            },
            Steps =
            [
                new ExecutionStep
                {
                    Id = Guid.NewGuid().ToString(),
                    Name = "Step 1",
                    Description = "First step of the execution plan",
                    Status = ExecutionStepStatus.Completed,
                    StartedAt = DateTime.UtcNow.AddMinutes(-30),
                    CompletedAt = DateTime.UtcNow.AddMinutes(-15),
                    DurationMs = 900000, // 15 minutes
                    Result = new ExecutionStepResult
                    {
                        IsSuccessful = true,
                        StartedAt = DateTime.UtcNow.AddMinutes(-30),
                        CompletedAt = DateTime.UtcNow.AddMinutes(-15),
                        DurationMs = 900000, // 15 minutes
                        Output = "Step 1 completed successfully"
                    }
                },

                new ExecutionStep
                {
                    Id = Guid.NewGuid().ToString(),
                    Name = "Step 2",
                    Description = "Second step of the execution plan",
                    Status = ExecutionStepStatus.InProgress,
                    StartedAt = DateTime.UtcNow.AddMinutes(-15),
                    CompletedAt = null,
                    DurationMs = null,
                    Result = null
                }
            ]
        };
        _executionPlans.Add(plan2);
    }

    public Task<List<ExecutionPlan>> GetExecutionPlansAsync()
    {
        return Task.FromResult(_executionPlans);
    }

    public Task<ExecutionPlan> GetExecutionPlanAsync(string id)
    {
        var plan = _executionPlans.Find(p => p.Id == id);
        return Task.FromResult(plan);
    }

    public Task<ExecutionPlan> CreateExecutionPlanAsync(string name, string description, List<string> tags)
    {
        var plan = new ExecutionPlan
        {
            Id = Guid.NewGuid().ToString(),
            Name = name,
            Description = description,
            Status = ExecutionPlanStatus.Created,
            CreatedAt = DateTime.UtcNow,
            UpdatedAt = DateTime.UtcNow,
            Metadata = new Dictionary<string, string>
            {
                { "Tags", string.Join(",", tags ?? []) }
            },
            Context = new TarsEngine.Models.ExecutionContext
            {
                Id = Guid.NewGuid().ToString(),
                Mode = ExecutionMode.DryRun,
                Environment = ExecutionEnvironment.Development,
                CreatedAt = DateTime.UtcNow,
                Logs = []
            },
            Steps = []
        };

        _executionPlans.Add(plan);
        return Task.FromResult(plan);
    }

    public Task<ExecutionPlan> StartExecutionAsync(string id)
    {
        var plan = _executionPlans.Find(p => p.Id == id);
        if (plan != null)
        {
            plan.Status = ExecutionPlanStatus.InProgress;
            plan.ExecutedAt = DateTime.UtcNow;
            plan.UpdatedAt = DateTime.UtcNow;

            // Create or update result
            if (plan.Result == null)
            {
                plan.Result = new ExecutionPlanResult
                {
                    ExecutionPlanId = plan.Id,
                    Status = ExecutionPlanStatus.InProgress,
                    StartedAt = DateTime.UtcNow,
                    Output = "Execution in progress"
                };
            }
            else
            {
                plan.Result.Status = ExecutionPlanStatus.InProgress;
                plan.Result.StartedAt = DateTime.UtcNow;
                plan.Result.Output = "Execution in progress";
            }

            // Add log entry
            if (plan.Context != null)
            {
                plan.Context.UpdatedAt = DateTime.UtcNow;
                plan.Context.Logs.Add(new ExecutionLog
                {
                    Timestamp = DateTime.UtcNow,
                    Level = TarsEngine.Models.LogLevel.Information,
                    Message = "Execution started",
                    Source = "MockExecutionService"
                });
            }
        }
        return Task.FromResult(plan);
    }

    public Task<ExecutionPlan> StopExecutionAsync(string id)
    {
        var plan = _executionPlans.Find(p => p.Id == id);
        if (plan != null)
        {
            plan.Status = ExecutionPlanStatus.Completed;
            plan.UpdatedAt = DateTime.UtcNow;

            // Update result
            if (plan.Result != null)
            {
                plan.Result.Status = ExecutionPlanStatus.Completed;
                plan.Result.CompletedAt = DateTime.UtcNow;
                plan.Result.IsSuccessful = true;

                // Calculate duration if start time is available
                if (plan.Result.StartedAt.HasValue)
                {
                    var duration = DateTime.UtcNow - plan.Result.StartedAt.Value;
                    plan.Result.DurationMs = (long)duration.TotalMilliseconds;
                }

                plan.Result.Output = "Execution completed successfully";
            }

            // Add log entry
            if (plan.Context != null)
            {
                plan.Context.UpdatedAt = DateTime.UtcNow;
                plan.Context.Logs.Add(new ExecutionLog
                {
                    Timestamp = DateTime.UtcNow,
                    Level = TarsEngine.Models.LogLevel.Information,
                    Message = "Execution stopped",
                    Source = "MockExecutionService"
                });
            }
        }
        return Task.FromResult(plan);
    }

    public Task<bool> DeleteExecutionPlanAsync(string id)
    {
        var plan = _executionPlans.Find(p => p.Id == id);
        if (plan != null)
        {
            _executionPlans.Remove(plan);
            return Task.FromResult(true);
        }
        return Task.FromResult(false);
    }

    public Task<List<TarsEngine.Models.LogEntry>> GetExecutionLogsAsync(string id)
    {
        var plan = _executionPlans.Find(p => p.Id == id);
        if (plan?.Context?.Logs == null)
        {
            return Task.FromResult(new List<TarsEngine.Models.LogEntry>());
        }

        // Convert ExecutionLog to TarsEngine.Models.LogEntry
        var logEntries = plan.Context.Logs.Select(log => new TarsEngine.Models.LogEntry
        {
            Timestamp = log.Timestamp,
            Level = log.Level,
            Message = log.Message,
            Source = log.Source,
            Category = log.Source // Use Source as Category since ExecutionLog doesn't have Category
        }).ToList();

        return Task.FromResult(logEntries);
    }

    // Helper method to convert TarsEngine.Models.LogLevel to Microsoft.Extensions.Logging.LogLevel
    private MsLogLevel ConvertToMsLogLevel(TarsEngine.Models.LogLevel level)
    {
        return level switch
        {
            TarsEngine.Models.LogLevel.Trace => MsLogLevel.Trace,
            TarsEngine.Models.LogLevel.Debug => MsLogLevel.Debug,
            TarsEngine.Models.LogLevel.Information => MsLogLevel.Information,
            TarsEngine.Models.LogLevel.Warning => MsLogLevel.Warning,
            TarsEngine.Models.LogLevel.Error => MsLogLevel.Error,
            TarsEngine.Models.LogLevel.Critical => MsLogLevel.Critical,
            _ => MsLogLevel.None
        };
    }
}