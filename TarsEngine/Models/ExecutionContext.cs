using System;
using System.Collections.Generic;

namespace TarsEngine.Models;

/// <summary>
/// Represents the context for executing an improvement
/// </summary>
public class ExecutionContext
{
    /// <summary>
    /// Gets or sets the ID of the execution context
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the ID of the execution plan
    /// </summary>
    public string ExecutionPlanId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the ID of the improvement
    /// </summary>
    public string ImprovementId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the ID of the metascript
    /// </summary>
    public string MetascriptId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the timestamp when the execution context was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the timestamp when the execution context was last updated
    /// </summary>
    public DateTime? UpdatedAt { get; set; }

    /// <summary>
    /// Gets or sets the execution mode
    /// </summary>
    public ExecutionMode Mode { get; set; } = ExecutionMode.DryRun;

    /// <summary>
    /// Gets or sets the execution environment
    /// </summary>
    public ExecutionEnvironment Environment { get; set; } = ExecutionEnvironment.Sandbox;

    /// <summary>
    /// Gets or sets the execution timeout in milliseconds
    /// </summary>
    public int TimeoutMs { get; set; } = 30000;

    /// <summary>
    /// Gets or sets the execution variables
    /// </summary>
    public Dictionary<string, object> Variables { get; set; } = new Dictionary<string, object>();

    /// <summary>
    /// Gets or sets the execution state
    /// </summary>
    public Dictionary<string, object> State { get; set; } = new Dictionary<string, object>();

    /// <summary>
    /// Gets or sets the execution permissions
    /// </summary>
    public ExecutionPermissions Permissions { get; set; } = new ExecutionPermissions();

    /// <summary>
    /// Gets or sets the execution options
    /// </summary>
    public Dictionary<string, string> Options { get; set; } = new Dictionary<string, string>();

    /// <summary>
    /// Gets or sets the execution logs
    /// </summary>
    public List<ExecutionLog> Logs { get; set; } = new List<ExecutionLog>();

    /// <summary>
    /// Gets or sets the execution errors
    /// </summary>
    public List<ExecutionError> Errors { get; set; } = new List<ExecutionError>();

    /// <summary>
    /// Gets or sets the execution warnings
    /// </summary>
    public List<ExecutionWarning> Warnings { get; set; } = new List<ExecutionWarning>();

    /// <summary>
    /// Gets or sets the execution metrics
    /// </summary>
    public Dictionary<string, double> Metrics { get; set; } = new Dictionary<string, double>();

    /// <summary>
    /// Gets or sets additional metadata about the execution context
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();

    /// <summary>
    /// Gets or sets the working directory for the execution
    /// </summary>
    public string WorkingDirectory { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the backup directory for the execution
    /// </summary>
    public string BackupDirectory { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the output directory for the execution
    /// </summary>
    public string OutputDirectory { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the list of files affected by the execution
    /// </summary>
    public List<string> AffectedFiles { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the list of files backed up by the execution
    /// </summary>
    public List<string> BackedUpFiles { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the list of files modified by the execution
    /// </summary>
    public List<string> ModifiedFiles { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the list of files created by the execution
    /// </summary>
    public List<string> CreatedFiles { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the list of files deleted by the execution
    /// </summary>
    public List<string> DeletedFiles { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets whether the execution is in dry run mode
    /// </summary>
    public bool IsDryRun => Mode == ExecutionMode.DryRun;

    /// <summary>
    /// Gets or sets whether the execution is in sandbox environment
    /// </summary>
    public bool IsSandbox => Environment == ExecutionEnvironment.Sandbox;

    /// <summary>
    /// Gets or sets whether the execution has errors
    /// </summary>
    public bool HasErrors => Errors.Count > 0;

    /// <summary>
    /// Gets or sets whether the execution has warnings
    /// </summary>
    public bool HasWarnings => Warnings.Count > 0;

    /// <summary>
    /// Adds a log entry to the execution context
    /// </summary>
    /// <param name="level">The log level</param>
    /// <param name="message">The log message</param>
    /// <param name="source">The log source</param>
    public void AddLog(LogLevel level, string message, string source = "Execution")
    {
        Logs.Add(new ExecutionLog
        {
            Level = level,
            Message = message,
            Source = source,
            Timestamp = DateTime.UtcNow
        });
    }

    /// <summary>
    /// Adds an error to the execution context
    /// </summary>
    /// <param name="message">The error message</param>
    /// <param name="source">The error source</param>
    /// <param name="exception">The exception</param>
    public void AddError(string message, string source = "Execution", Exception? exception = null)
    {
        Errors.Add(new ExecutionError
        {
            Message = message,
            Source = source,
            Exception = exception,
            Timestamp = DateTime.UtcNow
        });

        AddLog(LogLevel.Error, message, source);
    }

    /// <summary>
    /// Adds a warning to the execution context
    /// </summary>
    /// <param name="message">The warning message</param>
    /// <param name="source">The warning source</param>
    public void AddWarning(string message, string source = "Execution")
    {
        Warnings.Add(new ExecutionWarning
        {
            Message = message,
            Source = source,
            Timestamp = DateTime.UtcNow
        });

        AddLog(LogLevel.Warning, message, source);
    }

    /// <summary>
    /// Sets a variable in the execution context
    /// </summary>
    /// <param name="name">The variable name</param>
    /// <param name="value">The variable value</param>
    public void SetVariable(string name, object value)
    {
        Variables[name] = value;
    }

    /// <summary>
    /// Gets a variable from the execution context
    /// </summary>
    /// <typeparam name="T">The variable type</typeparam>
    /// <param name="name">The variable name</param>
    /// <param name="defaultValue">The default value</param>
    /// <returns>The variable value</returns>
    public T GetVariable<T>(string name, T defaultValue = default!)
    {
        if (Variables.TryGetValue(name, out var value) && value is T typedValue)
        {
            return typedValue;
        }

        return defaultValue;
    }

    /// <summary>
    /// Sets a state value in the execution context
    /// </summary>
    /// <param name="name">The state name</param>
    /// <param name="value">The state value</param>
    public void SetState(string name, object value)
    {
        State[name] = value;
        UpdatedAt = DateTime.UtcNow;
    }

    /// <summary>
    /// Gets a state value from the execution context
    /// </summary>
    /// <typeparam name="T">The state type</typeparam>
    /// <param name="name">The state name</param>
    /// <param name="defaultValue">The default value</param>
    /// <returns>The state value</returns>
    public T GetState<T>(string name, T defaultValue = default!)
    {
        if (State.TryGetValue(name, out var value) && value is T typedValue)
        {
            return typedValue;
        }

        return defaultValue;
    }

    /// <summary>
    /// Sets a metric in the execution context
    /// </summary>
    /// <param name="name">The metric name</param>
    /// <param name="value">The metric value</param>
    public void SetMetric(string name, double value)
    {
        Metrics[name] = value;
    }

    /// <summary>
    /// Gets a metric from the execution context
    /// </summary>
    /// <param name="name">The metric name</param>
    /// <param name="defaultValue">The default value</param>
    /// <returns>The metric value</returns>
    public double GetMetric(string name, double defaultValue = 0.0)
    {
        if (Metrics.TryGetValue(name, out var value))
        {
            return value;
        }

        return defaultValue;
    }

    /// <summary>
    /// Adds a file to the list of affected files
    /// </summary>
    /// <param name="filePath">The file path</param>
    public void AddAffectedFile(string filePath)
    {
        if (!AffectedFiles.Contains(filePath))
        {
            AffectedFiles.Add(filePath);
        }
    }

    /// <summary>
    /// Adds a file to the list of backed up files
    /// </summary>
    /// <param name="filePath">The file path</param>
    public void AddBackedUpFile(string filePath)
    {
        if (!BackedUpFiles.Contains(filePath))
        {
            BackedUpFiles.Add(filePath);
        }
    }

    /// <summary>
    /// Adds a file to the list of modified files
    /// </summary>
    /// <param name="filePath">The file path</param>
    public void AddModifiedFile(string filePath)
    {
        if (!ModifiedFiles.Contains(filePath))
        {
            ModifiedFiles.Add(filePath);
        }
        AddAffectedFile(filePath);
    }

    /// <summary>
    /// Adds a file to the list of created files
    /// </summary>
    /// <param name="filePath">The file path</param>
    public void AddCreatedFile(string filePath)
    {
        if (!CreatedFiles.Contains(filePath))
        {
            CreatedFiles.Add(filePath);
        }
        AddAffectedFile(filePath);
    }

    /// <summary>
    /// Adds a file to the list of deleted files
    /// </summary>
    /// <param name="filePath">The file path</param>
    public void AddDeletedFile(string filePath)
    {
        if (!DeletedFiles.Contains(filePath))
        {
            DeletedFiles.Add(filePath);
        }
        AddAffectedFile(filePath);
    }
}
