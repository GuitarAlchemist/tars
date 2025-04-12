using System.Text;

namespace TarsCli.Services;

/// <summary>
/// Service for tracking and logging TARS operations
/// </summary>
public class OperationSummaryService
{
    private readonly ILogger<OperationSummaryService> _logger;
    private readonly string _summaryDirectory;
    private readonly List<OperationRecord> _operations = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="OperationSummaryService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public OperationSummaryService(ILogger<OperationSummaryService> logger)
    {
        _logger = logger;
        
        // Create directory for summary files
        var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
        _summaryDirectory = Path.Combine(appDataPath, "TARS", "Summaries");
        
        if (!Directory.Exists(_summaryDirectory))
        {
            Directory.CreateDirectory(_summaryDirectory);
        }
        
        _logger.LogInformation($"Operation summaries will be stored in {_summaryDirectory}");
    }
    
    /// <summary>
    /// Record a file creation operation
    /// </summary>
    /// <param name="filePath">The path of the created file</param>
    /// <param name="description">A brief description of the operation</param>
    public void RecordFileCreation(string filePath, string description)
    {
        _operations.Add(new OperationRecord
        {
            Type = OperationType.FileCreation,
            Path = filePath,
            Description = description,
            Timestamp = DateTime.Now
        });
        
        _logger.LogInformation($"Recorded file creation: {filePath} - {description}");
    }
    
    /// <summary>
    /// Record a file modification operation
    /// </summary>
    /// <param name="filePath">The path of the modified file</param>
    /// <param name="description">A brief description of the operation</param>
    public void RecordFileModification(string filePath, string description)
    {
        _operations.Add(new OperationRecord
        {
            Type = OperationType.FileModification,
            Path = filePath,
            Description = description,
            Timestamp = DateTime.Now
        });
        
        _logger.LogInformation($"Recorded file modification: {filePath} - {description}");
    }
    
    /// <summary>
    /// Record a task completion operation
    /// </summary>
    /// <param name="taskName">The name of the completed task</param>
    /// <param name="description">A brief description of the operation</param>
    public void RecordTaskCompletion(string taskName, string description)
    {
        _operations.Add(new OperationRecord
        {
            Type = OperationType.TaskCompletion,
            Path = taskName,
            Description = description,
            Timestamp = DateTime.Now
        });
        
        _logger.LogInformation($"Recorded task completion: {taskName} - {description}");
    }
    
    /// <summary>
    /// Record an LLM operation
    /// </summary>
    /// <param name="model">The model used</param>
    /// <param name="description">A brief description of the operation</param>
    public void RecordLlmOperation(string model, string description)
    {
        _operations.Add(new OperationRecord
        {
            Type = OperationType.LlmOperation,
            Path = model,
            Description = description,
            Timestamp = DateTime.Now
        });
        
        _logger.LogInformation($"Recorded LLM operation: {model} - {description}");
    }
    
    /// <summary>
    /// Save the operation summary to a file
    /// </summary>
    /// <returns>The path to the summary file</returns>
    public string SaveSummary()
    {
        if (_operations.Count == 0)
        {
            _logger.LogInformation("No operations to save");
            return string.Empty;
        }
        
        var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var summaryFilePath = Path.Combine(_summaryDirectory, $"TARS_Summary_{timestamp}.txt");
        
        var sb = new StringBuilder();
        sb.AppendLine("# TARS Operation Summary");
        sb.AppendLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine();
        
        // Group operations by type
        var groupedOperations = _operations.GroupBy(o => o.Type);
        
        foreach (var group in groupedOperations)
        {
            sb.AppendLine($"## {GetOperationTypeTitle(group.Key)}");
            sb.AppendLine();
            
            foreach (var operation in group.OrderBy(o => o.Timestamp))
            {
                sb.AppendLine($"- {operation.Timestamp:HH:mm:ss} | {operation.Path}");
                sb.AppendLine($"  {operation.Description}");
                sb.AppendLine();
            }
        }
        
        // Add summary statistics
        sb.AppendLine("## Summary Statistics");
        sb.AppendLine();
        sb.AppendLine($"- Total operations: {_operations.Count}");
        
        foreach (var group in groupedOperations)
        {
            sb.AppendLine($"- {GetOperationTypeTitle(group.Key)}: {group.Count()}");
        }
        
        // Write to file
        File.WriteAllText(summaryFilePath, sb.ToString());
        
        _logger.LogInformation($"Saved operation summary to {summaryFilePath}");
        
        // Clear operations after saving
        _operations.Clear();
        
        return summaryFilePath;
    }
    
    private string GetOperationTypeTitle(OperationType type)
    {
        return type switch
        {
            OperationType.FileCreation => "File Creations",
            OperationType.FileModification => "File Modifications",
            OperationType.TaskCompletion => "Task Completions",
            OperationType.LlmOperation => "LLM Operations",
            _ => "Other Operations"
        };
    }
    
    /// <summary>
    /// Record type for operations
    /// </summary>
    private class OperationRecord
    {
        /// <summary>
        /// The type of operation
        /// </summary>
        public OperationType Type { get; set; }
        
        /// <summary>
        /// The path or identifier for the operation
        /// </summary>
        public string Path { get; set; } = string.Empty;
        
        /// <summary>
        /// A brief description of the operation
        /// </summary>
        public string Description { get; set; } = string.Empty;
        
        /// <summary>
        /// The timestamp of the operation
        /// </summary>
        public DateTime Timestamp { get; set; }
    }
    
    /// <summary>
    /// Types of operations
    /// </summary>
    private enum OperationType
    {
        /// <summary>
        /// File creation operation
        /// </summary>
        FileCreation,
        
        /// <summary>
        /// File modification operation
        /// </summary>
        FileModification,
        
        /// <summary>
        /// Task completion operation
        /// </summary>
        TaskCompletion,
        
        /// <summary>
        /// LLM operation
        /// </summary>
        LlmOperation
    }
}
