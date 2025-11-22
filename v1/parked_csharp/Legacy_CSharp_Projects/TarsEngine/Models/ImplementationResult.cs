namespace TarsEngine.Models;

/// <summary>
/// Represents the result of implementing a task
/// </summary>
public class ImplementationResult
{
    /// <summary>
    /// The task description
    /// </summary>
    public string TaskDescription { get; set; } = string.Empty;
    
    /// <summary>
    /// The implementation plan
    /// </summary>
    public ImplementationPlan? Plan { get; set; }
    
    /// <summary>
    /// The list of step results
    /// </summary>
    public List<StepResult> StepResults { get; set; } = new();
    
    /// <summary>
    /// The test results
    /// </summary>
    public TestResults? TestResults { get; set; }
    
    /// <summary>
    /// Whether the implementation was successful
    /// </summary>
    public bool Success { get; set; } = true;
    
    /// <summary>
    /// The error message if the implementation failed
    /// </summary>
    public string ErrorMessage { get; set; } = string.Empty;
    
    /// <summary>
    /// The start time of the implementation
    /// </summary>
    public DateTime StartTime { get; set; }
    
    /// <summary>
    /// The end time of the implementation
    /// </summary>
    public DateTime EndTime { get; set; }
    
    /// <summary>
    /// The duration of the implementation
    /// </summary>
    public TimeSpan Duration { get; set; }
    
    /// <summary>
    /// Convert the implementation result to a string
    /// </summary>
    /// <returns>A string representation of the implementation result</returns>
    public override string ToString()
    {
        var result = new System.Text.StringBuilder();
        
        result.AppendLine($"# Implementation Result for: {TaskDescription}");
        result.AppendLine($"Status: {(Success ? "Success" : "Failed")}");
        if (!Success)
        {
            result.AppendLine($"Error: {ErrorMessage}");
        }
        result.AppendLine($"Duration: {Duration.TotalSeconds:F2} seconds");
        result.AppendLine();
        
        result.AppendLine("## Step Results");
        foreach (var stepResult in StepResults)
        {
            result.AppendLine($"### Step {stepResult.StepNumber}: {(stepResult.Success ? "Success" : "Failed")}");
            result.AppendLine($"File: {stepResult.FilePath}");
            result.AppendLine($"Duration: {stepResult.Duration.TotalSeconds:F2} seconds");
            if (!stepResult.Success)
            {
                result.AppendLine($"Error: {stepResult.ErrorMessage}");
            }
            result.AppendLine();
        }
        
        if (TestResults != null)
        {
            result.AppendLine("## Test Results");
            result.AppendLine($"Status: {(TestResults.Success ? "Success" : "Failed")}");
            if (!TestResults.Success)
            {
                result.AppendLine($"Error: {TestResults.ErrorMessage}");
            }
            result.AppendLine($"Duration: {TestResults.Duration.TotalSeconds:F2} seconds");
            result.AppendLine();
            
            result.AppendLine("### Tests");
            foreach (var testResult in TestResults.Tests)
            {
                result.AppendLine($"- {testResult.ComponentName}: {(testResult.Success ? "Passed" : "Failed")}");
                result.AppendLine($"  File: {testResult.TestFilePath}");
                if (!testResult.Success)
                {
                    result.AppendLine($"  Error: {testResult.ErrorMessage}");
                }
            }
        }
        
        return result.ToString();
    }
}
