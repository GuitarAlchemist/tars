namespace TarsApp.ViewModels;

public class ExecutionStepViewModel
{
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string Status { get; set; } = string.Empty;
    public DateTime? StartTime { get; set; }
    public DateTime? EndTime { get; set; }
    public double Progress { get; set; }
    public List<LogEntryViewModel> Logs { get; set; } = new();
}