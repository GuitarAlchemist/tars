using TarsApp.ViewModels;

namespace TarsApp.Services.Interfaces;

public interface IExecutionPlannerService
{
    Task<List<ExecutionViewModel>> GetExecutionPlansAsync();
    Task<ExecutionViewModel> GetExecutionPlanAsync(string id);
    Task<ExecutionViewModel> CreateExecutionPlanAsync(string name, string description, List<string> tags);
    Task<ExecutionViewModel> StartExecutionAsync(string id);
    Task<ExecutionViewModel> StopExecutionAsync(string id);
    Task<bool> DeleteExecutionAsync(string id);
    Task<List<LogEntryViewModel>> GetExecutionLogsAsync(string id);
}