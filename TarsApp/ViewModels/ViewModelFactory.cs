using System;
using System.Collections.Generic;
using System.Linq;
using TarsEngine.Models;

namespace TarsApp.ViewModels
{
    public class ViewModelFactory
    {
        public ExecutionViewModel CreateExecutionViewModel(ExecutionPlan executionPlan)
        {
            if (executionPlan == null)
                return null;

            var viewModel = new ExecutionViewModel
            {
                Id = executionPlan.Id,
                Name = executionPlan.Name,
                Description = executionPlan.Description,
                Status = executionPlan.Status.ToString(),
                StartTime = executionPlan.Result?.StartedAt ?? DateTime.MinValue,
                EndTime = executionPlan.Result?.CompletedAt ?? DateTime.MinValue,
                Progress = executionPlan.Progress,
                Tags = executionPlan.Metadata.ContainsKey("Tags") ? executionPlan.Metadata["Tags"].Split(',').ToList() : new List<string>()
            };

            // Convert logs from context
            if (executionPlan.Context?.Logs != null)
            {
                viewModel.Logs = executionPlan.Context.Logs
                    .Select(log => new LogEntryViewModel
                    {
                        Timestamp = log.Timestamp,
                        LogLevel = LogEntryViewModel.ConvertLogLevel(log.Level),
                        Message = log.Message,
                        Source = log.Source
                    })
                    .ToList();
            }

            // Convert steps
            if (executionPlan.Steps != null)
            {
                viewModel.Steps = executionPlan.Steps
                    .Select(step => new ExecutionStepViewModel
                    {
                        Id = step.Id,
                        Name = step.Name,
                        Description = step.Description,
                        Status = step.Status.ToString(),
                        StartTime = step.StartedAt,
                        EndTime = step.CompletedAt,
                        Progress = CalculateStepProgress(step),
                        Logs = new List<LogEntryViewModel>() // Steps don't have logs directly in the new model
                    })
                    .ToList();
            }

            return viewModel;
        }

        public List<ExecutionViewModel> CreateExecutionViewModels(IEnumerable<ExecutionPlan> executionPlans)
        {
            if (executionPlans == null)
                return new List<ExecutionViewModel>();

            return executionPlans
                .Select(CreateExecutionViewModel)
                .Where(vm => vm != null)
                .ToList();
        }

        /// <summary>
        /// Calculates the progress of an execution step based on its status
        /// </summary>
        private double CalculateStepProgress(ExecutionStep step)
        {
            return step.Status switch
            {
                ExecutionStepStatus.Completed => 100,
                ExecutionStepStatus.Failed => 100,
                ExecutionStepStatus.Skipped => 100,
                ExecutionStepStatus.Cancelled => 100,
                ExecutionStepStatus.RolledBack => 100,
                ExecutionStepStatus.InProgress => 50, // Assuming 50% progress for in-progress steps
                _ => 0
            };
        }
    }
}
