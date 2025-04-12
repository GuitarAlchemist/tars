using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsApp.Services.Interfaces;
using TarsApp.ViewModels;
using TarsEngine.Models;
using IEngineExecutionService = TarsEngine.Services.Interfaces.IExecutionService;

namespace TarsApp.Services
{
    public class ExecutionPlannerService : IExecutionPlannerService
    {
        private readonly ILogger<ExecutionPlannerService> _logger;
        private readonly ViewModelFactory _viewModelFactory;
        private readonly IEngineExecutionService _executionService;

        public ExecutionPlannerService(
            ILogger<ExecutionPlannerService> logger,
            ViewModelFactory viewModelFactory,
            IEngineExecutionService executionService)
        {
            _logger = logger;
            _viewModelFactory = viewModelFactory;
            _executionService = executionService;
        }

        public async Task<List<ExecutionViewModel>> GetExecutionPlansAsync()
        {
            try
            {
                var executionPlans = await _executionService.GetExecutionPlansAsync();
                return _viewModelFactory.CreateExecutionViewModels(executionPlans);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving execution plans");
                return new List<ExecutionViewModel>();
            }
        }

        public async Task<ExecutionViewModel> GetExecutionPlanAsync(string id)
        {
            try
            {
                var executionPlan = await _executionService.GetExecutionPlanAsync(id);
                return _viewModelFactory.CreateExecutionViewModel(executionPlan);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving execution plan {Id}", id);
                return null;
            }
        }

        public async Task<ExecutionViewModel> CreateExecutionPlanAsync(string name, string description, List<string> tags)
        {
            try
            {
                var executionPlan = await _executionService.CreateExecutionPlanAsync(name, description, tags);
                return _viewModelFactory.CreateExecutionViewModel(executionPlan);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating execution plan {Name}", name);
                return null;
            }
        }

        public async Task<ExecutionViewModel> StartExecutionAsync(string id)
        {
            try
            {
                var executionPlan = await _executionService.StartExecutionAsync(id);
                return _viewModelFactory.CreateExecutionViewModel(executionPlan);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error starting execution plan {Id}", id);
                return null;
            }
        }

        public async Task<ExecutionViewModel> StopExecutionAsync(string id)
        {
            try
            {
                var executionPlan = await _executionService.StopExecutionAsync(id);
                return _viewModelFactory.CreateExecutionViewModel(executionPlan);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error stopping execution plan {Id}", id);
                return null;
            }
        }

        public async Task<bool> DeleteExecutionAsync(string id)
        {
            try
            {
                return await _executionService.DeleteExecutionPlanAsync(id);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting execution plan {Id}", id);
                return false;
            }
        }

        public async Task<List<LogEntryViewModel>> GetExecutionLogsAsync(string id)
        {
            try
            {
                var logs = await _executionService.GetExecutionLogsAsync(id);
                return logs.Select(LogEntryViewModel.FromLogEntry).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving logs for execution plan {Id}", id);
                return new List<LogEntryViewModel>();
            }
        }
    }
}
