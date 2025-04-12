using System;
using System.Collections.Generic;

namespace TarsApp.ViewModels
{
    public class ExecutionViewModel
    {
        public string Id { get; set; }
        public string Name { get; set; }
        public string Description { get; set; }
        public string Status { get; set; }
        public DateTime StartTime { get; set; }
        public DateTime? EndTime { get; set; }
        public double Progress { get; set; }
        public List<string> Tags { get; set; } = new();
        public List<LogEntryViewModel> Logs { get; set; } = new();
        public List<ExecutionStepViewModel> Steps { get; set; } = new();
    }
}
