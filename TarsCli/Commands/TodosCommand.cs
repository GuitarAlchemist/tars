using System.CommandLine.Invocation;
using Microsoft.Extensions.DependencyInjection;
using TarsEngine.SelfImprovement;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for managing TODOs
/// </summary>
public class TodosCommand : Command
{
    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// Initializes a new instance of the <see cref="TodosCommand"/> class
    /// </summary>
    /// <param name="serviceProvider">The service provider</param>
    public TodosCommand(IServiceProvider serviceProvider) : base("todos", "Manage TODOs")
    {
        _serviceProvider = serviceProvider;

        // Add subcommands
        AddCommand(new MarkTaskCompletedCommand(_serviceProvider));
        AddCommand(new AddSubtasksCommand(_serviceProvider));
        AddCommand(new UpdateProgressCommand(_serviceProvider));
        AddCommand(new AddTaskCommand(_serviceProvider));
    }

    /// <summary>
    /// Command for marking a task as completed
    /// </summary>
    private class MarkTaskCompletedCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        /// <summary>
        /// Initializes a new instance of the <see cref="MarkTaskCompletedCommand"/> class
        /// </summary>
        /// <param name="serviceProvider">The service provider</param>
        public MarkTaskCompletedCommand(IServiceProvider serviceProvider) : base("mark-completed", "Mark a task as completed")
        {
            _serviceProvider = serviceProvider;

            // Add options
            var taskOption = new Option<string>(
                "--task",
                description: "The task to mark as completed")
            {
                IsRequired = true
            };

            var todosFileOption = new Option<string>(
                "--todos-file",
                description: "The path to the TODOs file",
                getDefaultValue: () => Path.Combine(Directory.GetCurrentDirectory(), "Tars - TODOs.txt"));

            AddOption(taskOption);
            AddOption(todosFileOption);

            this.SetHandler((InvocationContext context) =>
            {
                var task = context.ParseResult.GetValueForOption(taskOption);
                var todosFile = context.ParseResult.GetValueForOption(todosFileOption);

                var logger = _serviceProvider.GetRequiredService<ILogger<TarsEngine.SelfImprovement.TodosUpdater>>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();
                var operationSummaryService = _serviceProvider.GetRequiredService<OperationSummaryService>();

                try
                {
                    consoleService.WriteHeader("Marking Task as Completed");
                    consoleService.WriteLine($"Task: {task}");
                    consoleService.WriteLine($"TODOs file: {todosFile}");
                    consoleService.WriteLine();

                    var todosUpdater = new TodosUpdater(logger, todosFile);
                    var markTask = todosUpdater.MarkTaskAsCompletedAsync(task);
                    markTask.Wait();

                    consoleService.WriteSuccess("Task marked as completed successfully");

                    // Record the operation
                    operationSummaryService.RecordTaskCompletion(task, $"Marked task as completed in {todosFile}");

                    // Save the summary
                    var summaryPath = operationSummaryService.SaveSummary();
                    if (!string.IsNullOrEmpty(summaryPath))
                    {
                        consoleService.WriteLine();
                        consoleService.WriteSuccess($"Operation summary saved to: {summaryPath}");
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error marking task as completed");
                    consoleService.WriteError($"Error marking task as completed: {ex.Message}");
                }
            });
        }
    }

    /// <summary>
    /// Command for adding subtasks to a task
    /// </summary>
    private class AddSubtasksCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        /// <summary>
        /// Initializes a new instance of the <see cref="AddSubtasksCommand"/> class
        /// </summary>
        /// <param name="serviceProvider">The service provider</param>
        public AddSubtasksCommand(IServiceProvider serviceProvider) : base("add-subtasks", "Add subtasks to a task")
        {
            _serviceProvider = serviceProvider;

            // Add options
            var parentTaskOption = new Option<string>(
                "--parent-task",
                description: "The parent task")
            {
                IsRequired = true
            };

            var subtasksOption = new Option<string[]>(
                "--subtasks",
                description: "The subtasks to add")
            {
                IsRequired = true,
                AllowMultipleArgumentsPerToken = true
            };

            var todosFileOption = new Option<string>(
                "--todos-file",
                description: "The path to the TODOs file",
                getDefaultValue: () => Path.Combine(Directory.GetCurrentDirectory(), "Tars - TODOs.txt"));

            AddOption(parentTaskOption);
            AddOption(subtasksOption);
            AddOption(todosFileOption);

            this.SetHandler((InvocationContext context) =>
            {
                var parentTask = context.ParseResult.GetValueForOption(parentTaskOption);
                var subtasks = context.ParseResult.GetValueForOption(subtasksOption);
                var todosFile = context.ParseResult.GetValueForOption(todosFileOption);

                var logger = _serviceProvider.GetRequiredService<ILogger<TarsEngine.SelfImprovement.TodosUpdater>>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteHeader("Adding Subtasks");
                    consoleService.WriteLine($"Parent task: {parentTask}");
                    consoleService.WriteLine($"TODOs file: {todosFile}");
                    consoleService.WriteLine();

                    consoleService.WriteLine("Subtasks:");
                    foreach (var subtask in subtasks)
                    {
                        consoleService.WriteLine($"  - {subtask}");
                    }
                    consoleService.WriteLine();

                    var todosUpdater = new TodosUpdater(logger, todosFile);
                    var addSubtasksTask = todosUpdater.AddSubtasksAsync(parentTask, subtasks);
                    addSubtasksTask.Wait();

                    consoleService.WriteSuccess("Subtasks added successfully");
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error adding subtasks");
                    consoleService.WriteError($"Error adding subtasks: {ex.Message}");
                }
            });
        }
    }

    /// <summary>
    /// Command for updating progress percentage
    /// </summary>
    private class UpdateProgressCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        /// <summary>
        /// Initializes a new instance of the <see cref="UpdateProgressCommand"/> class
        /// </summary>
        /// <param name="serviceProvider">The service provider</param>
        public UpdateProgressCommand(IServiceProvider serviceProvider) : base("update-progress", "Update progress percentage")
        {
            _serviceProvider = serviceProvider;

            // Add options
            var sectionOption = new Option<string>(
                "--section",
                description: "The section to update")
            {
                IsRequired = true
            };

            var percentageOption = new Option<int>(
                "--percentage",
                description: "The new percentage")
            {
                IsRequired = true
            };

            var todosFileOption = new Option<string>(
                "--todos-file",
                description: "The path to the TODOs file",
                getDefaultValue: () => Path.Combine(Directory.GetCurrentDirectory(), "Tars - TODOs.txt"));

            var calculateOption = new Option<bool>(
                "--calculate",
                description: "Calculate the percentage automatically",
                getDefaultValue: () => false);

            AddOption(sectionOption);
            AddOption(percentageOption);
            AddOption(todosFileOption);
            AddOption(calculateOption);

            this.SetHandler((InvocationContext context) =>
            {
                var section = context.ParseResult.GetValueForOption(sectionOption);
                var percentage = context.ParseResult.GetValueForOption(percentageOption);
                var todosFile = context.ParseResult.GetValueForOption(todosFileOption);
                var calculate = context.ParseResult.GetValueForOption(calculateOption);

                var logger = _serviceProvider.GetRequiredService<ILogger<TarsEngine.SelfImprovement.TodosUpdater>>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteHeader("Updating Progress");
                    consoleService.WriteLine($"Section: {section}");
                    consoleService.WriteLine($"TODOs file: {todosFile}");
                    consoleService.WriteLine();

                    var todosUpdater = new TodosUpdater(logger, todosFile);

                    if (calculate)
                    {
                        var calculateTask = todosUpdater.CalculateCompletionPercentageAsync(section);
                        calculateTask.Wait();
                        percentage = calculateTask.Result;
                        consoleService.WriteLine($"Calculated percentage: {percentage}%");
                    }
                    else
                    {
                        consoleService.WriteLine($"New percentage: {percentage}%");
                    }

                    var updateTask = todosUpdater.UpdateProgressPercentageAsync(section, percentage);
                    updateTask.Wait();

                    consoleService.WriteSuccess("Progress updated successfully");
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error updating progress");
                    consoleService.WriteError($"Error updating progress: {ex.Message}");
                }
            });
        }
    }

    /// <summary>
    /// Command for adding a task
    /// </summary>
    private class AddTaskCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        /// <summary>
        /// Initializes a new instance of the <see cref="AddTaskCommand"/> class
        /// </summary>
        /// <param name="serviceProvider">The service provider</param>
        public AddTaskCommand(IServiceProvider serviceProvider) : base("add-task", "Add a task")
        {
            _serviceProvider = serviceProvider;

            // Add options
            var sectionOption = new Option<string>(
                "--section",
                description: "The section to add the task to")
            {
                IsRequired = true
            };

            var taskOption = new Option<string>(
                "--task",
                description: "The task to add")
            {
                IsRequired = true
            };

            var todosFileOption = new Option<string>(
                "--todos-file",
                description: "The path to the TODOs file",
                getDefaultValue: () => Path.Combine(Directory.GetCurrentDirectory(), "Tars - TODOs.txt"));

            AddOption(sectionOption);
            AddOption(taskOption);
            AddOption(todosFileOption);

            this.SetHandler((InvocationContext context) =>
            {
                var section = context.ParseResult.GetValueForOption(sectionOption);
                var task = context.ParseResult.GetValueForOption(taskOption);
                var todosFile = context.ParseResult.GetValueForOption(todosFileOption);

                var logger = _serviceProvider.GetRequiredService<ILogger<TarsEngine.SelfImprovement.TodosUpdater>>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteHeader("Adding Task");
                    consoleService.WriteLine($"Section: {section}");
                    consoleService.WriteLine($"Task: {task}");
                    consoleService.WriteLine($"TODOs file: {todosFile}");
                    consoleService.WriteLine();

                    var todosUpdater = new TodosUpdater(logger, todosFile);
                    var addTask = todosUpdater.AddTaskAsync(section, task);
                    addTask.Wait();

                    consoleService.WriteSuccess("Task added successfully");
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error adding task");
                    consoleService.WriteError($"Error adding task: {ex.Message}");
                }
            });
        }
    }
}