using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsEngine.SelfImprovement
{
    /// <summary>
    /// Service for updating the TODOs file
    /// </summary>
    public class TodosUpdater
    {
        private readonly ILogger<TodosUpdater> _logger;
        private readonly string _todosFilePath;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="TodosUpdater"/> class
        /// </summary>
        /// <param name="logger">The logger</param>
        /// <param name="todosFilePath">The path to the TODOs file</param>
        public TodosUpdater(ILogger<TodosUpdater> logger, string todosFilePath)
        {
            _logger = logger;
            _todosFilePath = todosFilePath;
        }
        
        /// <summary>
        /// Marks a task as completed in the TODOs file
        /// </summary>
        /// <param name="taskPattern">The pattern to match the task</param>
        /// <returns>A task representing the asynchronous operation</returns>
        public async Task MarkTaskAsCompletedAsync(string taskPattern)
        {
            _logger.LogInformation("Marking task as completed: {TaskPattern}", taskPattern);
            
            try
            {
                // Read the TODOs file
                var content = await File.ReadAllTextAsync(_todosFilePath);
                
                // Find the task and mark it as completed
                var regex = new Regex($"(\\s*[*-]\\s+)({Regex.Escape(taskPattern)})(?!\\s+\\u2705)");
                var updatedContent = regex.Replace(content, "$1$2 \u2705");
                
                // Write the updated content back to the file
                await File.WriteAllTextAsync(_todosFilePath, updatedContent);
                
                _logger.LogInformation("Task marked as completed successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error marking task as completed");
                throw;
            }
        }
        
        /// <summary>
        /// Adds subtasks to a task in the TODOs file
        /// </summary>
        /// <param name="parentTaskPattern">The pattern to match the parent task</param>
        /// <param name="subtasks">The subtasks to add</param>
        /// <returns>A task representing the asynchronous operation</returns>
        public async Task AddSubtasksAsync(string parentTaskPattern, IEnumerable<string> subtasks)
        {
            _logger.LogInformation("Adding subtasks to task: {ParentTaskPattern}", parentTaskPattern);
            
            try
            {
                // Read the TODOs file
                var content = await File.ReadAllTextAsync(_todosFilePath);
                var lines = content.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None).ToList();
                
                // Find the parent task
                var parentTaskIndex = -1;
                var parentTaskIndentation = "";
                var regex = new Regex($"(\\s*)([*-]\\s+)({Regex.Escape(parentTaskPattern)})");
                
                for (int i = 0; i < lines.Count; i++)
                {
                    var match = regex.Match(lines[i]);
                    if (match.Success)
                    {
                        parentTaskIndex = i;
                        parentTaskIndentation = match.Groups[1].Value;
                        break;
                    }
                }
                
                if (parentTaskIndex == -1)
                {
                    _logger.LogWarning("Parent task not found: {ParentTaskPattern}", parentTaskPattern);
                    return;
                }
                
                // Find the next task at the same level or higher
                var insertIndex = parentTaskIndex + 1;
                var parentIndentationLevel = parentTaskIndentation.Length;
                
                while (insertIndex < lines.Count)
                {
                    var line = lines[insertIndex];
                    var indentationMatch = Regex.Match(line, "^(\\s*)([*-]\\s+)");
                    
                    if (indentationMatch.Success)
                    {
                        var indentationLevel = indentationMatch.Groups[1].Value.Length;
                        
                        if (indentationLevel <= parentIndentationLevel)
                        {
                            break;
                        }
                    }
                    
                    insertIndex++;
                }
                
                // Add the subtasks
                var subtaskIndentation = parentTaskIndentation + "  ";
                var subtaskLines = subtasks.Select(subtask => $"{subtaskIndentation}* {subtask}");
                lines.InsertRange(insertIndex, subtaskLines);
                
                // Write the updated content back to the file
                var updatedContent = string.Join(Environment.NewLine, lines);
                await File.WriteAllTextAsync(_todosFilePath, updatedContent);
                
                _logger.LogInformation("Subtasks added successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error adding subtasks");
                throw;
            }
        }
        
        /// <summary>
        /// Updates the progress percentage in the TODOs file
        /// </summary>
        /// <param name="sectionPattern">The pattern to match the section</param>
        /// <param name="newPercentage">The new percentage</param>
        /// <returns>A task representing the asynchronous operation</returns>
        public async Task UpdateProgressPercentageAsync(string sectionPattern, int newPercentage)
        {
            _logger.LogInformation("Updating progress percentage for section: {SectionPattern}", sectionPattern);
            
            try
            {
                // Read the TODOs file
                var content = await File.ReadAllTextAsync(_todosFilePath);
                
                // Find the section and update the percentage
                var regex = new Regex($"({Regex.Escape(sectionPattern)}\\s+\\(.*?\\))\\s+\\-\\s+(\\d+)%");
                var updatedContent = regex.Replace(content, $"$1 - {newPercentage}%");
                
                // Write the updated content back to the file
                await File.WriteAllTextAsync(_todosFilePath, updatedContent);
                
                _logger.LogInformation("Progress percentage updated successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating progress percentage");
                throw;
            }
        }
        
        /// <summary>
        /// Calculates the completion percentage for a section
        /// </summary>
        /// <param name="sectionPattern">The pattern to match the section</param>
        /// <returns>The completion percentage</returns>
        public async Task<int> CalculateCompletionPercentageAsync(string sectionPattern)
        {
            _logger.LogInformation("Calculating completion percentage for section: {SectionPattern}", sectionPattern);
            
            try
            {
                // Read the TODOs file
                var content = await File.ReadAllTextAsync(_todosFilePath);
                var lines = content.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);
                
                // Find the section
                var sectionStartIndex = -1;
                var sectionEndIndex = -1;
                var sectionRegex = new Regex($"^\\s*{Regex.Escape(sectionPattern)}");
                
                for (int i = 0; i < lines.Length; i++)
                {
                    if (sectionRegex.IsMatch(lines[i]))
                    {
                        sectionStartIndex = i;
                        break;
                    }
                }
                
                if (sectionStartIndex == -1)
                {
                    _logger.LogWarning("Section not found: {SectionPattern}", sectionPattern);
                    return 0;
                }
                
                // Find the end of the section
                var sectionIndentation = Regex.Match(lines[sectionStartIndex], "^(\\s*)").Groups[1].Value.Length;
                
                for (int i = sectionStartIndex + 1; i < lines.Length; i++)
                {
                    var line = lines[i];
                    
                    if (line.Trim().Length == 0)
                    {
                        continue;
                    }
                    
                    var indentation = Regex.Match(line, "^(\\s*)").Groups[1].Value.Length;
                    
                    if (indentation <= sectionIndentation && Regex.IsMatch(line, "^\\s*[#*-]"))
                    {
                        sectionEndIndex = i - 1;
                        break;
                    }
                }
                
                if (sectionEndIndex == -1)
                {
                    sectionEndIndex = lines.Length - 1;
                }
                
                // Count completed and total tasks
                var taskRegex = new Regex("^\\s*[*-]\\s+");
                var completedTaskRegex = new Regex("^\\s*[*-]\\s+.*\\u2705");
                var totalTasks = 0;
                var completedTasks = 0;
                
                for (int i = sectionStartIndex + 1; i <= sectionEndIndex; i++)
                {
                    var line = lines[i];
                    
                    if (taskRegex.IsMatch(line))
                    {
                        totalTasks++;
                        
                        if (completedTaskRegex.IsMatch(line))
                        {
                            completedTasks++;
                        }
                    }
                }
                
                // Calculate the percentage
                var percentage = totalTasks > 0 ? (int)Math.Round((double)completedTasks / totalTasks * 100) : 0;
                
                _logger.LogInformation("Completion percentage calculated: {Percentage}%", percentage);
                
                return percentage;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error calculating completion percentage");
                throw;
            }
        }
        
        /// <summary>
        /// Adds a new task to the TODOs file
        /// </summary>
        /// <param name="sectionPattern">The pattern to match the section</param>
        /// <param name="task">The task to add</param>
        /// <returns>A task representing the asynchronous operation</returns>
        public async Task AddTaskAsync(string sectionPattern, string task)
        {
            _logger.LogInformation("Adding task to section: {SectionPattern}", sectionPattern);
            
            try
            {
                // Read the TODOs file
                var content = await File.ReadAllTextAsync(_todosFilePath);
                var lines = content.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None).ToList();
                
                // Find the section
                var sectionStartIndex = -1;
                var sectionEndIndex = -1;
                var sectionRegex = new Regex($"^\\s*{Regex.Escape(sectionPattern)}");
                
                for (int i = 0; i < lines.Count; i++)
                {
                    if (sectionRegex.IsMatch(lines[i]))
                    {
                        sectionStartIndex = i;
                        break;
                    }
                }
                
                if (sectionStartIndex == -1)
                {
                    _logger.LogWarning("Section not found: {SectionPattern}", sectionPattern);
                    return;
                }
                
                // Find the end of the section
                var sectionIndentation = Regex.Match(lines[sectionStartIndex], "^(\\s*)").Groups[1].Value.Length;
                
                for (int i = sectionStartIndex + 1; i < lines.Count; i++)
                {
                    var line = lines[i];
                    
                    if (line.Trim().Length == 0)
                    {
                        continue;
                    }
                    
                    var indentation = Regex.Match(line, "^(\\s*)").Groups[1].Value.Length;
                    
                    if (indentation <= sectionIndentation && Regex.IsMatch(line, "^\\s*[#*-]"))
                    {
                        sectionEndIndex = i - 1;
                        break;
                    }
                }
                
                if (sectionEndIndex == -1)
                {
                    sectionEndIndex = lines.Count - 1;
                }
                
                // Add the task
                var taskIndentation = new string(' ', sectionIndentation + 2);
                var taskLine = $"{taskIndentation}- {task}";
                lines.Insert(sectionEndIndex + 1, taskLine);
                
                // Write the updated content back to the file
                var updatedContent = string.Join(Environment.NewLine, lines);
                await File.WriteAllTextAsync(_todosFilePath, updatedContent);
                
                _logger.LogInformation("Task added successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error adding task");
                throw;
            }
        }
    }
}
