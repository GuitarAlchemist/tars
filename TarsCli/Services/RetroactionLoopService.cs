using Microsoft.FSharp.Core;
using Microsoft.Extensions.Logging;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.FSharp.Control;
using TarsEngine.SelfImprovement;

namespace TarsCli.Services;

/// <summary>
/// Service for managing the retroaction loop
/// </summary>
public class RetroactionLoopService(ILogger<RetroactionLoopService> logger, ConsoleService consoleService)
{
    /// <summary>
    /// Runs the retroaction loop
    /// </summary>
    /// <returns>True if the retroaction loop ran successfully</returns>
    public async Task<bool> RunRetroactionLoopAsync()
    {
        try
        {
            consoleService.WriteHeader("TARS Retroaction Loop");
            consoleService.WriteInfo("Running retroaction loop...");

            var result = await FSharpAsync.StartAsTask(
                RetroactionLoop.runRetroactionLoop(logger),
                FSharpOption<TaskCreationOptions>.None,
                FSharpOption<CancellationToken>.None);

            consoleService.WriteSuccess("Retroaction loop completed successfully");

            // Display some statistics
            var stats = RetroactionLoop.getStatistics(result);
            consoleService.WriteInfo($"Total patterns: {stats.TotalPatterns}");
            consoleService.WriteInfo($"Active patterns: {stats.ActivePatterns}");
            consoleService.WriteInfo($"Success rate: {stats.SuccessRate:P2}");
            consoleService.WriteInfo($"Average pattern score: {stats.AveragePatternScore:F2}");

            return true;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error running retroaction loop");
            consoleService.WriteError($"Error: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Creates a new pattern and adds it to the retroaction loop
    /// </summary>
    /// <param name="name">Pattern name</param>
    /// <param name="description">Pattern description</param>
    /// <param name="pattern">Regex pattern</param>
    /// <param name="replacement">Replacement string</param>
    /// <param name="context">Context (language)</param>
    /// <returns>True if the pattern was created successfully</returns>
    public async Task<bool> CreatePatternAsync(string name, string description, string pattern, string replacement, string context)
    {
        try
        {
            consoleService.WriteHeader("TARS Retroaction Loop - Create Pattern");
            consoleService.WriteInfo($"Creating pattern '{name}' for context '{context}'...");

            // Load the current state
            var state = await FSharpAsync.StartAsTask(
                RetroactionLoop.loadState(),
                FSharpOption<TaskCreationOptions>.None,
                FSharpOption<CancellationToken>.None);

            // Create the pattern
            var newPattern = RetroactionLoop.createPattern(name, description, pattern, replacement, context);

            // Add the pattern to the state
            var updatedState = RetroactionLoop.addPattern(state, newPattern);

            // Save the updated state
            var saveResult = await FSharpAsync.StartAsTask(
                RetroactionLoop.saveState(updatedState),
                FSharpOption<TaskCreationOptions>.None,
                FSharpOption<CancellationToken>.None);

            if (saveResult)
            {
                consoleService.WriteSuccess("Pattern created successfully");
                return true;
            }
            else
            {
                consoleService.WriteError("Error saving pattern");
                return false;
            }
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error creating pattern");
            consoleService.WriteError($"Error: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Records feedback on a pattern application
    /// </summary>
    /// <param name="patternId">Pattern ID</param>
    /// <param name="value">Feedback value (-1.0 to 1.0)</param>
    /// <param name="context">Context (language)</param>
    /// <returns>True if the feedback was recorded successfully</returns>
    public async Task<bool> RecordFeedbackAsync(string patternId, double value, string context)
    {
        try
        {
            consoleService.WriteHeader("TARS Retroaction Loop - Record Feedback");
            consoleService.WriteInfo($"Recording feedback for pattern '{patternId}'...");

            // Load the current state
            var state = await FSharpAsync.StartAsTask(
                RetroactionLoop.loadState(),
                FSharpOption<TaskCreationOptions>.None,
                FSharpOption<CancellationToken>.None);

            // Create the event
            var newEvent = RetroactionLoop.createEvent(patternId, "Feedback", value, context);

            // Add the event to the state
            var updatedState = RetroactionLoop.recordEvent(state, newEvent);

            // Save the updated state
            var saveResult = await FSharpAsync.StartAsTask(
                RetroactionLoop.saveState(updatedState),
                FSharpOption<TaskCreationOptions>.None,
                FSharpOption<CancellationToken>.None);

            if (saveResult)
            {
                consoleService.WriteSuccess("Feedback recorded successfully");
                return true;
            }
            else
            {
                consoleService.WriteError("Error saving feedback");
                return false;
            }
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error recording feedback");
            consoleService.WriteError($"Error: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Applies patterns to a code snippet
    /// </summary>
    /// <param name="code">Code snippet</param>
    /// <param name="context">Context (language)</param>
    /// <returns>The improved code and a list of applied patterns</returns>
    public async Task<(string ImprovedCode, List<string> AppliedPatterns)> ApplyPatternsAsync(string code, string context)
    {
        try
        {
            consoleService.WriteInfo($"Applying patterns for context '{context}'...");

            // Load the current state
            var state = await FSharpAsync.StartAsTask(
                RetroactionLoop.loadState(),
                FSharpOption<TaskCreationOptions>.None,
                FSharpOption<CancellationToken>.None);

            // Apply patterns
            var (improvedCode, results) = RetroactionLoop.applyPatterns(state, context, code);

            // Convert results to a list of pattern IDs
            var appliedPatterns = results
                .Where(r => r.Success && r.BeforeCode != r.AfterCode)
                .Select(r => r.PatternId)
                .ToList();

            // Update the state with the results
            var updatedState = results.Aggregate(state, RetroactionLoop.recordPatternApplication);

            // Save the updated state
            await FSharpAsync.StartAsTask(
                RetroactionLoop.saveState(updatedState),
                FSharpOption<TaskCreationOptions>.None,
                FSharpOption<CancellationToken>.None);

            return (improvedCode, appliedPatterns);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error applying patterns");
            return (code, new List<string>());
        }
    }

    /// <summary>
    /// Gets statistics about the retroaction loop
    /// </summary>
    /// <returns>Statistics about the retroaction loop</returns>
    public async Task<dynamic> GetStatisticsAsync()
    {
        try
        {
            // Load the current state
            var state = await FSharpAsync.StartAsTask(
                RetroactionLoop.loadState(),
                FSharpOption<TaskCreationOptions>.None,
                FSharpOption<CancellationToken>.None);

            // Get statistics
            var stats = RetroactionLoop.getStatistics(state);
            return stats;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error getting statistics");
            return null;
        }
    }

    /// <summary>
    /// Gets the top patterns for a given context
    /// </summary>
    /// <param name="context">Context (language)</param>
    /// <param name="count">Number of patterns to return</param>
    /// <returns>List of top patterns</returns>
    public async Task<List<ImprovementPattern>> GetTopPatternsAsync(string context, int count)
    {
        try
        {
            // Load the current state
            var state = await FSharpAsync.StartAsTask(
                RetroactionLoop.loadState(),
                FSharpOption<TaskCreationOptions>.None,
                FSharpOption<CancellationToken>.None);

            // Get top patterns
            var patterns = RetroactionLoop.getTopPatterns(state, context, count);
            return patterns.ToList();
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error getting top patterns");
            return [];
        }
    }
}