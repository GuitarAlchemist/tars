using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Orchestrator for the Improvement Generation System
/// </summary>
public class ImprovementGenerationOrchestrator
{
    private readonly ILogger<ImprovementGenerationOrchestrator> _logger;
    private readonly ICodeAnalyzerService _codeAnalyzer;
    private readonly IPatternMatcherService _patternMatcher;
    private readonly IMetascriptGeneratorService _metascriptGenerator;
    private readonly IImprovementPrioritizerService _improvementPrioritizer;
    private readonly IProgressReporter _progressReporter;

    /// <summary>
    /// Initializes a new instance of the <see cref="ImprovementGenerationOrchestrator"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="codeAnalyzer">The code analyzer service</param>
    /// <param name="patternMatcher">The pattern matcher service</param>
    /// <param name="metascriptGenerator">The metascript generator service</param>
    /// <param name="improvementPrioritizer">The improvement prioritizer service</param>
    /// <param name="progressReporter">The progress reporter</param>
    public ImprovementGenerationOrchestrator(
        ILogger<ImprovementGenerationOrchestrator> logger,
        ICodeAnalyzerService codeAnalyzer,
        IPatternMatcherService patternMatcher,
        IMetascriptGeneratorService metascriptGenerator,
        IImprovementPrioritizerService improvementPrioritizer,
        IProgressReporter progressReporter)
    {
        _logger = logger;
        _codeAnalyzer = codeAnalyzer;
        _patternMatcher = patternMatcher;
        _metascriptGenerator = metascriptGenerator;
        _improvementPrioritizer = improvementPrioritizer;
        _progressReporter = progressReporter;
    }

    /// <summary>
    /// Runs the end-to-end improvement generation workflow
    /// </summary>
    /// <param name="path">The path to analyze</param>
    /// <param name="options">Optional workflow options</param>
    /// <returns>The list of prioritized improvements</returns>
    public async Task<List<PrioritizedImprovement>> RunWorkflowAsync(string path, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Starting improvement generation workflow for path: {Path}", path);
            _progressReporter.ReportProgress("Starting improvement generation workflow", 0);

            // Parse options
            var recursive = ParseOption(options, "Recursive", true);
            var filePattern = ParseOption(options, "FilePattern", "*.cs;*.fs");
            var maxImprovements = ParseOption(options, "MaxImprovements", 10);
            var executeImprovements = ParseOption(options, "ExecuteImprovements", false);
            var dryRun = ParseOption(options, "DryRun", true);

            // Step 1: Analyze code
            _progressReporter.ReportProgress("Analyzing code", 10);
            var analysisResults = await AnalyzeCodeAsync(path, recursive, filePattern, options);
            _progressReporter.ReportProgress("Code analysis completed", 25);

            // Step 2: Match patterns
            _progressReporter.ReportProgress("Matching patterns", 30);
            var patternMatches = await MatchPatternsAsync(path, recursive, filePattern, options);
            _progressReporter.ReportProgress("Pattern matching completed", 50);

            // Step 3: Generate metascripts
            _progressReporter.ReportProgress("Generating metascripts", 60);
            var metascripts = await GenerateMetascriptsAsync(patternMatches, options);
            _progressReporter.ReportProgress("Metascript generation completed", 75);

            // Step 4: Prioritize improvements
            _progressReporter.ReportProgress("Prioritizing improvements", 80);
            var improvements = await PrioritizeImprovementsAsync(metascripts, options);
            _progressReporter.ReportProgress("Improvement prioritization completed", 90);

            // Step 5: Execute improvements (if requested)
            if (executeImprovements)
            {
                _progressReporter.ReportProgress("Executing improvements", 95);
                await ExecuteImprovementsAsync(improvements, maxImprovements, dryRun, options);
            }

            _progressReporter.ReportProgress("Improvement generation workflow completed", 100);
            _logger.LogInformation("Improvement generation workflow completed");

            return improvements;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running improvement generation workflow");
            _progressReporter.ReportError("Error running improvement generation workflow", ex);
            throw;
        }
    }

    /// <summary>
    /// Analyzes code for improvement opportunities
    /// </summary>
    /// <param name="path">The path to analyze</param>
    /// <param name="recursive">Whether to analyze directories recursively</param>
    /// <param name="filePattern">The file pattern to match</param>
    /// <param name="options">Optional analysis options</param>
    /// <returns>The analysis results</returns>
    public async Task<Dictionary<string, List<CodeAnalysisResult>>> AnalyzeCodeAsync(
        string path, 
        bool recursive, 
        string filePattern, 
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Analyzing code at path: {Path}", path);
            _progressReporter.ReportProgress("Analyzing code", 0);

            Dictionary<string, List<CodeAnalysisResult>> results;

            if (File.Exists(path))
            {
                // Analyze single file
                _progressReporter.ReportProgress($"Analyzing file: {Path.GetFileName(path)}", 50);
                var fileResults = await _codeAnalyzer.AnalyzeFileAsync(path, options);
                results = new Dictionary<string, List<CodeAnalysisResult>> { { path, fileResults } };
            }
            else if (Directory.Exists(path))
            {
                // Analyze directory
                _progressReporter.ReportProgress($"Analyzing directory: {path}", 10);
                results = await _codeAnalyzer.AnalyzeDirectoryAsync(path, recursive, filePattern, options);
            }
            else
            {
                throw new FileNotFoundException($"Path not found: {path}");
            }

            // Count total issues
            var totalIssues = results.Values.Sum(list => list.Count);
            _logger.LogInformation("Code analysis completed. Found {IssueCount} issues in {FileCount} files", 
                totalIssues, results.Count);
            _progressReporter.ReportProgress($"Code analysis completed. Found {totalIssues} issues in {results.Count} files", 100);

            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing code");
            _progressReporter.ReportError("Error analyzing code", ex);
            throw;
        }
    }

    /// <summary>
    /// Matches patterns in code
    /// </summary>
    /// <param name="path">The path to match patterns in</param>
    /// <param name="recursive">Whether to match patterns in directories recursively</param>
    /// <param name="filePattern">The file pattern to match</param>
    /// <param name="options">Optional matching options</param>
    /// <returns>The pattern matches</returns>
    public async Task<List<PatternMatch>> MatchPatternsAsync(
        string path, 
        bool recursive, 
        string filePattern, 
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Matching patterns in code at path: {Path}", path);
            _progressReporter.ReportProgress("Matching patterns", 0);

            Dictionary<string, List<PatternMatch>> results;

            if (File.Exists(path))
            {
                // Match patterns in single file
                _progressReporter.ReportProgress($"Matching patterns in file: {Path.GetFileName(path)}", 50);
                var fileMatches = await _patternMatcher.FindPatternsInFileAsync(path, options);
                results = new Dictionary<string, List<PatternMatch>> { { path, fileMatches } };
            }
            else if (Directory.Exists(path))
            {
                // Match patterns in directory
                _progressReporter.ReportProgress($"Matching patterns in directory: {path}", 10);
                results = await _patternMatcher.FindPatternsInDirectoryAsync(path, recursive, filePattern, options);
            }
            else
            {
                throw new FileNotFoundException($"Path not found: {path}");
            }

            // Flatten results
            var matches = results.Values.SelectMany(list => list).ToList();

            // Filter by confidence threshold
            var confidenceThreshold = ParseOption(options, "ConfidenceThreshold", 0.7);
            matches = matches.Where(m => m.Confidence >= confidenceThreshold).ToList();

            _logger.LogInformation("Pattern matching completed. Found {MatchCount} matches in {FileCount} files", 
                matches.Count, results.Count);
            _progressReporter.ReportProgress($"Pattern matching completed. Found {matches.Count} matches in {results.Count} files", 100);

            return matches;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error matching patterns");
            _progressReporter.ReportError("Error matching patterns", ex);
            throw;
        }
    }

    /// <summary>
    /// Generates metascripts from pattern matches
    /// </summary>
    /// <param name="patternMatches">The pattern matches</param>
    /// <param name="options">Optional generation options</param>
    /// <returns>The generated metascripts</returns>
    public async Task<List<GeneratedMetascript>> GenerateMetascriptsAsync(
        List<PatternMatch> patternMatches, 
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Generating metascripts from {MatchCount} pattern matches", patternMatches.Count);
            _progressReporter.ReportProgress("Generating metascripts", 0);

            var metascripts = new List<GeneratedMetascript>();
            var total = patternMatches.Count;
            var current = 0;

            foreach (var match in patternMatches)
            {
                try
                {
                    current++;
                    var progress = (int)((double)current / total * 100);
                    _progressReporter.ReportProgress($"Generating metascript for pattern: {match.PatternName} ({current}/{total})", progress);

                    var metascript = await _metascriptGenerator.GenerateMetascriptAsync(match, options);
                    metascripts.Add(metascript);

                    _logger.LogInformation("Generated metascript: {MetascriptName} ({MetascriptId})", 
                        metascript.Name, metascript.Id);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error generating metascript for pattern: {PatternName} ({PatternId})", 
                        match.PatternName, match.PatternId);
                    _progressReporter.ReportWarning($"Error generating metascript for pattern: {match.PatternName}", ex);
                }
            }

            _logger.LogInformation("Metascript generation completed. Generated {MetascriptCount} metascripts", metascripts.Count);
            _progressReporter.ReportProgress($"Metascript generation completed. Generated {metascripts.Count} metascripts", 100);

            return metascripts;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating metascripts");
            _progressReporter.ReportError("Error generating metascripts", ex);
            throw;
        }
    }

    /// <summary>
    /// Prioritizes improvements from metascripts
    /// </summary>
    /// <param name="metascripts">The metascripts</param>
    /// <param name="options">Optional prioritization options</param>
    /// <returns>The prioritized improvements</returns>
    public async Task<List<PrioritizedImprovement>> PrioritizeImprovementsAsync(
        List<GeneratedMetascript> metascripts, 
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Prioritizing improvements from {MetascriptCount} metascripts", metascripts.Count);
            _progressReporter.ReportProgress("Prioritizing improvements", 0);

            var improvements = new List<PrioritizedImprovement>();
            var total = metascripts.Count;
            var current = 0;

            foreach (var metascript in metascripts)
            {
                try
                {
                    current++;
                    var progress = (int)((double)current / total * 100);
                    _progressReporter.ReportProgress($"Creating improvement for metascript: {metascript.Name} ({current}/{total})", progress);

                    var improvement = await _improvementPrioritizer.CreateImprovementFromMetascriptAsync(metascript, options);
                    improvements.Add(improvement);

                    _logger.LogInformation("Created improvement: {ImprovementName} ({ImprovementId})", 
                        improvement.Name, improvement.Id);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error creating improvement for metascript: {MetascriptName} ({MetascriptId})", 
                        metascript.Name, metascript.Id);
                    _progressReporter.ReportWarning($"Error creating improvement for metascript: {metascript.Name}", ex);
                }
            }

            // Prioritize improvements
            _progressReporter.ReportProgress("Prioritizing improvements", 50);
            improvements = await _improvementPrioritizer.PrioritizeImprovementsAsync(improvements, options);

            // Sort by priority score
            improvements = improvements.OrderByDescending(i => i.PriorityScore).ToList();

            _logger.LogInformation("Improvement prioritization completed. Prioritized {ImprovementCount} improvements", improvements.Count);
            _progressReporter.ReportProgress($"Improvement prioritization completed. Prioritized {improvements.Count} improvements", 100);

            return improvements;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error prioritizing improvements");
            _progressReporter.ReportError("Error prioritizing improvements", ex);
            throw;
        }
    }

    /// <summary>
    /// Executes improvements
    /// </summary>
    /// <param name="improvements">The improvements to execute</param>
    /// <param name="maxImprovements">The maximum number of improvements to execute</param>
    /// <param name="dryRun">Whether to perform a dry run without making changes</param>
    /// <param name="options">Optional execution options</param>
    /// <returns>The execution results</returns>
    public async Task<List<MetascriptExecutionResult>> ExecuteImprovementsAsync(
        List<PrioritizedImprovement> improvements, 
        int maxImprovements, 
        bool dryRun, 
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Executing improvements");
            _progressReporter.ReportProgress("Executing improvements", 0);

            // Get next improvements to execute
            var nextImprovements = await _improvementPrioritizer.GetNextImprovementsAsync(maxImprovements, options);
            var results = new List<MetascriptExecutionResult>();
            var total = nextImprovements.Count;
            var current = 0;

            foreach (var improvement in nextImprovements)
            {
                try
                {
                    current++;
                    var progress = (int)((double)current / total * 100);
                    _progressReporter.ReportProgress($"Executing improvement: {improvement.Name} ({current}/{total})", progress);

                    // Get metascript
                    if (string.IsNullOrEmpty(improvement.MetascriptId))
                    {
                        _logger.LogWarning("Improvement has no associated metascript: {ImprovementId}", improvement.Id);
                        _progressReporter.ReportWarning($"Improvement has no associated metascript: {improvement.Name}", null);
                        continue;
                    }

                    var metascript = await _metascriptGenerator.GetMetascriptAsync(improvement.MetascriptId);
                    if (metascript == null)
                    {
                        _logger.LogWarning("Metascript not found: {MetascriptId}", improvement.MetascriptId);
                        _progressReporter.ReportWarning($"Metascript not found: {improvement.MetascriptId}", null);
                        continue;
                    }

                    // Execute metascript
                    var executionOptions = new Dictionary<string, string>(options ?? new Dictionary<string, string>());
                    if (dryRun)
                    {
                        executionOptions["DryRun"] = "true";
                    }

                    var result = await _metascriptGenerator.ExecuteMetascriptAsync(metascript, executionOptions);
                    results.Add(result);

                    // Update improvement status
                    if (result.IsSuccessful)
                    {
                        improvement.Status = dryRun ? ImprovementStatus.Pending : ImprovementStatus.Completed;
                        await _improvementPrioritizer.UpdateImprovementAsync(improvement);
                    }

                    _logger.LogInformation("Executed improvement: {ImprovementName} ({ImprovementId}), Status: {Status}", 
                        improvement.Name, improvement.Id, result.Status);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error executing improvement: {ImprovementName} ({ImprovementId})", 
                        improvement.Name, improvement.Id);
                    _progressReporter.ReportWarning($"Error executing improvement: {improvement.Name}", ex);
                }
            }

            _logger.LogInformation("Improvement execution completed. Executed {ImprovementCount} improvements", results.Count);
            _progressReporter.ReportProgress($"Improvement execution completed. Executed {results.Count} improvements", 100);

            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing improvements");
            _progressReporter.ReportError("Error executing improvements", ex);
            throw;
        }
    }

    private T ParseOption<T>(Dictionary<string, string>? options, string key, T defaultValue)
    {
        if (options == null || !options.TryGetValue(key, out var value))
        {
            return defaultValue;
        }

        try
        {
            return (T)Convert.ChangeType(value, typeof(T));
        }
        catch
        {
            return defaultValue;
        }
    }
}
