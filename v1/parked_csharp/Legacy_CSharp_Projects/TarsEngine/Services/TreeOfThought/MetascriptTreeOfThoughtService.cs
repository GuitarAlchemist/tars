using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Compilation;

namespace TarsEngine.Services.TreeOfThought
{
    /// <summary>
    /// Service for Tree-of-Thought reasoning with metascripts.
    /// </summary>
    public class MetascriptTreeOfThoughtService
    {
        private readonly ILogger<MetascriptTreeOfThoughtService> _logger;
        private readonly FSharpScriptExecutor _scriptExecutor;
        private readonly string _fsharpModulePath;
        private readonly string _fsharpGenerationModulePath;
        private readonly string _fsharpValidationModulePath;
        private readonly string _fsharpExecutionModulePath;
        private readonly string _fsharpResultAnalysisModulePath;

        /// <summary>
        /// Initializes a new instance of the <see cref="MetascriptTreeOfThoughtService"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="scriptExecutor">The F# script executor.</param>
        public MetascriptTreeOfThoughtService(ILogger<MetascriptTreeOfThoughtService> logger, FSharpScriptExecutor scriptExecutor)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _scriptExecutor = scriptExecutor ?? throw new ArgumentNullException(nameof(scriptExecutor));
            
            // Set paths to F# modules
            _fsharpModulePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TarsEngine", "FSharp", "MetascriptToT.fs");
            _fsharpGenerationModulePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TarsEngine", "FSharp", "MetascriptGeneration.fs");
            _fsharpValidationModulePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TarsEngine", "FSharp", "MetascriptValidation.fs");
            _fsharpExecutionModulePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TarsEngine", "FSharp", "MetascriptExecution.fs");
            _fsharpResultAnalysisModulePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TarsEngine", "FSharp", "MetascriptResultAnalysis.fs");
            
            // Ensure the F# modules exist
            CheckModules();
        }

        private void CheckModules()
        {
            var modules = new Dictionary<string, string>
            {
                { "MetascriptToT", _fsharpModulePath },
                { "MetascriptGeneration", _fsharpGenerationModulePath },
                { "MetascriptValidation", _fsharpValidationModulePath },
                { "MetascriptExecution", _fsharpExecutionModulePath },
                { "MetascriptResultAnalysis", _fsharpResultAnalysisModulePath }
            };

            foreach (var module in modules)
            {
                if (!File.Exists(module.Value))
                {
                    _logger.LogWarning("F# module {Module} not found at {Path}", module.Key, module.Value);
                }
                else
                {
                    _logger.LogInformation("F# module {Module} found at {Path}", module.Key, module.Value);
                }
            }
        }

        /// <summary>
        /// Generates a metascript using Tree-of-Thought reasoning.
        /// </summary>
        /// <param name="templateContent">The template content.</param>
        /// <param name="templateValues">The template values.</param>
        /// <returns>The generated metascript and the thought tree as JSON.</returns>
        public async Task<(string Metascript, string ThoughtTreeJson)> GenerateMetascriptAsync(string templateContent, Dictionary<string, string> templateValues)
        {
            _logger.LogInformation("Generating metascript using Tree-of-Thought reasoning");
            
            try
            {
                // Convert template values to F# map
                var templateValuesMap = ConvertDictionaryToFSharpMap(templateValues);
                
                // Create the F# script
                var script = $@"
                    // Load the modules
                    #load ""{_fsharpModulePath}""
                    #load ""{_fsharpGenerationModulePath}""
                    
                    open TarsEngine.FSharp.MetascriptToT
                    open TarsEngine.FSharp.MetascriptGeneration
                    
                    // Create the template
                    let template = Templates.createTemplate ""Template"" @""{templateContent.Replace("\"", "\\\"")}""  ""Template description""
                    
                    // Create the template values
                    let templateValues = {templateValuesMap}
                    
                    // Generate the metascript
                    let (thoughtTree, metascript) = Generation.generateFromTemplate template templateValues
                    
                    // Return the results
                    let thoughtTreeJson = ThoughtTree.toJson thoughtTree
                    
                    // Return the results as a tuple
                    (metascript, thoughtTreeJson)
                ";
                
                // Execute the script
                var result = await _scriptExecutor.ExecuteScriptAsync(script);
                
                if (!result.Success)
                {
                    _logger.LogError("Error generating metascript: {Errors}", string.Join(Environment.NewLine, result.Errors));
                    throw new Exception($"Error generating metascript: {string.Join(Environment.NewLine, result.Errors)}");
                }
                
                // Parse the result
                // In a real implementation, this would use a proper parser
                var output = result.Output.Trim();
                var metascript = output.Substring(1, output.IndexOf("\",") - 1);
                var thoughtTreeJson = output.Substring(output.IndexOf("\",") + 2, output.Length - output.IndexOf("\",") - 3);
                
                return (metascript, thoughtTreeJson);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating metascript");
                throw;
            }
        }

        /// <summary>
        /// Validates a metascript using Tree-of-Thought reasoning.
        /// </summary>
        /// <param name="metascript">The metascript to validate.</param>
        /// <returns>The validation results and the thought tree as JSON.</returns>
        public async Task<(bool IsValid, List<string> Errors, List<string> Warnings, string ThoughtTreeJson)> ValidateMetascriptAsync(string metascript)
        {
            _logger.LogInformation("Validating metascript using Tree-of-Thought reasoning");
            
            try
            {
                // Create the F# script
                var script = $@"
                    // Load the modules
                    #load ""{_fsharpModulePath}""
                    #load ""{_fsharpValidationModulePath}""
                    
                    open TarsEngine.FSharp.MetascriptToT
                    open TarsEngine.FSharp.MetascriptValidation
                    
                    // Validate the metascript
                    let (thoughtTree, syntaxErrors, semanticErrors, syntaxCorrectionSuggestions, semanticCorrectionSuggestions) = 
                        Validation.validateMetascript @""{metascript.Replace("\"", "\\\"")}""
                    
                    // Determine if the metascript is valid
                    let isValid = List.isEmpty syntaxErrors && List.isEmpty semanticErrors
                    
                    // Convert errors to strings
                    let syntaxErrorStrings = 
                        syntaxErrors 
                        |> List.map (fun e -> sprintf ""Line %d, Column %d: %s"" e.LineNumber e.ColumnNumber e.Message)
                    
                    let semanticErrorStrings = 
                        semanticErrors 
                        |> List.map (fun e -> sprintf ""Line %d, Column %d: %s (%s)"" e.LineNumber e.ColumnNumber e.Message e.Context)
                    
                    // Combine all errors
                    let allErrors = syntaxErrorStrings @ semanticErrorStrings
                    
                    // Get warnings
                    let syntaxWarnings = 
                        syntaxErrors 
                        |> List.filter (fun e -> e.Severity = ""Warning"") 
                        |> List.map (fun e -> sprintf ""Line %d, Column %d: %s"" e.LineNumber e.ColumnNumber e.Message)
                    
                    let semanticWarnings = 
                        semanticErrors 
                        |> List.filter (fun e -> e.Severity = ""Warning"") 
                        |> List.map (fun e -> sprintf ""Line %d, Column %d: %s (%s)"" e.LineNumber e.ColumnNumber e.Message e.Context)
                    
                    // Combine all warnings
                    let allWarnings = syntaxWarnings @ semanticWarnings
                    
                    // Convert the thought tree to JSON
                    let thoughtTreeJson = ThoughtTree.toJson thoughtTree
                    
                    // Return the results as a tuple
                    (isValid, allErrors, allWarnings, thoughtTreeJson)
                ";
                
                // Execute the script
                var result = await _scriptExecutor.ExecuteScriptAsync(script);
                
                if (!result.Success)
                {
                    _logger.LogError("Error validating metascript: {Errors}", string.Join(Environment.NewLine, result.Errors));
                    throw new Exception($"Error validating metascript: {string.Join(Environment.NewLine, result.Errors)}");
                }
                
                // Parse the result
                // In a real implementation, this would use a proper parser
                var output = result.Output.Trim();
                
                // Extract isValid
                var isValidStr = output.Substring(1, output.IndexOf(",") - 1);
                var isValid = bool.Parse(isValidStr);
                
                // Extract errors
                var errorsStart = output.IndexOf("[");
                var errorsEnd = output.IndexOf("]");
                var errorsStr = output.Substring(errorsStart, errorsEnd - errorsStart + 1);
                var errors = ParseStringList(errorsStr);
                
                // Extract warnings
                var warningsStart = output.IndexOf("[", errorsEnd);
                var warningsEnd = output.IndexOf("]", warningsStart);
                var warningsStr = output.Substring(warningsStart, warningsEnd - warningsStart + 1);
                var warnings = ParseStringList(warningsStr);
                
                // Extract thought tree JSON
                var thoughtTreeJson = output.Substring(output.LastIndexOf(",") + 1, output.Length - output.LastIndexOf(",") - 2);
                
                return (isValid, errors, warnings, thoughtTreeJson);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error validating metascript");
                throw;
            }
        }

        /// <summary>
        /// Executes a metascript using Tree-of-Thought reasoning.
        /// </summary>
        /// <param name="metascript">The metascript to execute.</param>
        /// <returns>The execution results and the thought tree as JSON.</returns>
        public async Task<(string Output, bool Success, string Report, string ThoughtTreeJson)> ExecuteMetascriptAsync(string metascript)
        {
            _logger.LogInformation("Executing metascript using Tree-of-Thought reasoning");
            
            try
            {
                // Create the F# script
                var script = $@"
                    // Load the modules
                    #load ""{_fsharpModulePath}""
                    #load ""{_fsharpExecutionModulePath}""
                    
                    open TarsEngine.FSharp.MetascriptToT
                    open TarsEngine.FSharp.MetascriptExecution
                    
                    // Execute the metascript
                    let (thoughtTree, bestPlan, output, metrics, report) = 
                        Execution.planAndExecuteMetascript @""{metascript.Replace("\"", "\\\"")}""
                    
                    // Convert the thought tree to JSON
                    let thoughtTreeJson = ThoughtTree.toJson thoughtTree
                    
                    // Return the results as a tuple
                    (output, metrics.Success, report, thoughtTreeJson)
                ";
                
                // Execute the script
                var result = await _scriptExecutor.ExecuteScriptAsync(script);
                
                if (!result.Success)
                {
                    _logger.LogError("Error executing metascript: {Errors}", string.Join(Environment.NewLine, result.Errors));
                    throw new Exception($"Error executing metascript: {string.Join(Environment.NewLine, result.Errors)}");
                }
                
                // Parse the result
                // In a real implementation, this would use a proper parser
                var output = result.Output.Trim();
                
                // Extract output
                var outputStr = output.Substring(1, output.IndexOf(",") - 1);
                
                // Extract success
                var successStart = output.IndexOf(",") + 1;
                var successEnd = output.IndexOf(",", successStart);
                var successStr = output.Substring(successStart, successEnd - successStart);
                var success = bool.Parse(successStr);
                
                // Extract report
                var reportStart = output.IndexOf(",", successEnd) + 1;
                var reportEnd = output.LastIndexOf(",");
                var reportStr = output.Substring(reportStart, reportEnd - reportStart);
                
                // Extract thought tree JSON
                var thoughtTreeJson = output.Substring(output.LastIndexOf(",") + 1, output.Length - output.LastIndexOf(",") - 2);
                
                return (outputStr, success, reportStr, thoughtTreeJson);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error executing metascript");
                throw;
            }
        }

        /// <summary>
        /// Analyzes the results of a metascript execution using Tree-of-Thought reasoning.
        /// </summary>
        /// <param name="output">The output of the metascript execution.</param>
        /// <param name="executionTime">The execution time in milliseconds.</param>
        /// <param name="peakMemoryUsage">The peak memory usage in megabytes.</param>
        /// <param name="errorCount">The number of errors encountered.</param>
        /// <returns>The analysis results and the thought tree as JSON.</returns>
        public async Task<(bool Success, List<string> Errors, List<string> Warnings, string Impact, List<string> Recommendations, string ThoughtTreeJson)> AnalyzeResultsAsync(
            string output, int executionTime, int peakMemoryUsage, int errorCount)
        {
            _logger.LogInformation("Analyzing metascript results using Tree-of-Thought reasoning");
            
            try
            {
                // Create the F# script
                var script = $@"
                    // Load the modules
                    #load ""{_fsharpModulePath}""
                    #load ""{_fsharpExecutionModulePath}""
                    #load ""{_fsharpResultAnalysisModulePath}""
                    
                    open TarsEngine.FSharp.MetascriptToT
                    open TarsEngine.FSharp.MetascriptExecution
                    open TarsEngine.FSharp.MetascriptResultAnalysis
                    
                    // Create execution metrics
                    let metrics = {{
                        ExecutionTime = {executionTime}
                        PeakMemoryUsage = {peakMemoryUsage}
                        CpuUsage = 0.0 // Not available
                        ErrorCount = {errorCount}
                        Success = {(errorCount == 0).ToString().ToLower()}
                    }}
                    
                    // Analyze the results
                    let (thoughtTree, resultAnalysis) = 
                        Analysis.analyzeResults @""{output.Replace("\"", "\\\"")}""  metrics
                    
                    // Convert the thought tree to JSON
                    let thoughtTreeJson = ThoughtTree.toJson thoughtTree
                    
                    // Return the results as a tuple
                    (resultAnalysis.Success, resultAnalysis.Errors, resultAnalysis.Warnings, resultAnalysis.Impact, resultAnalysis.Recommendations, thoughtTreeJson)
                ";
                
                // Execute the script
                var result = await _scriptExecutor.ExecuteScriptAsync(script);
                
                if (!result.Success)
                {
                    _logger.LogError("Error analyzing metascript results: {Errors}", string.Join(Environment.NewLine, result.Errors));
                    throw new Exception($"Error analyzing metascript results: {string.Join(Environment.NewLine, result.Errors)}");
                }
                
                // Parse the result
                // In a real implementation, this would use a proper parser
                var output2 = result.Output.Trim();
                
                // Extract success
                var successStr = output2.Substring(1, output2.IndexOf(",") - 1);
                var success = bool.Parse(successStr);
                
                // Extract errors
                var errorsStart = output2.IndexOf("[");
                var errorsEnd = output2.IndexOf("]");
                var errorsStr = output2.Substring(errorsStart, errorsEnd - errorsStart + 1);
                var errors = ParseStringList(errorsStr);
                
                // Extract warnings
                var warningsStart = output2.IndexOf("[", errorsEnd);
                var warningsEnd = output2.IndexOf("]", warningsStart);
                var warningsStr = output2.Substring(warningsStart, warningsEnd - warningsStart + 1);
                var warnings = ParseStringList(warningsStr);
                
                // Extract impact
                var impactStart = output2.IndexOf(",", warningsEnd) + 1;
                var impactEnd = output2.IndexOf(",", impactStart);
                var impact = output2.Substring(impactStart, impactEnd - impactStart);
                
                // Extract recommendations
                var recommendationsStart = output2.IndexOf("[", impactEnd);
                var recommendationsEnd = output2.IndexOf("]", recommendationsStart);
                var recommendationsStr = output2.Substring(recommendationsStart, recommendationsEnd - recommendationsStart + 1);
                var recommendations = ParseStringList(recommendationsStr);
                
                // Extract thought tree JSON
                var thoughtTreeJson = output2.Substring(output2.LastIndexOf(",") + 1, output2.Length - output2.LastIndexOf(",") - 2);
                
                return (success, errors, warnings, impact, recommendations, thoughtTreeJson);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error analyzing metascript results");
                throw;
            }
        }

        /// <summary>
        /// Integrates the results of a metascript execution with the pipeline.
        /// </summary>
        /// <param name="output">The output of the metascript execution.</param>
        /// <param name="success">Whether the execution was successful.</param>
        /// <param name="recommendations">The recommendations from the analysis.</param>
        /// <returns>The integration results and the thought tree as JSON.</returns>
        public async Task<(bool Integrated, string Message, string ThoughtTreeJson)> IntegrateResultsAsync(
            string output, bool success, List<string> recommendations)
        {
            _logger.LogInformation("Integrating metascript results using Tree-of-Thought reasoning");
            
            try
            {
                // Create the F# script
                var script = $@"
                    // Load the modules
                    #load ""{_fsharpModulePath}""
                    
                    open TarsEngine.FSharp.MetascriptToT
                    
                    // Create the root thought
                    let root = ThoughtTree.createNode ""Integrate Metascript Results""
                    
                    // Create integration thought
                    let integrationThought = 
                        ThoughtTree.createNode ""Integrate Results""
                        |> ThoughtTree.addMetadata ""output"" @""{output.Replace("\"", "\\\"")}""
                        |> ThoughtTree.addMetadata ""success"" {success.ToString().ToLower()}
                    
                    // Simulate integration
                    let integrated = {success.ToString().ToLower()}
                    let message = 
                        if {success.ToString().ToLower()} then
                            ""Results integrated successfully""
                        else
                            ""Failed to integrate results due to errors""
                    
                    let integrationMetrics =
                        let correctness = if {success.ToString().ToLower()} then 0.9 else 0.5
                        let efficiency = 0.8
                        let robustness = if {success.ToString().ToLower()} then 0.9 else 0.6
                        let maintainability = 0.8
                        
                        Evaluation.createMetrics correctness efficiency robustness maintainability
                    
                    let evaluatedIntegrationThought = 
                        integrationThought
                        |> ThoughtTree.evaluateNode integrationMetrics
                        |> ThoughtTree.addMetadata ""integrated"" integrated
                        |> ThoughtTree.addMetadata ""message"" message
                    
                    // Add thoughts to root
                    let rootWithChildren = 
                        root
                        |> ThoughtTree.addChild evaluatedIntegrationThought
                    
                    // Evaluate root
                    let rootMetrics =
                        let correctness = if {success.ToString().ToLower()} then 0.9 else 0.5
                        let efficiency = 0.8
                        let robustness = if {success.ToString().ToLower()} then 0.9 else 0.6
                        let maintainability = 0.8
                        
                        Evaluation.createMetrics correctness efficiency robustness maintainability
                    
                    let evaluatedRoot = 
                        rootWithChildren
                        |> ThoughtTree.evaluateNode rootMetrics
                        |> ThoughtTree.addMetadata ""integrated"" integrated
                        |> ThoughtTree.addMetadata ""message"" message
                    
                    // Convert the thought tree to JSON
                    let thoughtTreeJson = ThoughtTree.toJson evaluatedRoot
                    
                    // Return the results as a tuple
                    (integrated, message, thoughtTreeJson)
                ";
                
                // Execute the script
                var result = await _scriptExecutor.ExecuteScriptAsync(script);
                
                if (!result.Success)
                {
                    _logger.LogError("Error integrating metascript results: {Errors}", string.Join(Environment.NewLine, result.Errors));
                    throw new Exception($"Error integrating metascript results: {string.Join(Environment.NewLine, result.Errors)}");
                }
                
                // Parse the result
                // In a real implementation, this would use a proper parser
                var output2 = result.Output.Trim();
                
                // Extract integrated
                var integratedStr = output2.Substring(1, output2.IndexOf(",") - 1);
                var integrated = bool.Parse(integratedStr);
                
                // Extract message
                var messageStart = output2.IndexOf(",") + 1;
                var messageEnd = output2.LastIndexOf(",");
                var message = output2.Substring(messageStart, messageEnd - messageStart);
                
                // Extract thought tree JSON
                var thoughtTreeJson = output2.Substring(output2.LastIndexOf(",") + 1, output2.Length - output2.LastIndexOf(",") - 2);
                
                return (integrated, message, thoughtTreeJson);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error integrating metascript results");
                throw;
            }
        }

        private string ConvertDictionaryToFSharpMap(Dictionary<string, string> dictionary)
        {
            var entries = new List<string>();
            
            foreach (var kvp in dictionary)
            {
                entries.Add($"\"{kvp.Key}\", \"{kvp.Value.Replace("\"", "\\\"")}\"");
            }
            
            return $"Map.ofList [({string.Join("); (", entries)})]";
        }

        private List<string> ParseStringList(string listStr)
        {
            var result = new List<string>();
            
            // Simple parsing for demonstration purposes
            // In a real implementation, this would use a proper parser
            if (listStr == "[]")
            {
                return result;
            }
            
            var items = listStr.Substring(1, listStr.Length - 2).Split(';');
            
            foreach (var item in items)
            {
                if (!string.IsNullOrWhiteSpace(item))
                {
                    result.Add(item.Trim());
                }
            }
            
            return result;
        }
    }
}
