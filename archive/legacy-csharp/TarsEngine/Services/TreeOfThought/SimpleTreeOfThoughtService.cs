using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Compilation;

namespace TarsEngine.Services.TreeOfThought
{
    /// <summary>
    /// A simplified service for Tree-of-Thought reasoning.
    /// </summary>
    public class SimpleTreeOfThoughtService
    {
        private readonly ILogger<SimpleTreeOfThoughtService> _logger;
        private readonly FSharpScriptExecutor _scriptExecutor;
        private readonly string _fsharpModulePath;

        /// <summary>
        /// Initializes a new instance of the <see cref="SimpleTreeOfThoughtService"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="scriptExecutor">The F# script executor.</param>
        public SimpleTreeOfThoughtService(ILogger<SimpleTreeOfThoughtService> logger, FSharpScriptExecutor scriptExecutor)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _scriptExecutor = scriptExecutor ?? throw new ArgumentNullException(nameof(scriptExecutor));
            _fsharpModulePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TarsEngine", "FSharp", "SimpleTreeOfThought.fs");
            
            // Ensure the F# module exists
            if (!File.Exists(_fsharpModulePath))
            {
                _logger.LogWarning("F# Simple Tree-of-Thought module not found at {Path}", _fsharpModulePath);
            }
            else
            {
                _logger.LogInformation("F# Simple Tree-of-Thought module found at {Path}", _fsharpModulePath);
            }
        }

        /// <summary>
        /// Analyzes code using Tree-of-Thought reasoning.
        /// </summary>
        /// <param name="code">The code to analyze.</param>
        /// <returns>The analysis result as JSON.</returns>
        public async Task<string> AnalyzeCodeAsync(string code)
        {
            _logger.LogInformation("Analyzing code using Tree-of-Thought reasoning");
            
            try
            {
                // Create the F# script
                var script = $@"
                    // Load the Simple Tree-of-Thought module
                    #load ""{_fsharpModulePath}""
                    
                    open TarsEngine.FSharp.SimpleTreeOfThought
                    
                    // Analyze the code
                    let analysisTree = Analysis.analyzeCode @""{code.Replace("\"", "\\\"")}""
                    
                    // Convert to JSON
                    ThoughtTree.toJson analysisTree
                ";
                
                // Execute the script
                var result = await _scriptExecutor.ExecuteScriptAsync(script);
                
                if (!result.Success)
                {
                    _logger.LogError("Error analyzing code: {Errors}", string.Join(Environment.NewLine, result.Errors));
                    throw new Exception($"Error analyzing code: {string.Join(Environment.NewLine, result.Errors)}");
                }
                
                return result.Output.Trim();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error analyzing code");
                throw;
            }
        }

        /// <summary>
        /// Generates fixes for an issue using Tree-of-Thought reasoning.
        /// </summary>
        /// <param name="issue">The issue to fix.</param>
        /// <returns>The fix generation result as JSON.</returns>
        public async Task<string> GenerateFixesAsync(string issue)
        {
            _logger.LogInformation("Generating fixes using Tree-of-Thought reasoning");
            
            try
            {
                // Create the F# script
                var script = $@"
                    // Load the Simple Tree-of-Thought module
                    #load ""{_fsharpModulePath}""
                    
                    open TarsEngine.FSharp.SimpleTreeOfThought
                    
                    // Generate fixes
                    let fixTree = FixGeneration.generateFixes @""{issue.Replace("\"", "\\\"")}""
                    
                    // Convert to JSON
                    ThoughtTree.toJson fixTree
                ";
                
                // Execute the script
                var result = await _scriptExecutor.ExecuteScriptAsync(script);
                
                if (!result.Success)
                {
                    _logger.LogError("Error generating fixes: {Errors}", string.Join(Environment.NewLine, result.Errors));
                    throw new Exception($"Error generating fixes: {string.Join(Environment.NewLine, result.Errors)}");
                }
                
                return result.Output.Trim();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating fixes");
                throw;
            }
        }

        /// <summary>
        /// Applies a fix using Tree-of-Thought reasoning.
        /// </summary>
        /// <param name="fix">The fix to apply.</param>
        /// <returns>The fix application result as JSON.</returns>
        public async Task<string> ApplyFixAsync(string fix)
        {
            _logger.LogInformation("Applying fix using Tree-of-Thought reasoning");
            
            try
            {
                // Create the F# script
                var script = $@"
                    // Load the Simple Tree-of-Thought module
                    #load ""{_fsharpModulePath}""
                    
                    open TarsEngine.FSharp.SimpleTreeOfThought
                    
                    // Apply fix
                    let applicationTree = FixApplication.applyFix @""{fix.Replace("\"", "\\\"")}""
                    
                    // Convert to JSON
                    ThoughtTree.toJson applicationTree
                ";
                
                // Execute the script
                var result = await _scriptExecutor.ExecuteScriptAsync(script);
                
                if (!result.Success)
                {
                    _logger.LogError("Error applying fix: {Errors}", string.Join(Environment.NewLine, result.Errors));
                    throw new Exception($"Error applying fix: {string.Join(Environment.NewLine, result.Errors)}");
                }
                
                return result.Output.Trim();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error applying fix");
                throw;
            }
        }

        /// <summary>
        /// Selects the best approach from a thought tree.
        /// </summary>
        /// <param name="thoughtTreeJson">The thought tree as JSON.</param>
        /// <returns>The best approach as JSON.</returns>
        public async Task<string> SelectBestApproachAsync(string thoughtTreeJson)
        {
            _logger.LogInformation("Selecting best approach");
            
            try
            {
                // For simplicity, we'll just extract the highest-scored child from the JSON
                // In a real implementation, we would parse the JSON and use the Selection module
                
                // Create the F# script
                var script = $@"
                    // Load the Simple Tree-of-Thought module
                    #load ""{_fsharpModulePath}""
                    
                    open TarsEngine.FSharp.SimpleTreeOfThought
                    open System.Text.Json
                    
                    // Parse the thought tree JSON (simplified)
                    let thoughtTreeJson = @""{thoughtTreeJson.Replace("\"", "\\\"")}""
                    
                    // Extract the highest-scored child (simplified)
                    let jsonDoc = JsonDocument.Parse(thoughtTreeJson)
                    let root = jsonDoc.RootElement
                    
                    // Get the children
                    let children = 
                        if root.TryGetProperty(""children"", out let childrenProp) then
                            childrenProp
                        else
                            JsonDocument.Parse(""[]"").RootElement
                    
                    // Find the highest-scored child
                    let mutable bestChild = null
                    let mutable bestScore = 0.0
                    
                    for i = 0 to children.GetArrayLength() - 1 do
                        let child = children[i]
                        if child.TryGetProperty(""score"", out let scoreProp) then
                            let score = scoreProp.GetDouble()
                            if score > bestScore then
                                bestScore <- score
                                bestChild <- child
                    
                    // Return the best child as JSON
                    if bestChild <> null then
                        bestChild.ToString()
                    else
                        ""{{}}""
                ";
                
                // Execute the script
                var result = await _scriptExecutor.ExecuteScriptAsync(script);
                
                if (!result.Success)
                {
                    _logger.LogError("Error selecting best approach: {Errors}", string.Join(Environment.NewLine, result.Errors));
                    throw new Exception($"Error selecting best approach: {string.Join(Environment.NewLine, result.Errors)}");
                }
                
                return result.Output.Trim();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error selecting best approach");
                throw;
            }
        }
    }
}
