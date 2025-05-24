using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Compilation;

namespace TarsEngine.Services.TreeOfThought
{
    /// <summary>
    /// Service for Tree-of-Thought reasoning.
    /// </summary>
    public class TreeOfThoughtService
    {
        private readonly ILogger<TreeOfThoughtService> _logger;
        private readonly FSharpScriptExecutor _scriptExecutor;
        private readonly string _fsharpModulePath;

        /// <summary>
        /// Initializes a new instance of the <see cref="TreeOfThoughtService"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="scriptExecutor">The F# script executor.</param>
        public TreeOfThoughtService(ILogger<TreeOfThoughtService> logger, FSharpScriptExecutor scriptExecutor)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _scriptExecutor = scriptExecutor ?? throw new ArgumentNullException(nameof(scriptExecutor));
            _fsharpModulePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TarsEngine", "FSharp", "TreeOfThought.fs");
            
            // Ensure the F# module exists
            if (!File.Exists(_fsharpModulePath))
            {
                _logger.LogWarning("F# Tree-of-Thought module not found at {Path}", _fsharpModulePath);
            }
            else
            {
                _logger.LogInformation("F# Tree-of-Thought module found at {Path}", _fsharpModulePath);
            }
        }

        /// <summary>
        /// Creates a thought tree for code analysis.
        /// </summary>
        /// <param name="code">The code to analyze.</param>
        /// <param name="branchingFactor">The branching factor.</param>
        /// <param name="beamWidth">The beam width.</param>
        /// <returns>The thought tree as JSON.</returns>
        public async Task<string> CreateAnalysisThoughtTreeAsync(string code, int branchingFactor = 3, int beamWidth = 2)
        {
            _logger.LogInformation("Creating analysis thought tree with branching factor {BranchingFactor} and beam width {BeamWidth}", branchingFactor, beamWidth);
            
            // Create the F# script
            var script = $@"
                // Load the Tree-of-Thought module
                #load ""{_fsharpModulePath}""
                
                open TarsEngine.FSharp.TreeOfThought
                
                // Create the root thought
                let root = ThoughtTree.createNode ""Code Analysis Problem""
                
                // Generate analysis approaches
                let approaches = Branching.generateAnalysisApproaches @""{code.Replace("\"", "\\\"")}""
                
                // Add approaches to root
                let rootWithApproaches = 
                    approaches
                    |> List.fold (fun r a -> ThoughtTree.addChild r a) root
                
                // Evaluate approaches
                let evaluatedApproaches =
                    rootWithApproaches.Children
                    |> List.map Pruning.evaluateAnalysisNode
                
                // Update root with evaluated approaches
                let rootWithEvaluatedApproaches =
                    {{ rootWithApproaches with Children = evaluatedApproaches }}
                
                // Perform beam search
                let prunedTree = 
                    Pruning.beamSearch rootWithEvaluatedApproaches {beamWidth} Pruning.scoreNode
                
                // Convert to JSON
                ThoughtTree.toJson prunedTree
            ";
            
            // Execute the script
            var result = await _scriptExecutor.ExecuteScriptAsync(script);
            
            if (!result.Success)
            {
                _logger.LogError("Error creating analysis thought tree: {Errors}", string.Join(Environment.NewLine, result.Errors));
                throw new Exception($"Error creating analysis thought tree: {string.Join(Environment.NewLine, result.Errors)}");
            }
            
            return result.Output.Trim();
        }

        /// <summary>
        /// Creates a thought tree for fix generation.
        /// </summary>
        /// <param name="issue">The issue to fix.</param>
        /// <param name="branchingFactor">The branching factor.</param>
        /// <param name="beamWidth">The beam width.</param>
        /// <returns>The thought tree as JSON.</returns>
        public async Task<string> CreateFixGenerationThoughtTreeAsync(string issue, int branchingFactor = 3, int beamWidth = 2)
        {
            _logger.LogInformation("Creating fix generation thought tree with branching factor {BranchingFactor} and beam width {BeamWidth}", branchingFactor, beamWidth);
            
            // Create the F# script
            var script = $@"
                // Load the Tree-of-Thought module
                #load ""{_fsharpModulePath}""
                
                open TarsEngine.FSharp.TreeOfThought
                
                // Create the root thought
                let root = ThoughtTree.createNode ""Fix Generation Problem""
                
                // Generate fix approaches
                let approaches = Branching.generateFixApproaches @""{issue.Replace("\"", "\\\"")}""
                
                // Add approaches to root
                let rootWithApproaches = 
                    approaches
                    |> List.fold (fun r a -> ThoughtTree.addChild r a) root
                
                // Evaluate approaches
                let evaluatedApproaches =
                    rootWithApproaches.Children
                    |> List.map Pruning.evaluateFixNode
                
                // Update root with evaluated approaches
                let rootWithEvaluatedApproaches =
                    {{ rootWithApproaches with Children = evaluatedApproaches }}
                
                // Perform beam search
                let prunedTree = 
                    Pruning.beamSearch rootWithEvaluatedApproaches {beamWidth} Pruning.scoreNode
                
                // Convert to JSON
                ThoughtTree.toJson prunedTree
            ";
            
            // Execute the script
            var result = await _scriptExecutor.ExecuteScriptAsync(script);
            
            if (!result.Success)
            {
                _logger.LogError("Error creating fix generation thought tree: {Errors}", string.Join(Environment.NewLine, result.Errors));
                throw new Exception($"Error creating fix generation thought tree: {string.Join(Environment.NewLine, result.Errors)}");
            }
            
            return result.Output.Trim();
        }

        /// <summary>
        /// Creates a thought tree for fix application.
        /// </summary>
        /// <param name="fix">The fix to apply.</param>
        /// <param name="branchingFactor">The branching factor.</param>
        /// <param name="beamWidth">The beam width.</param>
        /// <returns>The thought tree as JSON.</returns>
        public async Task<string> CreateFixApplicationThoughtTreeAsync(string fix, int branchingFactor = 3, int beamWidth = 2)
        {
            _logger.LogInformation("Creating fix application thought tree with branching factor {BranchingFactor} and beam width {BeamWidth}", branchingFactor, beamWidth);
            
            // Create the F# script
            var script = $@"
                // Load the Tree-of-Thought module
                #load ""{_fsharpModulePath}""
                
                open TarsEngine.FSharp.TreeOfThought
                
                // Create the root thought
                let root = ThoughtTree.createNode ""Fix Application Problem""
                
                // Generate application approaches
                let approaches = Branching.generateApplicationApproaches @""{fix.Replace("\"", "\\\"")}""
                
                // Add approaches to root
                let rootWithApproaches = 
                    approaches
                    |> List.fold (fun r a -> ThoughtTree.addChild r a) root
                
                // Evaluate approaches
                let evaluatedApproaches =
                    rootWithApproaches.Children
                    |> List.map Pruning.evaluateApplicationNode
                
                // Update root with evaluated approaches
                let rootWithEvaluatedApproaches =
                    {{ rootWithApproaches with Children = evaluatedApproaches }}
                
                // Perform beam search
                let prunedTree = 
                    Pruning.beamSearch rootWithEvaluatedApproaches {beamWidth} Pruning.scoreNode
                
                // Convert to JSON
                ThoughtTree.toJson prunedTree
            ";
            
            // Execute the script
            var result = await _scriptExecutor.ExecuteScriptAsync(script);
            
            if (!result.Success)
            {
                _logger.LogError("Error creating fix application thought tree: {Errors}", string.Join(Environment.NewLine, result.Errors));
                throw new Exception($"Error creating fix application thought tree: {string.Join(Environment.NewLine, result.Errors)}");
            }
            
            return result.Output.Trim();
        }

        /// <summary>
        /// Selects the best approach from a thought tree.
        /// </summary>
        /// <param name="thoughtTreeJson">The thought tree as JSON.</param>
        /// <param name="selectionStrategy">The selection strategy to use.</param>
        /// <returns>The best approach as JSON.</returns>
        public async Task<string> SelectBestApproachAsync(string thoughtTreeJson, string selectionStrategy = "bestFirst")
        {
            _logger.LogInformation("Selecting best approach using strategy {SelectionStrategy}", selectionStrategy);
            
            // Create the F# script
            var script = $@"
                // Load the Tree-of-Thought module
                #load ""{_fsharpModulePath}""
                
                open TarsEngine.FSharp.TreeOfThought
                open System.Text.Json
                
                // Parse the thought tree JSON
                let thoughtTreeJson = @""{thoughtTreeJson.Replace("\"", "\\\"")}""
                
                // TODO: Implement JSON parsing to convert back to ThoughtNode
                // For now, just extract the children and select the best one
                
                // Simple JSON parsing to get the children
                let jsonDoc = JsonDocument.Parse(thoughtTreeJson)
                let root = jsonDoc.RootElement
                let children = 
                    if root.TryGetProperty(""children"", out let childrenProp) then
                        childrenProp
                    else
                        JsonDocument.Parse(""[]"").RootElement
                
                // Convert children to JSON strings
                let childrenJson = 
                    [0..children.GetArrayLength() - 1]
                    |> List.map (fun i -> children[i].ToString())
                
                // Select the best child based on the strategy
                let bestChildJson =
                    match ""{selectionStrategy}"" with
                    | ""bestFirst"" -> 
                        if childrenJson.Length > 0 then
                            childrenJson.[0]
                        else
                            ""{{}}""
                    | ""diversityBased"" ->
                        if childrenJson.Length > 0 then
                            childrenJson.[0]
                        else
                            ""{{}}""
                    | ""confidenceBased"" ->
                        if childrenJson.Length > 0 then
                            childrenJson.[0]
                        else
                            ""{{}}""
                    | ""hybridSelection"" ->
                        if childrenJson.Length > 0 then
                            childrenJson.[0]
                        else
                            ""{{}}""
                    | _ ->
                        if childrenJson.Length > 0 then
                            childrenJson.[0]
                        else
                            ""{{}}""
                
                // Return the best child
                bestChildJson
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
    }
}
