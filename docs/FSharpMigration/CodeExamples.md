# F# Migration Code Examples

This document provides detailed code examples for using the new F# implementation in `TarsEngine.FSharp.Core` and its C# adapters in `TarsEngine.FSharp.Adapters`.

## Table of Contents

1. [F# Examples](#f-examples)
   - [Creating and Manipulating Thought Nodes](#creating-and-manipulating-thought-nodes)
   - [Evaluating Thought Nodes](#evaluating-thought-nodes)
   - [Working with Thought Trees](#working-with-thought-trees)
   - [Visualizing Thought Trees](#visualizing-thought-trees)
2. [C# Examples](#c-examples)
   - [Using the Adapter Layer](#using-the-adapter-layer)
   - [Using the Tree-of-Thought Service](#using-the-tree-of-thought-service)
   - [Creating a Custom Service](#creating-a-custom-service)
3. [Migration Examples](#migration-examples)
   - [Migrating F# Code](#migrating-f-code)
   - [Migrating C# Code](#migrating-c-code)

## F# Examples

### Creating and Manipulating Thought Nodes

```fsharp
open TarsEngine.FSharp.Core.TreeOfThought

// Create a thought node
let root = ThoughtNode.createNode "How to improve code quality?"

// Create child nodes
let staticAnalysis = ThoughtNode.createNode "Static Analysis"
let codeReviews = ThoughtNode.createNode "Code Reviews"
let automatedTesting = ThoughtNode.createNode "Automated Testing"
let refactoring = ThoughtNode.createNode "Refactoring"

// Add children to the root
let rootWithChildren = 
    root
    |> ThoughtNode.addChild staticAnalysis
    |> ThoughtNode.addChild codeReviews
    |> ThoughtNode.addChild automatedTesting
    |> ThoughtNode.addChild refactoring

// Add multiple children at once
let rootWithChildren2 = 
    ThoughtNode.addChildren root [staticAnalysis; codeReviews; automatedTesting; refactoring]

// Add metadata to a node
let rootWithMetadata = 
    rootWithChildren
    |> ThoughtNode.addMetadata "author" "John Doe"
    |> ThoughtNode.addMetadata "date" (System.DateTime.Now)

// Get metadata from a node
let author = ThoughtNode.getMetadata<string> rootWithMetadata "author"
```

### Evaluating Thought Nodes

```fsharp
open TarsEngine.FSharp.Core.TreeOfThought

// Create evaluation metrics with equal weights
let metrics1 = ThoughtNode.createMetrics 0.8 0.7 0.6 0.5

// Create evaluation metrics with custom weights
let metrics2 = ThoughtNode.createWeightedMetrics 0.8 0.7 0.6 0.5 (2.0, 1.0, 1.0, 1.0)

// Evaluate a node
let node = ThoughtNode.createNode "Static Analysis"
let evaluatedNode = ThoughtNode.evaluateNode node metrics1

// Get the score of a node
let score = ThoughtNode.getScore evaluatedNode
```

### Working with Thought Trees

```fsharp
open TarsEngine.FSharp.Core.TreeOfThought

// Create a thought tree
let root = ThoughtNode.createNode "How to improve code quality?"
let staticAnalysis = ThoughtNode.createNode "Static Analysis"
let codeReviews = ThoughtNode.createNode "Code Reviews"
let automatedTesting = ThoughtNode.createNode "Automated Testing"
let refactoring = ThoughtNode.createNode "Refactoring"

let rootWithChildren = 
    root
    |> ThoughtNode.addChild staticAnalysis
    |> ThoughtNode.addChild codeReviews
    |> ThoughtNode.addChild automatedTesting
    |> ThoughtNode.addChild refactoring

// Evaluate the nodes
let metrics1 = ThoughtNode.createMetrics 0.7 0.8 0.6 0.7
let metrics2 = ThoughtNode.createMetrics 0.9 0.7 0.8 0.9
let metrics3 = ThoughtNode.createMetrics 0.8 0.9 0.9 0.7
let metrics4 = ThoughtNode.createMetrics 0.7 0.6 0.8 0.9

let evaluatedStaticAnalysis = ThoughtNode.evaluateNode staticAnalysis metrics1
let evaluatedCodeReviews = ThoughtNode.evaluateNode codeReviews metrics2
let evaluatedAutomatedTesting = ThoughtNode.evaluateNode automatedTesting metrics3
let evaluatedRefactoring = ThoughtNode.evaluateNode refactoring metrics4

let evaluatedRoot = 
    root
    |> ThoughtNode.addChild evaluatedStaticAnalysis
    |> ThoughtNode.addChild evaluatedCodeReviews
    |> ThoughtNode.addChild evaluatedAutomatedTesting
    |> ThoughtNode.addChild evaluatedRefactoring

// Get the depth of the tree
let depth = ThoughtTree.depth evaluatedRoot

// Get the breadth of the tree at a specific level
let breadth = ThoughtTree.breadthAtLevel 1 evaluatedRoot

// Get the maximum breadth of the tree
let maxBreadth = ThoughtTree.maxBreadth evaluatedRoot

// Find a node by its thought content
let foundNode = ThoughtTree.findNode "Code Reviews" evaluatedRoot

// Find nodes that match a predicate
let highScoringNodes = 
    ThoughtTree.findNodes 
        (fun node -> 
            match node.Evaluation with
            | Some eval -> eval.Overall > 0.8
            | None -> false) 
        evaluatedRoot

// Select the best node based on evaluation
let bestNode = ThoughtTree.selectBestNode evaluatedRoot

// Prune nodes that don't meet a threshold
let prunedTree = ThoughtTree.pruneByThreshold 0.8 evaluatedRoot

// Prune all but the top k nodes at each level
let prunedTree2 = ThoughtTree.pruneBeamSearch 2 evaluatedRoot

// Map a function over all nodes in a tree
let mappedTree = 
    ThoughtTree.mapTree 
        (fun node -> 
            match node.Evaluation with
            | Some eval -> 
                let newEval = { eval with Overall = eval.Overall * 1.1 }
                { node with Evaluation = Some newEval }
            | None -> node) 
        evaluatedRoot

// Count the number of nodes in a tree
let nodeCount = ThoughtTree.countNodes evaluatedRoot

// Count the number of evaluated nodes in a tree
let evaluatedNodeCount = ThoughtTree.countEvaluatedNodes evaluatedRoot

// Count the number of pruned nodes in a tree
let prunedNodeCount = ThoughtTree.countPrunedNodes prunedTree
```

### Visualizing Thought Trees

```fsharp
open TarsEngine.FSharp.Core.TreeOfThought

// Create and evaluate a thought tree (as in the previous example)
// ...

// Convert a tree to JSON
let json = Visualization.toJson evaluatedRoot

// Convert a tree to a formatted JSON string
let formattedJson = Visualization.toFormattedJson evaluatedRoot

// Convert a tree to a Markdown representation
let markdown = Visualization.toMarkdown evaluatedRoot 0

// Convert a tree to a Markdown report
let report = Visualization.toMarkdownReport evaluatedRoot "Code Quality Improvement"

// Convert a tree to a DOT graph representation for Graphviz
let dotGraph = Visualization.toDotGraph evaluatedRoot "Code Quality Improvement"

// Save a DOT graph to a file
Visualization.saveDotGraph evaluatedRoot "Code Quality Improvement" "CodeQuality.dot"

// Save a Markdown report to a file
Visualization.saveMarkdownReport evaluatedRoot "Code Quality Improvement" "CodeQuality.md"

// Save a JSON representation to a file
Visualization.saveJsonReport evaluatedRoot "CodeQuality.json"
```

## C# Examples

### Using the Adapter Layer

```csharp
using TarsEngine.FSharp.Adapters.TreeOfThought;

// Create a thought node
var root = ThoughtNodeAdapter.CreateNode("How to improve code quality?");

// Create child nodes
var staticAnalysis = ThoughtNodeAdapter.CreateNode("Static Analysis");
var codeReviews = ThoughtNodeAdapter.CreateNode("Code Reviews");
var automatedTesting = ThoughtNodeAdapter.CreateNode("Automated Testing");
var refactoring = ThoughtNodeAdapter.CreateNode("Refactoring");

// Add children to the root
var rootWithChildren = root
    .AddChild(staticAnalysis)
    .AddChild(codeReviews)
    .AddChild(automatedTesting)
    .AddChild(refactoring);

// Add multiple children at once
var rootWithChildren2 = root.AddChildren(new[] { staticAnalysis, codeReviews, automatedTesting, refactoring });

// Add metadata to a node
var rootWithMetadata = rootWithChildren
    .AddMetadata("author", "John Doe")
    .AddMetadata("date", DateTime.Now);

// Get metadata from a node
var author = rootWithMetadata.GetMetadata<string>("author");

// Create evaluation metrics
var metrics1 = EvaluationMetricsAdapter.CreateMetrics(0.8, 0.7, 0.6, 0.5);
var metrics2 = EvaluationMetricsAdapter.CreateWeightedMetrics(0.8, 0.7, 0.6, 0.5, 2.0, 1.0, 1.0, 1.0);

// Evaluate a node
var evaluatedNode = staticAnalysis.EvaluateNode(metrics1);

// Get the score of a node
var score = evaluatedNode.GetScore();

// Find the best node
var bestNode = ThoughtTreeAdapter.SelectBestNode(rootWithChildren);

// Prune nodes that don't meet a threshold
var prunedTree = ThoughtTreeAdapter.PruneByThreshold(0.8, rootWithChildren);

// Convert a tree to a Markdown report
var report = VisualizationAdapter.ToMarkdownReport(rootWithChildren, "Code Quality Improvement");

// Save a Markdown report to a file
VisualizationAdapter.SaveMarkdownReport(rootWithChildren, "Code Quality Improvement", "CodeQuality.md");
```

### Using the Tree-of-Thought Service

```csharp
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Abstractions.TreeOfThought;
using TarsEngine.FSharp.Adapters.TreeOfThought;

// Create a logger
var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
var logger = loggerFactory.CreateLogger<FSharpTreeOfThoughtService>();

// Create the service
var service = new FSharpTreeOfThoughtService(logger);

// Create a thought tree
var options = new TreeCreationOptions
{
    Approaches = new List<string>
    {
        "Static Analysis",
        "Code Reviews",
        "Automated Testing",
        "Refactoring"
    },
    ApproachEvaluations = new Dictionary<string, EvaluationMetrics>
    {
        ["Static Analysis"] = new EvaluationMetrics(0.7, 0.8, 0.6, 0.7, 0.7),
        ["Code Reviews"] = new EvaluationMetrics(0.9, 0.7, 0.8, 0.9, 0.825),
        ["Automated Testing"] = new EvaluationMetrics(0.8, 0.9, 0.9, 0.7, 0.825),
        ["Refactoring"] = new EvaluationMetrics(0.7, 0.6, 0.8, 0.9, 0.75)
    }
};

var root = await service.CreateThoughtTreeAsync("How to improve code quality?", options);

// Select the best node
var bestNode = await service.SelectBestNodeAsync(root);
Console.WriteLine($"Best approach: {bestNode.Thought} (Score: {bestNode.Score})");

// Add child nodes to the best approach
await service.AddChildAsync(bestNode, "Implementation 1");
await service.AddChildAsync(bestNode, "Implementation 2");
await service.AddChildAsync(bestNode, "Implementation 3");

// Generate a report
var report = await service.GenerateReportAsync(root, "Code Quality Improvement");
Console.WriteLine(report);

// Save the report to a file
await service.SaveReportAsync(root, "Code Quality Improvement", "CodeQuality.md");
```

### Creating a Custom Service

```csharp
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Abstractions.TreeOfThought;
using TarsEngine.FSharp.Adapters.TreeOfThought;

public class CustomTreeOfThoughtService : ITreeOfThoughtService
{
    private readonly FSharpTreeOfThoughtService _fsharpService;
    private readonly ILogger<CustomTreeOfThoughtService> _logger;

    public CustomTreeOfThoughtService(ILogger<CustomTreeOfThoughtService> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _fsharpService = new FSharpTreeOfThoughtService(logger);
    }

    public async Task<IThoughtNode> CreateThoughtTreeAsync(string problem, TreeCreationOptions options)
    {
        _logger.LogInformation("Creating thought tree with custom service for problem: {Problem}", problem);
        
        // Add custom logic here
        
        return await _fsharpService.CreateThoughtTreeAsync(problem, options);
    }

    public async Task<IThoughtNode> EvaluateNodeAsync(IThoughtNode node, EvaluationMetrics metrics)
    {
        _logger.LogInformation("Evaluating node with custom service: {Thought}", node.Thought);
        
        // Add custom logic here
        
        return await _fsharpService.EvaluateNodeAsync(node, metrics);
    }

    public async Task<IThoughtNode> AddChildAsync(IThoughtNode parent, string childThought)
    {
        _logger.LogInformation("Adding child with custom service to node: {Thought}", parent.Thought);
        
        // Add custom logic here
        
        return await _fsharpService.AddChildAsync(parent, childThought);
    }

    public async Task<IThoughtNode> SelectBestNodeAsync(IThoughtNode root)
    {
        _logger.LogInformation("Selecting best node with custom service from tree: {Thought}", root.Thought);
        
        // Add custom logic here
        
        return await _fsharpService.SelectBestNodeAsync(root);
    }

    public async Task<IThoughtNode> PruneByThresholdAsync(IThoughtNode root, double threshold)
    {
        _logger.LogInformation("Pruning tree with custom service by threshold: {Threshold}", threshold);
        
        // Add custom logic here
        
        return await _fsharpService.PruneByThresholdAsync(root, threshold);
    }

    public async Task<string> GenerateReportAsync(IThoughtNode root, string title)
    {
        _logger.LogInformation("Generating report with custom service for tree: {Thought}", root.Thought);
        
        // Add custom logic here
        
        return await _fsharpService.GenerateReportAsync(root, title);
    }

    public async Task SaveReportAsync(IThoughtNode root, string title, string filePath)
    {
        _logger.LogInformation("Saving report with custom service for tree: {Thought}", root.Thought);
        
        // Add custom logic here
        
        await _fsharpService.SaveReportAsync(root, title, filePath);
    }
}
```

## Migration Examples

### Migrating F# Code

#### Old Code

```fsharp
open TarsEngine.FSharp.MetascriptToT

// Create a thought tree
let root = createNode "How to improve code quality?"
let staticAnalysis = createNode "Static Analysis"
let codeReviews = createNode "Code Reviews"
let automatedTesting = createNode "Automated Testing"
let refactoring = createNode "Refactoring"

let rootWithChildren = 
    root
    |> addChild staticAnalysis
    |> addChild codeReviews
    |> addChild automatedTesting
    |> addChild refactoring

// Evaluate the nodes
let metrics1 = { Correctness = 0.7; Efficiency = 0.8; Robustness = 0.6; Maintainability = 0.7; Overall = 0.7 }
let metrics2 = { Correctness = 0.9; Efficiency = 0.7; Robustness = 0.8; Maintainability = 0.9; Overall = 0.825 }
let metrics3 = { Correctness = 0.8; Efficiency = 0.9; Robustness = 0.9; Maintainability = 0.7; Overall = 0.825 }
let metrics4 = { Correctness = 0.7; Efficiency = 0.6; Robustness = 0.8; Maintainability = 0.9; Overall = 0.75 }

let evaluatedStaticAnalysis = evaluateNode staticAnalysis metrics1
let evaluatedCodeReviews = evaluateNode codeReviews metrics2
let evaluatedAutomatedTesting = evaluateNode automatedTesting metrics3
let evaluatedRefactoring = evaluateNode refactoring metrics4

let evaluatedRoot = 
    root
    |> addChild evaluatedStaticAnalysis
    |> addChild evaluatedCodeReviews
    |> addChild evaluatedAutomatedTesting
    |> addChild evaluatedRefactoring

// Select the best node
let bestNode = selectBestNode evaluatedRoot
```

#### New Code

```fsharp
open TarsEngine.FSharp.Core.TreeOfThought

// Create a thought tree
let root = ThoughtNode.createNode "How to improve code quality?"
let staticAnalysis = ThoughtNode.createNode "Static Analysis"
let codeReviews = ThoughtNode.createNode "Code Reviews"
let automatedTesting = ThoughtNode.createNode "Automated Testing"
let refactoring = ThoughtNode.createNode "Refactoring"

let rootWithChildren = 
    root
    |> ThoughtNode.addChild staticAnalysis
    |> ThoughtNode.addChild codeReviews
    |> ThoughtNode.addChild automatedTesting
    |> ThoughtNode.addChild refactoring

// Evaluate the nodes
let metrics1 = ThoughtNode.createMetrics 0.7 0.8 0.6 0.7
let metrics2 = ThoughtNode.createMetrics 0.9 0.7 0.8 0.9
let metrics3 = ThoughtNode.createMetrics 0.8 0.9 0.9 0.7
let metrics4 = ThoughtNode.createMetrics 0.7 0.6 0.8 0.9

let evaluatedStaticAnalysis = ThoughtNode.evaluateNode staticAnalysis metrics1
let evaluatedCodeReviews = ThoughtNode.evaluateNode codeReviews metrics2
let evaluatedAutomatedTesting = ThoughtNode.evaluateNode automatedTesting metrics3
let evaluatedRefactoring = ThoughtNode.evaluateNode refactoring metrics4

let evaluatedRoot = 
    root
    |> ThoughtNode.addChild evaluatedStaticAnalysis
    |> ThoughtNode.addChild evaluatedCodeReviews
    |> ThoughtNode.addChild evaluatedAutomatedTesting
    |> ThoughtNode.addChild evaluatedRefactoring

// Select the best node
let bestNode = ThoughtTree.selectBestNode evaluatedRoot
```

### Migrating C# Code

#### Old Code

```csharp
using TarsEngine.Services.TreeOfThought;

// Create the service
var service = new SimpleTreeOfThoughtService();

// Create a thought tree
var options = new TreeCreationOptions
{
    Approaches = new List<string>
    {
        "Static Analysis",
        "Code Reviews",
        "Automated Testing",
        "Refactoring"
    }
};

var root = await service.CreateThoughtTreeAsync("How to improve code quality?", options);

// Evaluate the approaches
await service.EvaluateNodeAsync(root.Children[0], new EvaluationMetrics(0.7, 0.8, 0.6, 0.7));
await service.EvaluateNodeAsync(root.Children[1], new EvaluationMetrics(0.9, 0.7, 0.8, 0.9));
await service.EvaluateNodeAsync(root.Children[2], new EvaluationMetrics(0.8, 0.9, 0.9, 0.7));
await service.EvaluateNodeAsync(root.Children[3], new EvaluationMetrics(0.7, 0.6, 0.8, 0.9));

// Select the best node
var bestNode = await service.SelectBestNodeAsync(root);
Console.WriteLine($"Best approach: {bestNode.Thought}");

// Generate a report
var report = await service.GenerateReportAsync(root, "Code Quality Improvement");
Console.WriteLine(report);
```

#### New Code

```csharp
using TarsEngine.Services.Abstractions.TreeOfThought;
using TarsEngine.FSharp.Adapters.TreeOfThought;

// Create the service
var service = new FSharpTreeOfThoughtService(logger);

// Create a thought tree
var options = new TreeCreationOptions
{
    Approaches = new List<string>
    {
        "Static Analysis",
        "Code Reviews",
        "Automated Testing",
        "Refactoring"
    },
    ApproachEvaluations = new Dictionary<string, EvaluationMetrics>
    {
        ["Static Analysis"] = new EvaluationMetrics(0.7, 0.8, 0.6, 0.7, 0.7),
        ["Code Reviews"] = new EvaluationMetrics(0.9, 0.7, 0.8, 0.9, 0.825),
        ["Automated Testing"] = new EvaluationMetrics(0.8, 0.9, 0.9, 0.7, 0.825),
        ["Refactoring"] = new EvaluationMetrics(0.7, 0.6, 0.8, 0.9, 0.75)
    }
};

var root = await service.CreateThoughtTreeAsync("How to improve code quality?", options);

// Select the best node
var bestNode = await service.SelectBestNodeAsync(root);
Console.WriteLine($"Best approach: {bestNode.Thought} (Score: {bestNode.Score})");

// Generate a report
var report = await service.GenerateReportAsync(root, "Code Quality Improvement");
Console.WriteLine(report);
```

Note that the new code is more concise because the `TreeCreationOptions` class now supports specifying evaluations directly, rather than having to evaluate each node separately.
