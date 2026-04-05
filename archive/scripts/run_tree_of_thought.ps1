# Script to run the F# Tree-of-Thought implementation

Write-Host "Running F# Tree-of-Thought implementation..."

# Create the F# script file
$scriptContent = @'
// Load the Tree-of-Thought module
#load "TarsEngine/FSharp/TreeOfThought.fs"

open TarsEngine.FSharp.TreeOfThought

// Run the example
main()

// Create a more complex example
let complexExample() =
    // Create the root thought
    let root = ThoughtTree.createNode "Code Analysis Problem"
    
    // Generate analysis approaches
    let approaches = Branching.generateAnalysisApproaches "sample code"
    
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
        { rootWithApproaches with Children = evaluatedApproaches }
    
    // Perform beam search
    let prunedTree = 
        Pruning.beamSearch rootWithEvaluatedApproaches 2 Pruning.scoreNode
    
    // Select best approach
    let bestApproach = 
        prunedTree.Children
        |> Selection.bestFirst
    
    // Print the result
    match bestApproach with
    | Some approach -> 
        printfn "\nBest approach: %s" approach.Thought
        match approach.Evaluation with
        | Some eval -> printfn "Score: %.2f" eval.Overall
        | None -> printfn "No evaluation"
    | None -> 
        printfn "\nNo approach selected"
    
    // Return the tree
    prunedTree

// Run the complex example
printfn "\nRunning complex example..."
let complexTree = complexExample()

// Print the tree as JSON
printfn "\nTree as JSON:"
printfn "%s" (ThoughtTree.toJson complexTree)

// Demonstrate Tree-of-Thought for fix generation
let fixGenerationExample() =
    // Create the root thought
    let root = ThoughtTree.createNode "Fix Generation Problem"
    
    // Generate fix approaches
    let approaches = Branching.generateFixApproaches "sample issue"
    
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
        { rootWithApproaches with Children = evaluatedApproaches }
    
    // Perform beam search
    let prunedTree = 
        Pruning.beamSearch rootWithEvaluatedApproaches 2 Pruning.scoreNode
    
    // Select best approach
    let bestApproach = 
        prunedTree.Children
        |> Selection.bestFirst
    
    // Print the result
    match bestApproach with
    | Some approach -> 
        printfn "\nBest fix approach: %s" approach.Thought
        match approach.Evaluation with
        | Some eval -> printfn "Score: %.2f" eval.Overall
        | None -> printfn "No evaluation"
    | None -> 
        printfn "\nNo fix approach selected"
    
    // Return the tree
    prunedTree

// Run the fix generation example
printfn "\nRunning fix generation example..."
let fixTree = fixGenerationExample()

// Demonstrate Tree-of-Thought for fix application
let fixApplicationExample() =
    // Create the root thought
    let root = ThoughtTree.createNode "Fix Application Problem"
    
    // Generate application approaches
    let approaches = Branching.generateApplicationApproaches "sample fix"
    
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
        { rootWithApproaches with Children = evaluatedApproaches }
    
    // Perform beam search
    let prunedTree = 
        Pruning.beamSearch rootWithEvaluatedApproaches 2 Pruning.scoreNode
    
    // Select best approach
    let bestApproach = 
        prunedTree.Children
        |> Selection.bestFirst
    
    // Print the result
    match bestApproach with
    | Some approach -> 
        printfn "\nBest application approach: %s" approach.Thought
        match approach.Evaluation with
        | Some eval -> printfn "Score: %.2f" eval.Overall
        | None -> printfn "No evaluation"
    | None -> 
        printfn "\nNo application approach selected"
    
    // Return the tree
    prunedTree

// Run the fix application example
printfn "\nRunning fix application example..."
let applicationTree = fixApplicationExample()

// Demonstrate different selection strategies
let selectionExample() =
    // Create some nodes with evaluations
    let nodes = [
        { ThoughtTree.createNode "Approach 1" with 
            Evaluation = Some (Evaluation.createAnalysisMetrics 0.9 0.8 0.7 0.9) }
        { ThoughtTree.createNode "Approach 2" with 
            Evaluation = Some (Evaluation.createAnalysisMetrics 0.8 0.9 0.8 0.7) }
        { ThoughtTree.createNode "Approach 3" with 
            Evaluation = Some (Evaluation.createAnalysisMetrics 0.7 0.7 0.9 0.8) }
        { ThoughtTree.createNode "Approach 4" with 
            Evaluation = Some (Evaluation.createAnalysisMetrics 0.6 0.6 0.6 0.6) }
        { ThoughtTree.createNode "Approach 5" with 
            Evaluation = Some (Evaluation.createAnalysisMetrics 0.5 0.5 0.5 0.5) }
    ]
    
    // Apply different selection strategies
    let bestFirst = Selection.bestFirst nodes
    let diversityBased = Selection.diversityBased nodes
    let confidenceBased = Selection.confidenceBased nodes
    let hybrid = Selection.hybridSelection nodes
    
    // Print the results
    printfn "\nSelection Strategies:"
    
    printfn "\nBest-First:"
    match bestFirst with
    | Some node -> 
        printfn "  %s" node.Thought
        match node.Evaluation with
        | Some eval -> printfn "  Score: %.2f" eval.Overall
        | None -> printfn "  No evaluation"
    | None -> 
        printfn "  No node selected"
    
    printfn "\nDiversity-Based:"
    for node in diversityBased do
        printfn "  %s" node.Thought
        match node.Evaluation with
        | Some eval -> printfn "  Score: %.2f" eval.Overall
        | None -> printfn "  No evaluation"
    
    printfn "\nConfidence-Based:"
    for node in confidenceBased do
        printfn "  %s" node.Thought
        match node.Evaluation with
        | Some eval -> 
            printfn "  Score: %.2f" eval.Overall
            match eval.Confidence with
            | Some conf -> printfn "  Confidence: %.2f" conf
            | None -> printfn "  No confidence"
        | None -> 
            printfn "  No evaluation"
    
    printfn "\nHybrid:"
    for node in hybrid do
        printfn "  %s" node.Thought
        match node.Evaluation with
        | Some eval -> printfn "  Score: %.2f" eval.Overall
        | None -> printfn "  No evaluation"

// Run the selection example
printfn "\nRunning selection example..."
selectionExample()

printfn "\nF# Tree-of-Thought implementation completed successfully!"
'@

# Save the script to a file
$scriptPath = "run_tree_of_thought.fsx"
$scriptContent | Out-File -FilePath $scriptPath -Encoding utf8

Write-Host "F# script created at $scriptPath"

# Run the F# script
Write-Host "Running F# script..."
dotnet fsi $scriptPath

Write-Host "F# Tree-of-Thought implementation completed successfully!"
