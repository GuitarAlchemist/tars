# Script to run a simple F# Tree-of-Thought test

Write-Host "Starting simple F# Tree-of-Thought test..."

# Generate F# code
$fsharpCode = @'
// Simple Tree-of-Thought implementation in F#
printfn "Tree-of-Thought Reasoning Example"

// Define the evaluation metrics
type EvaluationMetrics = {
    Correctness: float
    Efficiency: float
    Elegance: float
    Maintainability: float
    Overall: float
}

// Define the thought node type
type ThoughtNode = {
    Thought: string
    Evaluation: EvaluationMetrics option
    Pruned: bool
}

// Create a new thought node
let createThought thought =
    {
        Thought = thought
        Evaluation = None
        Pruned = false
    }

// Evaluate a thought node
let evaluateThought thought metrics =
    let overall = (metrics.Correctness + metrics.Efficiency + metrics.Elegance + metrics.Maintainability) / 4.0
    let evaluation = { metrics with Overall = overall }
    { thought with Evaluation = Some evaluation }

// Prune a thought node
let pruneThought thought =
    { thought with Pruned = true }

// Create some example thoughts
let root = createThought "Initial planning for problem solving"
let approach1 = createThought "Approach 1: Divide and conquer"
let approach2 = createThought "Approach 2: Dynamic programming"
let approach3 = createThought "Approach 3: Greedy algorithm"

// Evaluate the thoughts
let approach1Evaluated = evaluateThought approach1 { Correctness = 0.9; Efficiency = 0.8; Elegance = 0.9; Maintainability = 0.8; Overall = 0.0 }
let approach2Evaluated = evaluateThought approach2 { Correctness = 0.7; Efficiency = 0.9; Elegance = 0.6; Maintainability = 0.7; Overall = 0.0 }
let approach3Evaluated = evaluateThought approach3 { Correctness = 0.6; Efficiency = 0.7; Elegance = 0.5; Maintainability = 0.6; Overall = 0.0 }

// Prune the least promising approach
let approach3Pruned = pruneThought approach3Evaluated

// Print the thoughts
printfn "Root thought: %s" root.Thought

printfn "\nApproach 1: %s" approach1Evaluated.Thought
match approach1Evaluated.Evaluation with
| Some e -> printfn "  Score: %.2f" e.Overall
| None -> printfn "  Not evaluated"

printfn "\nApproach 2: %s" approach2Evaluated.Thought
match approach2Evaluated.Evaluation with
| Some e -> printfn "  Score: %.2f" e.Overall
| None -> printfn "  Not evaluated"

printfn "\nApproach 3: %s%s" approach3Pruned.Thought (if approach3Pruned.Pruned then " (Pruned)" else "")
match approach3Pruned.Evaluation with
| Some e -> printfn "  Score: %.2f" e.Overall
| None -> printfn "  Not evaluated"

printfn "\nTree-of-Thought reasoning completed successfully!"
'@

# Save the F# code to a file
$fsharpFilePath = "SimpleTreeOfThought.fsx"
$fsharpCode | Out-File -FilePath $fsharpFilePath -Encoding utf8

Write-Host "F# code generated and saved to $fsharpFilePath"

# Compile and run the F# code
Write-Host "Running F# code..."

try {
    # Check if dotnet and F# compiler are available
    $dotnetVersion = dotnet --version
    Write-Host "Using .NET version: $dotnetVersion"
    
    # Run the F# script
    $output = dotnet fsi $fsharpFilePath 2>&1
    $success = $?
    
    if ($success) {
        Write-Host "Execution succeeded"
        Write-Host "Output:"
        Write-Host $output
    } else {
        Write-Host "Execution failed"
        Write-Host "Error output:"
        Write-Host $output
    }
} catch {
    Write-Host "Error running F# code: $_"
    $success = $false
}

Write-Host "Simple F# Tree-of-Thought test completed"
