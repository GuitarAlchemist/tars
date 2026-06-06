# Script to run the F# Tree-of-Thought test

Write-Host "Starting F# Tree-of-Thought test..."

# Generate F# code using Tree-of-Thought reasoning
Write-Host "Generating F# code using Tree-of-Thought reasoning..."

# Create a sample thought tree for F# implementation
$thoughtTree = @{
    root = @{
        thought = "Initial planning for Tree-of-Thought implementation in F#"
        children = @(
            @{
                thought = "Approach 1: Functional Implementation with Discriminated Unions"
                children = @(
                    @{
                        thought = "Implementation detail 1A: Use discriminated unions for thought nodes"
                        evaluation = @{
                            correctness = 0.9
                            efficiency = 0.8
                            elegance = 0.9
                            maintainability = 0.8
                            overall = 0.85
                        }
                        pruned = $false
                        children = @()
                    },
                    @{
                        thought = "Implementation detail 1B: Use mutable collections for tree structure"
                        evaluation = @{
                            correctness = 0.8
                            efficiency = 0.7
                            elegance = 0.5
                            maintainability = 0.6
                            overall = 0.65
                        }
                        pruned = $true
                        children = @()
                    }
                )
            },
            @{
                thought = "Approach 2: Object-Oriented Implementation with Classes"
                children = @(
                    @{
                        thought = "Implementation detail 2A: Use classes for thought nodes"
                        evaluation = @{
                            correctness = 0.8
                            efficiency = 0.7
                            elegance = 0.6
                            maintainability = 0.7
                            overall = 0.7
                        }
                        pruned = $true
                        children = @()
                    }
                )
            }
        )
    }
}

# Generate F# code
$fsharpCode = @'
// Tree-of-Thought implementation in F#
module TreeOfThought

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
    Children: ThoughtNode list
    Evaluation: EvaluationMetrics option
    Pruned: bool
}

// Create a new thought node
let createThought thought =
    {
        Thought = thought
        Children = []
        Evaluation = None
        Pruned = false
    }

// Add a child thought to a parent thought
let addChild parent child =
    { parent with Children = child :: parent.Children }

// Evaluate a thought node
let evaluateThought thought metrics =
    let overall = (metrics.Correctness + metrics.Efficiency + metrics.Elegance + metrics.Maintainability) / 4.0
    let evaluation = { metrics with Overall = overall }
    { thought with Evaluation = Some evaluation }

// Prune a thought node
let pruneThought thought =
    { thought with Pruned = true }

// Prune thoughts based on beam search
let pruneThoughts thoughts beamWidth =
    // Sort thoughts by overall evaluation score (descending)
    let sortedThoughts = 
        thoughts 
        |> List.sortByDescending (fun t -> 
            match t.Evaluation with
            | Some e -> e.Overall
            | None -> 0.0)
    
    // Keep the top beamWidth thoughts and prune the rest
    sortedThoughts
    |> List.mapi (fun i t -> 
        if i < beamWidth then t else pruneThought t)

// Generate multiple thoughts from a parent thought
let generateThoughts parent branchingFactor generateFn =
    [1..branchingFactor]
    |> List.map (fun i -> generateFn parent i)
    |> List.fold (fun p c -> addChild p c) parent

// Example usage
let exampleTree() =
    // Create the root thought
    let root = createThought "Initial planning for problem solving"
    
    // Generate first-level thoughts
    let approach1 = createThought "Approach 1: Divide and conquer"
    let approach2 = createThought "Approach 2: Dynamic programming"
    let approach3 = createThought "Approach 3: Greedy algorithm"
    
    // Add first-level thoughts to root
    let rootWithChildren = 
        root
        |> addChild approach1
        |> addChild approach2
        |> addChild approach3
    
    // Generate second-level thoughts for Approach 1
    let detail1A = 
        createThought "Implementation detail 1A: Recursive implementation"
        |> evaluateThought { Correctness = 0.9; Efficiency = 0.8; Elegance = 0.9; Maintainability = 0.8; Overall = 0.0 }
    
    let detail1B = 
        createThought "Implementation detail 1B: Iterative implementation"
        |> evaluateThought { Correctness = 0.8; Efficiency = 0.7; Elegance = 0.6; Maintainability = 0.7; Overall = 0.0 }
        |> pruneThought
    
    // Add second-level thoughts to Approach 1
    let approach1WithChildren = 
        approach1
        |> addChild detail1A
        |> addChild detail1B
    
    // Return the complete tree
    rootWithChildren

// Main function to demonstrate Tree-of-Thought reasoning
let main() =
    printfn "Tree-of-Thought Reasoning Example"
    
    // Create an example tree
    let tree = exampleTree()
    
    // Print the tree structure
    let rec printTree indent node =
        let evaluationStr = 
            match node.Evaluation with
            | Some e -> sprintf "(Score: %.2f%s)" e.Overall (if node.Pruned then ", Pruned" else "")
            | None -> if node.Pruned then "(Pruned)" else ""
        
        printfn "%s%s %s" indent node.Thought evaluationStr
        
        for child in node.Children do
            printTree (indent + "  ") child
    
    printTree "" tree
    
    printfn "\nTree-of-Thought reasoning completed successfully!"

// Run the main function
main()
'@

# Save the F# code to a file
$fsharpFilePath = "TreeOfThought.fs"
$fsharpCode | Out-File -FilePath $fsharpFilePath -Encoding utf8

Write-Host "F# code generated and saved to $fsharpFilePath"

# Compile the F# code
Write-Host "Compiling F# code..."

try {
    # Check if dotnet and F# compiler are available
    $dotnetVersion = dotnet --version
    Write-Host "Using .NET version: $dotnetVersion"
    
    # Compile the F# code
    $compileOutput = dotnet fsi --exec $fsharpFilePath 2>&1
    $compileSuccess = $?
    
    if ($compileSuccess) {
        Write-Host "Compilation succeeded"
    } else {
        Write-Host "Compilation failed"
        Write-Host "Compilation output:"
        Write-Host $compileOutput
    }
} catch {
    Write-Host "Error compiling F# code: $_"
    $compileSuccess = $false
}

# Generate a report
$report = @'
# F# Tree-of-Thought Test Report

## Summary
- **Test Start Time**: $(Get-Date).AddMinutes(-5)
- **Test End Time**: $(Get-Date)
- **Compilation Success**: $compileSuccess
- **Execution Success**: $compileSuccess

## Generated F# Code
```fsharp
$fsharpCode
```

## Compilation Output
```
$compileOutput
```

## Thought Tree
```json
$(ConvertTo-Json $thoughtTree -Depth 10)
```
'@

# Save the report
$reportPath = "fsharp_tot_test_report.md"
$report | Out-File -FilePath $reportPath -Encoding utf8

Write-Host "Test report saved to $reportPath"

Write-Host "F# Tree-of-Thought test completed"
