#!/usr/bin/env dotnet fsi

// BLUE TO GREEN PROMOTION SYSTEM
// QA validation and promotion of blue nodes to green status

open System
open System.IO
open System.Diagnostics

type QATest = {
    Name: string
    Command: string
    ExpectedPattern: string
    Weight: float
}

type QAResult = {
    TestName: string
    Passed: bool
    Output: string
    Score: float
}

type PromotionDecision = {
    BlueNodePath: string
    OverallScore: float
    ShouldPromote: bool
    Reasoning: string
    QAResults: QAResult list
}

let qaTests = [
    { Name = "compilation"; Command = "dotnet build"; ExpectedPattern = "Build succeeded"; Weight = 0.4 }
    { Name = "execution"; Command = "dotnet run"; ExpectedPattern = "operational"; Weight = 0.3 }
    { Name = "output_quality"; Command = "dotnet run"; ExpectedPattern = "BLUE|GREEN"; Weight = 0.2 }
    { Name = "performance"; Command = "dotnet run"; ExpectedPattern = ""; Weight = 0.1 }
]

let runQATest (blueNodePath: string) (test: QATest) : QAResult =
    printfn "🧪 [QA] Running %s test..." test.Name
    
    try
        let startInfo = ProcessStartInfo()
        startInfo.FileName <- "dotnet"
        startInfo.Arguments <- test.Command.Replace("dotnet ", "")
        startInfo.WorkingDirectory <- blueNodePath
        startInfo.RedirectStandardOutput <- true
        startInfo.RedirectStandardError <- true
        startInfo.UseShellExecute <- false
        
        let stopwatch = Stopwatch.StartNew()
        use proc = Process.Start(startInfo)
        proc.WaitForExit(10000) |> ignore
        stopwatch.Stop()
        
        let output = proc.StandardOutput.ReadToEnd() + proc.StandardError.ReadToEnd()
        
        let passed = 
            match test.Name with
            | "compilation" -> proc.ExitCode = 0 && output.Contains("Build succeeded")
            | "execution" -> proc.ExitCode = 0 && output.ToLower().Contains("operational")
            | "output_quality" -> output.Contains("BLUE") || output.Contains("GREEN")
            | "performance" -> stopwatch.Elapsed.TotalSeconds < 5.0
            | _ -> proc.ExitCode = 0
        
        let score = if passed then test.Weight else 0.0
        
        printfn "  %s %s (%.2fs, score: %.2f)" 
            (if passed then "✅" else "❌") test.Name stopwatch.Elapsed.TotalSeconds score
        
        {
            TestName = test.Name
            Passed = passed
            Output = output.Substring(0, min 100 output.Length)
            Score = score
        }
    with
    | ex ->
        printfn "  ❌ %s (error: %s)" test.Name ex.Message
        { TestName = test.Name; Passed = false; Output = ex.Message; Score = 0.0 }

let runFullQA (blueNodePath: string) : PromotionDecision =
    printfn "🔍 [QA] Starting full QA on blue node: %s" (Path.GetFileName(blueNodePath))
    printfn "=================================================="
    
    let qaResults = qaTests |> List.map (runQATest blueNodePath)
    let overallScore = qaResults |> List.sumBy (fun r -> r.Score)
    let passedTests = qaResults |> List.filter (fun r -> r.Passed) |> List.length
    
    let shouldPromote = overallScore >= 0.8 && passedTests >= 3
    
    let reasoning = 
        if shouldPromote then
            sprintf "PROMOTION APPROVED: Score %.2f/1.0, %d/%d tests passed. Meets promotion criteria." 
                overallScore passedTests qaTests.Length
        else
            sprintf "PROMOTION DENIED: Score %.2f/1.0, %d/%d tests passed. Below threshold (0.8 score, 3+ tests)." 
                overallScore passedTests qaTests.Length
    
    printfn ""
    printfn "📊 [QA] RESULTS SUMMARY"
    printfn "======================="
    printfn "Overall Score: %.2f/1.0" overallScore
    printfn "Tests Passed: %d/%d" passedTests qaTests.Length
    printfn "Decision: %s" (if shouldPromote then "✅ PROMOTE" else "❌ REJECT")
    printfn "Reasoning: %s" reasoning
    
    {
        BlueNodePath = blueNodePath
        OverallScore = overallScore
        ShouldPromote = shouldPromote
        Reasoning = reasoning
        QAResults = qaResults
    }

let promoteBlueToGreen (blueNodePath: string) : string option =
    let blueNodeName = Path.GetFileName(blueNodePath)
    let greenNodeName = sprintf "Green_%s_%s" blueNodeName (DateTime.Now.ToString("HHmmss"))
    let greenNodePath = Path.Combine(".tars", "green", "promoted", greenNodeName)
    
    try
        Directory.CreateDirectory(greenNodePath) |> ignore
        
        // Copy all files from blue to green
        let rec copyDirectory source target =
            let sourceDir = DirectoryInfo(source)
            let targetDir = DirectoryInfo(target)
            
            if not targetDir.Exists then targetDir.Create()
            
            sourceDir.GetFiles() |> Array.iter (fun file ->
                file.CopyTo(Path.Combine(target, file.Name), true) |> ignore)
            
            sourceDir.GetDirectories() |> Array.iter (fun dir ->
                let newTargetDir = Path.Combine(target, dir.Name)
                copyDirectory dir.FullName newTargetDir)
        
        copyDirectory blueNodePath greenNodePath
        
        // Update the code to reflect green status
        let programPath = Path.Combine(greenNodePath, "src", "Program.fs")
        if File.Exists(programPath) then
            let content = File.ReadAllText(programPath)
            let updatedContent = content.Replace("🔵 EXPERIMENTAL BLUE NODE", "🟢 PROMOTED GREEN NODE")
                                       .Replace("⚠️ This is experimental code!", "✅ This is stable, promoted code!")
                                       .Replace("[BLUE]", "[GREEN]")
            File.WriteAllText(programPath, updatedContent)
        
        printfn "🟢 [PROMOTION] Successfully promoted to green: %s" greenNodePath
        Some greenNodePath
    with
    | ex ->
        printfn "❌ [PROMOTION] Failed to promote: %s" ex.Message
        None

// Demo: QA and Promotion Process
printfn "🔵➡️🟢 BLUE TO GREEN PROMOTION SYSTEM"
printfn "===================================="
printfn ""

// Find existing blue nodes
let blueNodesDir = Path.Combine(".tars", "blue", "experimental")
if Directory.Exists(blueNodesDir) then
    let blueNodes = Directory.GetDirectories(blueNodesDir)
    
    if blueNodes.Length > 0 then
        printfn "🔍 [DISCOVERY] Found %d blue node(s) for QA evaluation" blueNodes.Length
        
        for blueNode in blueNodes do
            printfn ""
            printfn "🧪 [QA] Evaluating blue node: %s" (Path.GetFileName(blueNode))
            
            let decision = runFullQA blueNode
            
            if decision.ShouldPromote then
                printfn ""
                printfn "🎯 [PROMOTION] Promoting blue node to green..."
                match promoteBlueToGreen blueNode with
                | Some greenPath ->
                    printfn "✅ [SUCCESS] Blue node promoted to green!"
                    printfn "📁 Green location: %s" greenPath
                    
                    // Test the promoted green node
                    printfn ""
                    printfn "🧪 [VALIDATION] Testing promoted green node..."
                    try
                        let startInfo = ProcessStartInfo()
                        startInfo.FileName <- "dotnet"
                        startInfo.Arguments <- "run"
                        startInfo.WorkingDirectory <- greenPath
                        startInfo.RedirectStandardOutput <- true
                        startInfo.UseShellExecute <- false
                        
                        use proc = Process.Start(startInfo)
                        proc.WaitForExit(5000) |> ignore
                        
                        if proc.ExitCode = 0 then
                            let output = proc.StandardOutput.ReadToEnd()
                            printfn "✅ [GREEN] Promoted node validation passed!"
                            printfn "📋 Output: %s" (output.Split('\n').[0])
                        else
                            printfn "❌ [GREEN] Promoted node validation failed!"
                    with
                    | ex -> printfn "❌ [GREEN] Validation error: %s" ex.Message
                    
                | None ->
                    printfn "❌ [FAILURE] Failed to promote blue node"
            else
                printfn ""
                printfn "🚫 [REJECTION] Blue node does not meet promotion criteria"
                printfn "🔧 [ACTION] Improve blue node and retry QA"
    else
        printfn "ℹ️ [INFO] No blue nodes found for evaluation"
        printfn "🔧 [ACTION] Create blue nodes first using the required green system"
else
    printfn "ℹ️ [INFO] Blue nodes directory does not exist"
    printfn "🔧 [ACTION] Run the required green system first to create blue nodes"

printfn ""
printfn "📊 [SUMMARY] Promotion System Status"
printfn "===================================="

// Count green and blue nodes
let greenRequiredDir = Path.Combine(".tars", "green", "required")
let greenPromotedDir = Path.Combine(".tars", "green", "promoted")
let blueExperimentalDir = Path.Combine(".tars", "blue", "experimental")

let countNodes dir = if Directory.Exists(dir) then Directory.GetDirectories(dir).Length else 0

let requiredGreenCount = countNodes greenRequiredDir
let promotedGreenCount = countNodes greenPromotedDir
let blueCount = countNodes blueExperimentalDir

printfn "🟢 Required Green Nodes: %d" requiredGreenCount
printfn "🟢 Promoted Green Nodes: %d" promotedGreenCount
printfn "🔵 Blue Nodes: %d" blueCount
printfn "📈 Total Green Nodes: %d" (requiredGreenCount + promotedGreenCount)

printfn ""
printfn "✅ Blue to Green promotion system operational!"
printfn "🎯 QA validation ensures only quality code gets promoted!"
printfn "🔒 System maintains stable green baseline with quality promotions!"
