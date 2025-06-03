#!/usr/bin/env dotnet fsi

// BLUE/GREEN QA PROMOTION SYSTEM
// Tests code on blue nodes, promotes to green when QA passes

open System
open System.IO
open System.Text.Json
open System.Diagnostics

type NodeType = Blue | Green
type QAStatus = Pending | Running | Passed | Failed | Promoted

type BlueNode = {
    Id: string
    Name: string
    ProjectPath: string
    CreatedAt: DateTime
    QAStatus: QAStatus
    QAResults: string list
    PromotionCandidate: bool
}

type GreenNode = {
    Id: string
    Name: string
    ProjectPath: string
    PromotedAt: DateTime
    SourceBlueNodeId: string
    StabilityScore: float
}

type QATest = {
    Name: string
    Description: string
    TestCommand: string
    ExpectedResult: string
    Weight: float
}

type QAResult = {
    TestName: string
    Passed: bool
    Output: string
    Duration: TimeSpan
    Score: float
}

type PromotionDecision = {
    BlueNodeId: string
    ShouldPromote: bool
    OverallScore: float
    Reasoning: string
    QAResults: QAResult list
    Timestamp: DateTime
}

module QATestSuite =
    let standardTests = [
        {
            Name = "compilation_test"
            Description = "Verify code compiles without errors"
            TestCommand = "dotnet build"
            ExpectedResult = "Build succeeded"
            Weight = 0.3
        }
        {
            Name = "execution_test"
            Description = "Verify application runs without crashes"
            TestCommand = "dotnet run"
            ExpectedResult = "TARS successfully generated"
            Weight = 0.25
        }
        {
            Name = "syntax_validation"
            Description = "Check F# syntax and style"
            TestCommand = "dotnet build --verbosity normal"
            ExpectedResult = "0 Warning(s)"
            Weight = 0.15
        }
        {
            Name = "performance_test"
            Description = "Verify reasonable execution time"
            TestCommand = "dotnet run"
            ExpectedResult = "< 5 seconds"
            Weight = 0.15
        }
        {
            Name = "output_validation"
            Description = "Verify expected output patterns"
            TestCommand = "dotnet run"
            ExpectedResult = "task manager|working code"
            Weight = 0.15
        }
    ]

module BlueGreenManager =
    let mutable blueNodes: BlueNode list = []
    let mutable greenNodes: GreenNode list = []
    let mutable qaHistory: PromotionDecision list = []
    
    let createBlueNode (projectPath: string) (name: string) =
        let node = {
            Id = Guid.NewGuid().ToString("N")[..7]
            Name = name
            ProjectPath = projectPath
            CreatedAt = DateTime.UtcNow
            QAStatus = Pending
            QAResults = []
            PromotionCandidate = false
        }
        blueNodes <- node :: blueNodes
        printfn "🔵 [BLUE] Created blue node: %s (%s)" node.Name node.Id
        node
    
    let runQATest (node: BlueNode) (test: QATest) : QAResult =
        printfn "🧪 [QA] Running %s on blue node %s..." test.Name node.Id
        
        let stopwatch = Stopwatch.StartNew()
        
        try
            let startInfo = ProcessStartInfo()
            startInfo.FileName <- "dotnet"
            startInfo.Arguments <- test.TestCommand.Replace("dotnet ", "")
            startInfo.WorkingDirectory <- node.ProjectPath
            startInfo.RedirectStandardOutput <- true
            startInfo.RedirectStandardError <- true
            startInfo.UseShellExecute <- false
            
            use proc = Process.Start(startInfo)
            proc.WaitForExit(10000) |> ignore
            stopwatch.Stop()
            
            let output = proc.StandardOutput.ReadToEnd() + proc.StandardError.ReadToEnd()
            
            let passed = 
                match test.Name with
                | "compilation_test" -> proc.ExitCode = 0 && output.Contains("Build succeeded")
                | "execution_test" -> proc.ExitCode = 0 && output.Contains("TARS successfully generated")
                | "syntax_validation" -> proc.ExitCode = 0 && not (output.Contains("warning"))
                | "performance_test" -> stopwatch.Elapsed.TotalSeconds < 5.0
                | "output_validation" -> output.ToLower().Contains("task manager") || output.ToLower().Contains("working code")
                | _ -> proc.ExitCode = 0
            
            let score = if passed then test.Weight else 0.0
            
            {
                TestName = test.Name
                Passed = passed
                Output = output.Substring(0, min 200 output.Length) + (if output.Length > 200 then "..." else "")
                Duration = stopwatch.Elapsed
                Score = score
            }
        with
        | ex ->
            stopwatch.Stop()
            {
                TestName = test.Name
                Passed = false
                Output = sprintf "Error: %s" ex.Message
                Duration = stopwatch.Elapsed
                Score = 0.0
            }
    
    let runFullQA (node: BlueNode) : PromotionDecision =
        printfn "🔍 [QA] Starting full QA on blue node %s (%s)" node.Name node.Id
        printfn "======================================================="
        
        // Update node status
        let updatedNode = { node with QAStatus = Running }
        blueNodes <- blueNodes |> List.map (fun n -> if n.Id = node.Id then updatedNode else n)
        
        // Run all tests
        let qaResults = QATestSuite.standardTests |> List.map (runQATest updatedNode)
        
        // Calculate overall score
        let overallScore = qaResults |> List.sumBy (fun r -> r.Score)
        let passedTests = qaResults |> List.filter (fun r -> r.Passed) |> List.length
        let totalTests = qaResults.Length
        
        // Determine promotion decision
        let shouldPromote = overallScore >= 0.8 && passedTests >= 4
        
        let reasoning = 
            if shouldPromote then
                sprintf "PROMOTION APPROVED: Score %.2f/1.0, %d/%d tests passed. Meets promotion criteria." overallScore passedTests totalTests
            else
                sprintf "PROMOTION DENIED: Score %.2f/1.0, %d/%d tests passed. Below promotion threshold (0.8 score, 4+ tests)." overallScore passedTests totalTests
        
        let decision = {
            BlueNodeId = node.Id
            ShouldPromote = shouldPromote
            OverallScore = overallScore
            Reasoning = reasoning
            QAResults = qaResults
            Timestamp = DateTime.UtcNow
        }
        
        // Update node status
        let finalStatus = if shouldPromote then Passed else Failed
        let finalNode = { 
            updatedNode with 
                QAStatus = finalStatus
                QAResults = qaResults |> List.map (fun r -> sprintf "%s: %s" r.TestName (if r.Passed then "✅" else "❌"))
                PromotionCandidate = shouldPromote
        }
        blueNodes <- blueNodes |> List.map (fun n -> if n.Id = node.Id then finalNode else n)
        
        // Store decision
        qaHistory <- decision :: qaHistory
        
        printfn ""
        printfn "📊 [QA] QA RESULTS FOR BLUE NODE %s" node.Id
        printfn "======================================"
        qaResults |> List.iter (fun r ->
            let status = if r.Passed then "✅" else "❌"
            printfn "  %s %s (%.2fs, score: %.2f)" status r.TestName r.Duration.TotalSeconds r.Score)
        
        printfn ""
        printfn "🎯 [DECISION] %s" reasoning
        printfn "📈 Overall Score: %.2f/1.0" overallScore
        printfn "✅ Tests Passed: %d/%d" passedTests totalTests
        
        decision
    
    let promoteToGreen (blueNodeId: string) : GreenNode option =
        match blueNodes |> List.tryFind (fun n -> n.Id = blueNodeId && n.PromotionCandidate) with
        | Some blueNode ->
            let greenId = Guid.NewGuid().ToString("N")[..7]
            let greenPath = Path.Combine(".tars", "green", sprintf "Green_%s_%s" blueNode.Name greenId)
            
            // Copy blue node to green location
            Directory.CreateDirectory(greenPath) |> ignore
            
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

            copyDirectory blueNode.ProjectPath greenPath
            
            let greenNode = {
                Id = greenId
                Name = sprintf "Green_%s" blueNode.Name
                ProjectPath = greenPath
                PromotedAt = DateTime.UtcNow
                SourceBlueNodeId = blueNode.Id
                StabilityScore = 0.95 // Initial high stability for promoted code
            }
            
            greenNodes <- greenNode :: greenNodes
            
            // Update blue node status
            let promotedBlueNode = { blueNode with QAStatus = Promoted }
            blueNodes <- blueNodes |> List.map (fun n -> if n.Id = blueNode.Id then promotedBlueNode else n)
            
            printfn "🟢 [GREEN] PROMOTED blue node %s to green node %s" blueNode.Id greenNode.Id
            printfn "📁 Green location: %s" greenPath
            
            Some greenNode
        | None ->
            printfn "❌ [ERROR] Cannot promote blue node %s - not found or not promotion candidate" blueNodeId
            None
    
    let getBlueNodes() = blueNodes
    let getGreenNodes() = greenNodes
    let getQAHistory() = qaHistory

// Demo: Create blue nodes and run QA
printfn "🔵🟢 BLUE/GREEN QA PROMOTION SYSTEM"
printfn "=================================="
printfn ""

// Create some blue nodes from existing projects
let blueNode1 = BlueGreenManager.createBlueNode ".tars/projects/TaskManager_20250601_222344" "TaskManager_Basic"
let blueNode2 = BlueGreenManager.createBlueNode ".tars/projects/TracedTaskManager_20250601_223127" "TaskManager_Traced"

printfn ""
printfn "🧪 [QA] Starting QA process on blue nodes..."
printfn "============================================"

// Run QA on blue node 1
printfn ""
let decision1 = BlueGreenManager.runFullQA blueNode1

// Run QA on blue node 2  
printfn ""
let decision2 = BlueGreenManager.runFullQA blueNode2

printfn ""
printfn "🎯 [PROMOTION] Processing promotion decisions..."
printfn "==============================================="

// Promote qualifying blue nodes to green
if decision1.ShouldPromote then
    match BlueGreenManager.promoteToGreen decision1.BlueNodeId with
    | Some greenNode -> printfn "✅ Successfully promoted %s to green!" greenNode.Name
    | None -> printfn "❌ Failed to promote blue node"

if decision2.ShouldPromote then
    match BlueGreenManager.promoteToGreen decision2.BlueNodeId with
    | Some greenNode -> printfn "✅ Successfully promoted %s to green!" greenNode.Name
    | None -> printfn "❌ Failed to promote blue node"

printfn ""
printfn "📊 [SUMMARY] Blue/Green Status"
printfn "=============================="
printfn "🔵 Blue Nodes: %d" (BlueGreenManager.getBlueNodes().Length)
BlueGreenManager.getBlueNodes() |> List.iter (fun node ->
    printfn "   %s (%s) - Status: %A" node.Name node.Id node.QAStatus)

printfn "🟢 Green Nodes: %d" (BlueGreenManager.getGreenNodes().Length)
BlueGreenManager.getGreenNodes() |> List.iter (fun node ->
    printfn "   %s (%s) - Promoted: %s" node.Name node.Id (node.PromotedAt.ToString("yyyy-MM-dd HH:mm")))

printfn ""
printfn "✅ Blue/Green QA promotion system operational!"
printfn "🎯 Only code that passes comprehensive QA gets promoted to stable green nodes!"
