#!/usr/bin/env dotnet fsi

// TARS GREEN BASELINE REDEPLOYMENT SYSTEM
// Critical: Green nodes MUST be working 24/7 as stable baseline

open System
open System.IO
open System.Diagnostics

type GreenNodeStatus = Working | Broken | Missing | Redeploying

type GreenNodeHealth = {
    Path: string
    Name: string
    Status: GreenNodeStatus
    ErrorMessage: string option
    LastTested: DateTime
    ContainerName: string option
}

let testGreenNode (greenPath: string) : GreenNodeHealth =
    let name = Path.GetFileName(greenPath)
    printfn "🧪 [CRITICAL] Testing green node: %s" name
    
    try
        if not (Directory.Exists(greenPath)) then
            {
                Path = greenPath
                Name = name
                Status = Missing
                ErrorMessage = Some "Directory does not exist"
                LastTested = DateTime.UtcNow
            }
        else
            let startInfo = ProcessStartInfo()
            startInfo.FileName <- "dotnet"
            startInfo.Arguments <- "build"
            startInfo.WorkingDirectory <- greenPath
            startInfo.RedirectStandardOutput <- true
            startInfo.RedirectStandardError <- true
            startInfo.UseShellExecute <- false
            
            use proc = Process.Start(startInfo)
            proc.WaitForExit(15000) |> ignore
            
            if proc.ExitCode = 0 then
                // Test execution
                let runInfo = ProcessStartInfo()
                runInfo.FileName <- "dotnet"
                runInfo.Arguments <- "run"
                runInfo.WorkingDirectory <- greenPath
                runInfo.RedirectStandardOutput <- true
                runInfo.RedirectStandardError <- true
                runInfo.UseShellExecute <- false
                
                use runProc = Process.Start(runInfo)
                runProc.WaitForExit(10000) |> ignore
                
                if runProc.ExitCode = 0 then
                    printfn "  ✅ %s - WORKING" name
                    {
                        Path = greenPath
                        Name = name
                        Status = Working
                        ErrorMessage = None
                        LastTested = DateTime.UtcNow
                    }
                else
                    let error = runProc.StandardError.ReadToEnd()
                    printfn "  ❌ %s - RUNTIME FAILURE" name
                    printfn "     Error: %s" error
                    {
                        Path = greenPath
                        Name = name
                        Status = Broken
                        ErrorMessage = Some ("Runtime error: " + error)
                        LastTested = DateTime.UtcNow
                    }
            else
                let error = proc.StandardError.ReadToEnd()
                printfn "  ❌ %s - COMPILATION FAILURE" name
                printfn "     Error: %s" error
                {
                    Path = greenPath
                    Name = name
                    Status = Broken
                    ErrorMessage = Some ("Compilation error: " + error)
                    LastTested = DateTime.UtcNow
                }
    with
    | ex ->
        printfn "  ❌ %s - EXCEPTION: %s" name ex.Message
        {
            Path = greenPath
            Name = name
            Status = Broken
            ErrorMessage = Some ex.Message
            LastTested = DateTime.UtcNow
        }

printfn "🚨 CRITICAL: GREEN NODE HEALTH CHECK"
printfn "===================================="
printfn "🔒 Green nodes are our stable baseline - they MUST work!"
printfn ""

// Find all green nodes
let greenDirs = [
    ".tars/green/required"
    ".tars/green/promoted" 
    ".tars/green/versions"
]

let allGreenNodes = 
    greenDirs
    |> List.collect (fun dir ->
        if Directory.Exists(dir) then
            Directory.GetDirectories(dir) |> Array.toList
        else
            [])

printfn "🔍 Found %d green nodes to test..." allGreenNodes.Length
printfn ""

// Test all green nodes
let healthResults = allGreenNodes |> List.map testGreenNode

printfn ""
printfn "📊 GREEN NODE HEALTH REPORT"
printfn "==========================="

let workingNodes = healthResults |> List.filter (fun h -> h.Status = Working)
let brokenNodes = healthResults |> List.filter (fun h -> h.Status = Broken)
let missingNodes = healthResults |> List.filter (fun h -> h.Status = Missing)

printfn "✅ Working Green Nodes: %d" workingNodes.Length
printfn "❌ Broken Green Nodes: %d" brokenNodes.Length
printfn "🚫 Missing Green Nodes: %d" missingNodes.Length

if brokenNodes.Length > 0 then
    printfn ""
    printfn "🚨 CRITICAL ALERT: BROKEN GREEN NODES DETECTED!"
    printfn "=============================================="
    brokenNodes |> List.iter (fun node ->
        printfn "❌ %s" node.Name
        printfn "   Path: %s" node.Path
        match node.ErrorMessage with
        | Some error -> printfn "   Error: %s" error
        | None -> printfn "   Error: Unknown"
        printfn "")

if missingNodes.Length > 0 then
    printfn ""
    printfn "🚫 MISSING GREEN NODES:"
    printfn "======================"
    missingNodes |> List.iter (fun node ->
        printfn "🚫 %s - %s" node.Name node.Path)

printfn ""
printfn "🎯 SYSTEM INTEGRITY STATUS"
printfn "=========================="

let systemHealthy = brokenNodes.Length = 0 && missingNodes.Length = 0 && workingNodes.Length > 0

if systemHealthy then
    printfn "✅ SYSTEM HEALTHY - All green nodes operational!"
    printfn "🔒 Stable baseline maintained"
else
    printfn "🚨 SYSTEM COMPROMISED - Green baseline integrity violated!"
    printfn "🔧 IMMEDIATE ACTION REQUIRED:"
    printfn "   1. Fix all broken green nodes"
    printfn "   2. Restore missing green nodes"
    printfn "   3. Verify system stability"
    printfn "   4. Block blue node operations until green baseline restored"

printfn ""
printfn "📋 DETAILED RESULTS:"
healthResults |> List.iter (fun result ->
    let statusIcon = match result.Status with
                     | Working -> "✅"
                     | Broken -> "❌"
                     | Missing -> "🚫"
    printfn "%s %s (%s)" statusIcon result.Name result.Path)

printfn ""
printfn "🔒 Green nodes are the foundation of TARS system integrity!"
printfn "🎯 They must be working 100% of the time!"
