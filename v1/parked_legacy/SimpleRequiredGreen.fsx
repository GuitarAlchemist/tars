#!/usr/bin/env dotnet fsi

// SIMPLE REQUIRED GREEN NODE SYSTEM
// Policy: Must have stable green baseline before blue experiments

open System
open System.IO

type SystemState = {
    HasRequiredGreen: bool
    GreenNodes: string list
    BlueNodes: string list
}

let mutable systemState = {
    HasRequiredGreen = false
    GreenNodes = []
    BlueNodes = []
}

let createRequiredGreenNode (name: string) =
    printfn "🟢 [REQUIRED] Creating required green node: %s" name
    
    let greenPath = Path.Combine(".tars", "green", "required", name)
    Directory.CreateDirectory(greenPath) |> ignore
    Directory.CreateDirectory(Path.Combine(greenPath, "src")) |> ignore
    
    let stableCode = """open System

type Task = { Id: int; Title: string; Status: string }

let mutable tasks = []

let addTask title =
    let id = List.length tasks + 1
    let task = { Id = id; Title = title; Status = "Pending" }
    tasks <- task :: tasks
    printfn "✅ [GREEN] Added stable task: %s" title

let completeTask id =
    tasks <- tasks |> List.map (fun t ->
        if t.Id = id then { t with Status = "Completed" }
        else t)
    printfn "🎉 [GREEN] Completed task %d" id

[<EntryPoint>]
let main argv =
    printfn "🟢 REQUIRED GREEN NODE - Stable Baseline"
    printfn "========================================"
    printfn "🔒 This is the required stable system"
    printfn ""
    
    addTask "System architecture review"
    addTask "Code quality audit"
    addTask "Performance optimization"
    
    completeTask 1
    
    printfn ""
    printfn "📊 Tasks: %d total" tasks.Length
    printfn "✅ [GREEN] Required baseline operational!"
    printfn "🎯 [SYSTEM] Ready for blue experiments"
    0
"""
    
    let projectFile = sprintf """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <Compile Include="src/Program.fs" />
  </ItemGroup>
</Project>"""
    
    File.WriteAllText(Path.Combine(greenPath, "src", "Program.fs"), stableCode)
    File.WriteAllText(Path.Combine(greenPath, name + ".fsproj"), projectFile)
    
    systemState <- {
        systemState with
            HasRequiredGreen = true
            GreenNodes = greenPath :: systemState.GreenNodes
    }
    
    printfn "✅ [REQUIRED] Green baseline created: %s" greenPath
    greenPath

let createBlueNode (name: string) =
    if not systemState.HasRequiredGreen then
        printfn "🚫 [POLICY] Cannot create blue node - no stable green baseline!"
        printfn "🔧 [ACTION] Create required green node first"
        None
    else
        printfn "🔵 [BLUE] Creating experimental blue node: %s" name
        
        let bluePath = Path.Combine(".tars", "blue", "experimental", name)
        Directory.CreateDirectory(bluePath) |> ignore
        Directory.CreateDirectory(Path.Combine(bluePath, "src")) |> ignore
        
        let experimentalCode = """open System

type ExperimentalTask = { Id: int; Title: string; Experimental: bool }

let mutable tasks = []

let addExperimentalTask title =
    let id = List.length tasks + 1
    let task = { Id = id; Title = title; Experimental = true }
    tasks <- task :: tasks
    printfn "🔵 [BLUE] Added experimental task: %s" title

[<EntryPoint>]
let main argv =
    printfn "🔵 EXPERIMENTAL BLUE NODE"
    printfn "========================="
    printfn "⚠️ This is experimental code!"
    
    addExperimentalTask "Test new feature"
    addExperimentalTask "Prototype UI"
    
    printfn "📋 Experimental tasks: %d" tasks.Length
    printfn "🔵 [BLUE] Experimental node operational"
    0
"""
        
        let projectFile = sprintf """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <Compile Include="src/Program.fs" />
  </ItemGroup>
</Project>"""
        
        File.WriteAllText(Path.Combine(bluePath, "src", "Program.fs"), experimentalCode)
        File.WriteAllText(Path.Combine(bluePath, name + ".fsproj"), projectFile)
        
        systemState <- {
            systemState with
                BlueNodes = bluePath :: systemState.BlueNodes
        }
        
        printfn "✅ [BLUE] Experimental node created: %s" bluePath
        Some bluePath

// Demo: Required Green Node System
printfn "🟢🔵 REQUIRED GREEN NODE SYSTEM"
printfn "==============================="
printfn "🎯 Policy: Must have stable green baseline before blue experiments"
printfn ""

// Try to create blue node without green baseline (should fail)
printfn "🧪 [TEST] Attempting to create blue node without green baseline..."
match createBlueNode "ExperimentalFeature" with
| Some _ -> printfn "❌ [ERROR] Blue node created without green baseline!"
| None -> printfn "✅ [POLICY] Correctly blocked blue node creation"

printfn ""

// Create required green node
printfn "🟢 [SETUP] Creating required green baseline..."
let requiredGreen = createRequiredGreenNode "StableTaskManager"

printfn ""

// Now try to create blue node (should succeed)
printfn "🧪 [TEST] Attempting to create blue node with green baseline..."
match createBlueNode "ExperimentalFeature" with
| Some bluePath -> printfn "✅ [SUCCESS] Blue node created: %s" bluePath
| None -> printfn "❌ [ERROR] Blue node creation failed"

printfn ""

// Test the green node
printfn "🧪 [TEST] Testing required green node..."
let greenTestResult = 
    try
        let startInfo = System.Diagnostics.ProcessStartInfo()
        startInfo.FileName <- "dotnet"
        startInfo.Arguments <- "run"
        startInfo.WorkingDirectory <- requiredGreen
        startInfo.RedirectStandardOutput <- true
        startInfo.UseShellExecute <- false
        
        use proc = System.Diagnostics.Process.Start(startInfo)
        proc.WaitForExit(5000) |> ignore
        
        if proc.ExitCode = 0 then
            let output = proc.StandardOutput.ReadToEnd()
            printfn "✅ [GREEN] Green node test passed!"
            printfn "📋 Output preview: %s" (output.Split('\n').[0])
            true
        else
            printfn "❌ [GREEN] Green node test failed!"
            false
    with
    | ex ->
        printfn "❌ [GREEN] Green node test error: %s" ex.Message
        false

printfn ""

// System summary
printfn "📊 [SUMMARY] System State"
printfn "========================"
printfn "🟢 Required Green Baseline: %s" (if systemState.HasRequiredGreen then "✅ ACTIVE" else "❌ MISSING")
printfn "🟢 Green Nodes: %d" systemState.GreenNodes.Length
printfn "🔵 Blue Nodes: %d" systemState.BlueNodes.Length
printfn "🔒 Green Node Functional: %s" (if greenTestResult then "✅ YES" else "❌ NO")

printfn ""
printfn "✅ Required green node system operational!"
printfn "🎯 Policy enforced: Stable green baseline required for blue experiments!"
printfn "🔒 System integrity maintained!"
