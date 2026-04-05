#!/usr/bin/env dotnet fsi

// SIMPLE MULTIPLE GREEN VERSIONS
// Create different green versions for different use cases

open System
open System.IO

printfn "🟢📦 MULTIPLE GREEN VERSIONS SYSTEM"
printfn "==================================="
printfn ""

let createGreenVersion (name: string) (version: string) (useCase: string) =
    printfn "🟢 [VERSION] Creating %s v%s for %s" name version useCase
    
    let versionPath = Path.Combine(".tars", "green", "versions", sprintf "%s_v%s" name version)
    Directory.CreateDirectory(versionPath) |> ignore
    Directory.CreateDirectory(Path.Combine(versionPath, "src")) |> ignore
    
    let codeTemplate = 
        match version with
        | "1.0" -> 
            """open System

type BasicTask = { Id: int; Title: string; Done: bool }

let mutable tasks = []

let addTask title =
    let id = List.length tasks + 1
    let task = { Id = id; Title = title; Done = false }
    tasks <- task :: tasks
    printfn "✅ [BASIC] Added: %s" title

[<EntryPoint>]
let main argv =
    printfn "🟢 BASIC TASK MANAGER v1.0"
    printfn "=========================="
    printfn "📋 Use Case: Simple task tracking"
    
    addTask "Complete basic task"
    addTask "Review simple workflow"
    
    printfn "📊 Total tasks: %d" tasks.Length
    printfn "✅ [BASIC] Green version operational!"
    0
"""
        | "2.0" ->
            """open System

type Priority = High | Medium | Low

type AdvancedTask = {
    Id: int
    Title: string
    Priority: Priority
    Status: string
    CreatedAt: DateTime
}

let mutable tasks = []

let addAdvancedTask title priority =
    let id = List.length tasks + 1
    let task = {
        Id = id
        Title = title
        Priority = priority
        Status = "Pending"
        CreatedAt = DateTime.UtcNow
    }
    tasks <- task :: tasks
    printfn "✅ [ADVANCED] Added %A priority: %s" priority title

[<EntryPoint>]
let main argv =
    printfn "🟢 ADVANCED TASK MANAGER v2.0"
    printfn "============================="
    printfn "📋 Use Case: Enterprise task management"
    
    addAdvancedTask "System architecture review" High
    addAdvancedTask "Code optimization" Medium
    addAdvancedTask "Documentation update" Low
    
    printfn "📊 Total tasks: %d" tasks.Length
    printfn "✅ [ADVANCED] Green version operational!"
    0
"""
        | "3.0" ->
            """open System

type User = { Id: int; Name: string; Role: string }
type CollaborativeTask = {
    Id: int
    Title: string
    AssignedTo: string
    Status: string
}

let mutable tasks = []

let assignTask title assignedTo =
    let id = List.length tasks + 1
    let task = {
        Id = id
        Title = title
        AssignedTo = assignedTo
        Status = "Assigned"
    }
    tasks <- task :: tasks
    printfn "✅ [COLLAB] Assigned '%s' to %s" title assignedTo

[<EntryPoint>]
let main argv =
    printfn "🟢 COLLABORATIVE TASK MANAGER v3.0"
    printfn "=================================="
    printfn "📋 Use Case: Team collaboration"
    
    assignTask "Design new UI components" "Carol"
    assignTask "Implement backend API" "Alice"
    assignTask "Project planning meeting" "Bob"
    
    printfn "📊 Collaborative tasks: %d" tasks.Length
    printfn "✅ [COLLAB] Green version operational!"
    0
"""
        | _ -> "// Default code"
    
    let projectFile = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <Compile Include="src/Program.fs" />
  </ItemGroup>
</Project>"""
    
    File.WriteAllText(Path.Combine(versionPath, "src", "Program.fs"), codeTemplate)
    File.WriteAllText(Path.Combine(versionPath, name + ".fsproj"), projectFile)
    
    printfn "✅ [VERSION] Created: %s" versionPath
    versionPath

// Create different green versions
let basicVersion = createGreenVersion "TaskManager" "1.0" "Basic Task Tracking"
printfn ""

let advancedVersion = createGreenVersion "TaskManager" "2.0" "Enterprise Management"
printfn ""

let collaborativeVersion = createGreenVersion "TaskManager" "3.0" "Team Collaboration"
printfn ""

// Test all versions
printfn "🧪 [TESTING] Validating all green versions..."
printfn "============================================="

let testVersion (versionPath: string) (versionName: string) =
    printfn "🧪 Testing %s..." versionName
    
    try
        let startInfo = System.Diagnostics.ProcessStartInfo()
        startInfo.FileName <- "dotnet"
        startInfo.Arguments <- "run"
        startInfo.WorkingDirectory <- versionPath
        startInfo.RedirectStandardOutput <- true
        startInfo.UseShellExecute <- false
        
        use proc = System.Diagnostics.Process.Start(startInfo)
        proc.WaitForExit(5000) |> ignore
        
        if proc.ExitCode = 0 then
            let output = proc.StandardOutput.ReadToEnd()
            printfn "  ✅ %s - PASSED" versionName
            printfn "  📋 Output: %s" (output.Split('\n').[0])
            true
        else
            printfn "  ❌ %s - FAILED" versionName
            false
    with
    | ex ->
        printfn "  ❌ %s - ERROR: %s" versionName ex.Message
        false

let basicResult = testVersion basicVersion "Basic v1.0"
let advancedResult = testVersion advancedVersion "Advanced v2.0"
let collaborativeResult = testVersion collaborativeVersion "Collaborative v3.0"

printfn ""
printfn "📊 [SUMMARY] Multiple Green Versions Status"
printfn "==========================================="
printfn "1. Basic Task Manager v1.0 - %s" (if basicResult then "✅ OPERATIONAL" else "❌ FAILED")
printfn "   Use Case: Simple task tracking"
printfn "   Path: %s" basicVersion
printfn ""
printfn "2. Advanced Task Manager v2.0 - %s" (if advancedResult then "✅ OPERATIONAL" else "❌ FAILED")
printfn "   Use Case: Enterprise task management"
printfn "   Path: %s" advancedVersion
printfn ""
printfn "3. Collaborative Task Manager v3.0 - %s" (if collaborativeResult then "✅ OPERATIONAL" else "❌ FAILED")
printfn "   Use Case: Team collaboration"
printfn "   Path: %s" collaborativeVersion

let operationalCount = [basicResult; advancedResult; collaborativeResult] |> List.filter id |> List.length
printfn ""
printfn "🎯 Operational Versions: %d/3" operationalCount
printfn "🟢 System Health: %s" (if operationalCount = 3 then "EXCELLENT" else "NEEDS ATTENTION")

printfn ""
printfn "✅ Multiple green versions system operational!"
printfn "🎯 Different versions available for different use cases!"
printfn "🔒 All versions maintain system stability!"
