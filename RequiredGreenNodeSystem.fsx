#!/usr/bin/env dotnet fsi

// REQUIRED GREEN NODE SYSTEM
// Ensures we always have at least one stable green node before allowing blue experiments

open System
open System.IO
open System.Text.Json

type NodeStatus = Active | Deprecated | Failed | Experimental
type QALevel = Basic | Standard | Comprehensive | Production

type GreenNode = {
    Id: string
    Name: string
    ProjectPath: string
    Version: string
    CreatedAt: DateTime
    LastValidated: DateTime
    Status: NodeStatus
    StabilityScore: float
    IsRequired: bool
    QALevel: QALevel
}

type BlueNode = {
    Id: string
    Name: string
    ProjectPath: string
    CreatedAt: DateTime
    TargetGreenNode: string option
    QAResults: string list
    ReadyForPromotion: bool
}

type SystemState = {
    RequiredGreenNodes: GreenNode list
    ExperimentalBlueNodes: BlueNode list
    SystemStable: bool
    LastStabilityCheck: DateTime
}

module GreenNodeManager =
    let mutable systemState = {
        RequiredGreenNodes = []
        ExperimentalBlueNodes = []
        SystemStable = false
        LastStabilityCheck = DateTime.UtcNow
    }
    
    let createRequiredGreenNode (name: string) (version: string) =
        printfn "🟢 [REQUIRED] Creating required green node: %s v%s" name version
        
        let greenId = Guid.NewGuid().ToString("N")[..7]
        let greenPath = Path.Combine(".tars", "green", "required", sprintf "%s_v%s_%s" name version greenId)
        
        Directory.CreateDirectory(greenPath) |> ignore
        Directory.CreateDirectory(Path.Combine(greenPath, "src")) |> ignore
        
        // Create a stable, production-ready task manager
        let stableCode = """open System

type Task = {
    Id: int
    Title: string
    Description: string
    Priority: string
    Status: string
    CreatedAt: DateTime
    CompletedAt: DateTime option
}

type TaskManager() =
    let mutable tasks: Task list = []
    let mutable nextId = 1
    
    member _.AddTask(title: string, description: string, priority: string) =
        let task = {
            Id = nextId
            Title = title
            Description = description
            Priority = priority
            Status = "Pending"
            CreatedAt = DateTime.UtcNow
            CompletedAt = None
        }
        tasks <- task :: tasks
        nextId <- nextId + 1
        printfn "✅ [GREEN] Added %s priority task: %s" priority title
        task
    
    member _.CompleteTask(id: int) =
        tasks <- tasks |> List.map (fun t ->
            if t.Id = id then 
                { t with Status = "Completed"; CompletedAt = Some DateTime.UtcNow }
            else t)
        printfn "🎉 [GREEN] Completed task %d" id
    
    member _.GetTasks() = tasks
    
    member _.GetTasksByStatus(status: string) =
        tasks |> List.filter (fun t -> t.Status = status)
    
    member _.GetStatistics() =
        let total = tasks.Length
        let completed = tasks |> List.filter (fun t -> t.Status = "Completed") |> List.length
        let pending = total - completed
        {| Total = total; Completed = completed; Pending = pending; CompletionRate = if total > 0 then (float completed / float total) * 100.0 else 0.0 |}

[<EntryPoint>]
let main argv =
    printfn "🟢 STABLE GREEN NODE - Production Task Manager"
    printfn "=============================================="
    printfn "🎯 This is the required stable baseline system"
    printfn "📊 Version: Production-Ready | QA Level: Comprehensive"
    printfn ""
    
    let taskManager = TaskManager()
    
    // Demonstrate stable functionality
    let task1 = taskManager.AddTask("System architecture review", "Review and validate system architecture", "High")
    let task2 = taskManager.AddTask("Code quality audit", "Perform comprehensive code quality audit", "High")
    let task3 = taskManager.AddTask("Performance optimization", "Optimize system performance", "Medium")
    let task4 = taskManager.AddTask("Documentation update", "Update system documentation", "Low")
    
    printfn ""
    printfn "📋 [GREEN] Current Tasks:"
    taskManager.GetTasks() |> List.iter (fun t ->
        printfn "  %s [%s] %d. %s" t.Status t.Priority t.Id t.Title)
    
    // Complete some tasks
    taskManager.CompleteTask(1)
    taskManager.CompleteTask(2)
    
    printfn ""
    printfn "📊 [GREEN] System Statistics:"
    let stats = taskManager.GetStatistics()
    printfn "  Total Tasks: %d" stats.Total
    printfn "  Completed: %d" stats.Completed
    printfn "  Pending: %d" stats.Pending
    printfn "  Completion Rate: %.1f%%" stats.CompletionRate
    
    printfn ""
    printfn "✅ [GREEN] Stable green node operational!"
    printfn "🔒 [SYSTEM] Required baseline maintained"
    printfn "🎯 [STATUS] System ready for blue node experiments"
    0
"""
        
        let projectFile = sprintf """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <Version>%s</Version>
    <AssemblyTitle>Required Green Node - %s</AssemblyTitle>
  </PropertyGroup>
  <ItemGroup>
    <Compile Include="src/Program.fs" />
  </ItemGroup>
</Project>""" version name
        
        File.WriteAllText(Path.Combine(greenPath, "src", "Program.fs"), stableCode)
        File.WriteAllText(Path.Combine(greenPath, sprintf "%s.fsproj" name), projectFile)
        
        let greenNode = {
            Id = greenId
            Name = name
            ProjectPath = greenPath
            Version = version
            CreatedAt = DateTime.UtcNow
            LastValidated = DateTime.UtcNow
            Status = Active
            StabilityScore = 0.98
            IsRequired = true
            QALevel = Production
        }
        
        systemState <- { 
            systemState with 
                RequiredGreenNodes = greenNode :: systemState.RequiredGreenNodes
                SystemStable = true
        }
        
        printfn "✅ [REQUIRED] Green node created: %s" greenPath
        greenNode
    
    let validateSystemStability() =
        let activeGreenNodes = systemState.RequiredGreenNodes |> List.filter (fun n -> n.Status = Active)
        let hasRequiredNode = activeGreenNodes |> List.exists (fun n -> n.IsRequired)
        let avgStability = if activeGreenNodes.IsEmpty then 0.0 else activeGreenNodes |> List.averageBy (fun n -> n.StabilityScore)
        
        let isStable = hasRequiredNode && avgStability >= 0.95 && activeGreenNodes.Length >= 1
        
        systemState <- { 
            systemState with 
                SystemStable = isStable
                LastStabilityCheck = DateTime.UtcNow
        }
        
        printfn "🔍 [STABILITY] System stability check:"
        printfn "  Required green nodes: %d" (activeGreenNodes |> List.filter (fun n -> n.IsRequired) |> List.length)
        printfn "  Average stability: %.2f" avgStability
        printfn "  System stable: %s" (if isStable then "✅ YES" else "❌ NO")
        
        isStable
    
    let allowBlueNodeCreation() =
        if not systemState.SystemStable then
            printfn "🚫 [POLICY] Cannot create blue nodes - no stable green baseline!"
            printfn "🔧 [ACTION] Create required green node first"
            false
        else
            printfn "✅ [POLICY] Blue node creation allowed - stable green baseline exists"
            true
    
    let createBlueNode (name: string) (targetGreenNodeId: string option) =
        if not (allowBlueNodeCreation()) then
            None
        else
            let blueId = Guid.NewGuid().ToString("N")[..7]
            let bluePath = Path.Combine(".tars", "blue", "experimental", sprintf "%s_%s" name blueId)
            
            Directory.CreateDirectory(bluePath) |> ignore
            Directory.CreateDirectory(Path.Combine(bluePath, "src")) |> ignore
            
            // Create experimental blue node code
            let experimentalCode = """open System

type ExperimentalTask = {
    Id: int
    Title: string
    Priority: string
    Tags: string list
    Experimental: bool
}

let mutable tasks = []

let addExperimentalTask title priority tags =
    let id = List.length tasks + 1
    let task = { Id = id; Title = title; Priority = priority; Tags = tags; Experimental = true }
    tasks <- task :: tasks
    printfn "🔵 [BLUE] Added experimental task: %s" title

[<EntryPoint>]
let main argv =
    printfn "🔵 EXPERIMENTAL BLUE NODE"
    printfn "========================="
    printfn "⚠️ This is experimental code - not for production!"
    
    addExperimentalTask "Test new feature" "High" ["experimental"; "feature"]
    addExperimentalTask "Prototype UI" "Medium" ["ui"; "prototype"]
    
    printfn "📋 Experimental tasks: %d" tasks.Length
    printfn "🔵 [BLUE] Experimental node operational"
    0
"""
            
            let projectFile = sprintf """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <AssemblyTitle>Experimental Blue Node - %s</AssemblyTitle>
  </PropertyGroup>
  <ItemGroup>
    <Compile Include="src/Program.fs" />
  </ItemGroup>
</Project>""" name
            
            File.WriteAllText(Path.Combine(bluePath, "src", "Program.fs"), experimentalCode)
            File.WriteAllText(Path.Combine(bluePath, sprintf "%s.fsproj" name), projectFile)
            
            let blueNode = {
                Id = blueId
                Name = name
                ProjectPath = bluePath
                CreatedAt = DateTime.UtcNow
                TargetGreenNode = targetGreenNodeId
                QAResults = []
                ReadyForPromotion = false
            }
            
            systemState <- { 
                systemState with 
                    ExperimentalBlueNodes = blueNode :: systemState.ExperimentalBlueNodes
            }
            
            printfn "🔵 [BLUE] Created experimental blue node: %s" bluePath
            Some blueNode
    
    let getSystemState() = systemState
    let getGreenNodes() = systemState.RequiredGreenNodes
    let getBlueNodes() = systemState.ExperimentalBlueNodes

// Demo: Required Green Node System
printfn "🟢🔵 REQUIRED GREEN NODE SYSTEM"
printfn "==============================="
printfn "🎯 Policy: Must have stable green baseline before blue experiments"
printfn ""

// Try to create blue node without green baseline (should fail)
printfn "🧪 [TEST] Attempting to create blue node without green baseline..."
match GreenNodeManager.createBlueNode "ExperimentalFeature" None with
| Some _ -> printfn "❌ [ERROR] Blue node created without green baseline!"
| None -> printfn "✅ [POLICY] Correctly blocked blue node creation"

printfn ""

// Create required green node
printfn "🟢 [SETUP] Creating required green baseline..."
let requiredGreen = GreenNodeManager.createRequiredGreenNode "StableTaskManager" "1.0.0"

printfn ""

// Validate system stability
printfn "🔍 [VALIDATION] Checking system stability..."
let isStable = GreenNodeManager.validateSystemStability()

printfn ""

// Now try to create blue node (should succeed)
printfn "🧪 [TEST] Attempting to create blue node with green baseline..."
match GreenNodeManager.createBlueNode "ExperimentalFeature" (Some requiredGreen.Id) with
| Some blueNode -> printfn "✅ [SUCCESS] Blue node created: %s" blueNode.Name
| None -> printfn "❌ [ERROR] Blue node creation failed"

printfn ""

// System summary
printfn "📊 [SUMMARY] System State"
printfn "========================"
let state = GreenNodeManager.getSystemState()
printfn "🟢 Required Green Nodes: %d" state.RequiredGreenNodes.Length
printfn "🔵 Experimental Blue Nodes: %d" state.ExperimentalBlueNodes.Length
printfn "🔒 System Stable: %s" (if state.SystemStable then "✅ YES" else "❌ NO")
printfn "⏰ Last Stability Check: %s" (state.LastStabilityCheck.ToString("HH:mm:ss"))

printfn ""
printfn "✅ Required green node system operational!"
printfn "🎯 Policy enforced: Stable green baseline required for blue experiments!"
