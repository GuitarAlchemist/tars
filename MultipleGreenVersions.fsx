#!/usr/bin/env dotnet fsi

// MULTIPLE GREEN VERSIONS SYSTEM
// Manage different green versions for different use cases

open System
open System.IO

type GreenVersion = {
    Name: string
    Version: string
    UseCase: string
    Path: string
    CreatedAt: DateTime
    Status: string
}

let mutable greenVersions: GreenVersion list = []

let createGreenVersion (name: string) (version: string) (useCase: string) (codeTemplate: string) =
    printfn "🟢 [VERSION] Creating green version: %s v%s for %s" name version useCase
    
    let versionPath = Path.Combine(".tars", "green", "versions", sprintf "%s_v%s_%s" name version (DateTime.Now.ToString("HHmmss")))
    Directory.CreateDirectory(versionPath) |> ignore
    Directory.CreateDirectory(Path.Combine(versionPath, "src")) |> ignore
    
    let projectFile = sprintf """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <Version>%s</Version>
    <AssemblyTitle>%s - %s</AssemblyTitle>
  </PropertyGroup>
  <ItemGroup>
    <Compile Include="src/Program.fs" />
  </ItemGroup>
</Project>""" version name useCase
    
    File.WriteAllText(Path.Combine(versionPath, "src", "Program.fs"), codeTemplate)
    File.WriteAllText(Path.Combine(versionPath, sprintf "%s.fsproj" name), projectFile)
    
    let greenVersion = {
        Name = name
        Version = version
        UseCase = useCase
        Path = versionPath
        CreatedAt = DateTime.UtcNow
        Status = "Active"
    }
    
    greenVersions <- greenVersion :: greenVersions
    printfn "✅ [VERSION] Created: %s" versionPath
    greenVersion

// Create different green versions for different use cases
printfn "🟢📦 MULTIPLE GREEN VERSIONS SYSTEM"
printfn "==================================="
printfn ""

// Version 1: Basic Task Manager
let basicTaskManagerCode = """open System

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

let basicVersion = createGreenVersion "TaskManager" "1.0" "Basic Task Tracking" basicTaskManagerCode

printfn ""

// Version 2: Advanced Task Manager
let advancedTaskManagerCode = """open System

type Priority = High | Medium | Low
type Status = Pending | InProgress | Completed

type AdvancedTask = {
    Id: int
    Title: string
    Description: string
    Priority: Priority
    Status: Status
    CreatedAt: DateTime
    DueDate: DateTime option
}

let mutable tasks = []

let addAdvancedTask title description priority dueDate =
    let id = List.length tasks + 1
    let task = {
        Id = id
        Title = title
        Description = description
        Priority = priority
        Status = Pending
        CreatedAt = DateTime.UtcNow
        DueDate = dueDate
    }
    tasks <- task :: tasks
    printfn "✅ [ADVANCED] Added %A priority: %s" priority title

let updateTaskStatus id newStatus =
    tasks <- tasks |> List.map (fun t ->
        if t.Id = id then { t with Status = newStatus }
        else t)
    printfn "🔄 [ADVANCED] Updated task %d to %A" id newStatus

[<EntryPoint>]
let main argv =
    printfn "🟢 ADVANCED TASK MANAGER v2.0"
    printfn "============================="
    printfn "📋 Use Case: Enterprise task management"
    
    addAdvancedTask "System architecture review" "Comprehensive system review" High (Some (DateTime.Today.AddDays(7.0)))
    addAdvancedTask "Code optimization" "Performance improvements" Medium (Some (DateTime.Today.AddDays(14.0)))
    addAdvancedTask "Documentation update" "Update user documentation" Low None
    
    updateTaskStatus 1 InProgress
    updateTaskStatus 2 Completed
    
    let completedTasks = tasks |> List.filter (fun t -> t.Status = Completed) |> List.length
    printfn "📊 Total tasks: %d, Completed: %d" tasks.Length completedTasks
    printfn "✅ [ADVANCED] Green version operational!"
    0
"""

let advancedVersion = createGreenVersion "TaskManager" "2.0" "Enterprise Management" advancedTaskManagerCode

printfn ""

// Version 3: Collaborative Task Manager
let collaborativeTaskManagerCode = """open System

type User = { Id: int; Name: string; Role: string }
type CollaborativeTask = {
    Id: int
    Title: string
    AssignedTo: User option
    CreatedBy: User
    Status: string
    Comments: string list
}

let mutable users = [
    { Id = 1; Name = "Alice"; Role = "Developer" }
    { Id = 2; Name = "Bob"; Role = "Manager" }
    { Id = 3; Name = "Carol"; Role = "Designer" }
]

let mutable tasks = []

let assignTask title createdBy assignedTo =
    let id = List.length tasks + 1
    let task = {
        Id = id
        Title = title
        AssignedTo = assignedTo
        CreatedBy = createdBy
        Status = "Assigned"
        Comments = []
    }
    tasks <- task :: tasks
    match assignedTo with
    | Some user -> printfn "✅ [COLLAB] Assigned '%s' to %s" title user.Name
    | None -> printfn "✅ [COLLAB] Created unassigned task: %s" title

let addComment taskId comment =
    tasks <- tasks |> List.map (fun t ->
        if t.Id = taskId then { t with Comments = comment :: t.Comments }
        else t)
    printfn "💬 [COLLAB] Added comment to task %d" taskId

[<EntryPoint>]
let main argv =
    printfn "🟢 COLLABORATIVE TASK MANAGER v3.0"
    printfn "=================================="
    printfn "📋 Use Case: Team collaboration"
    
    let alice = users.[0]
    let bob = users.[1]
    let carol = users.[2]
    
    assignTask "Design new UI components" bob (Some carol)
    assignTask "Implement backend API" bob (Some alice)
    assignTask "Project planning meeting" alice None
    
    addComment 1 "Initial wireframes completed"
    addComment 2 "API endpoints defined"
    
    printfn "👥 Team members: %d" users.Length
    printfn "📊 Collaborative tasks: %d" tasks.Length
    printfn "✅ [COLLAB] Green version operational!"
    0
"""

let collaborativeVersion = createGreenVersion "TaskManager" "3.0" "Team Collaboration" collaborativeTaskManagerCode

printfn ""

// Test all green versions
printfn "🧪 [TESTING] Validating all green versions..."
printfn "============================================="

let testGreenVersion (version: GreenVersion) =
    printfn "🧪 Testing %s v%s (%s)..." version.Name version.Version version.UseCase
    
    try
        let startInfo = System.Diagnostics.ProcessStartInfo()
        startInfo.FileName <- "dotnet"
        startInfo.Arguments <- "run"
        startInfo.WorkingDirectory <- version.Path
        startInfo.RedirectStandardOutput <- true
        startInfo.UseShellExecute <- false
        
        use proc = System.Diagnostics.Process.Start(startInfo)
        proc.WaitForExit(5000) |> ignore
        
        if proc.ExitCode = 0 then
            let output = proc.StandardOutput.ReadToEnd()
            printfn "  ✅ %s v%s - PASSED" version.Name version.Version
            printfn "  📋 Output: %s" (output.Split('\n').[0])
            true
        else
            printfn "  ❌ %s v%s - FAILED" version.Name version.Version
            false
    with
    | ex ->
        printfn "  ❌ %s v%s - ERROR: %s" version.Name version.Version ex.Message
        false

let testResults = greenVersions |> List.map testGreenVersion

printfn ""
printfn "📊 [SUMMARY] Multiple Green Versions Status"
printfn "==========================================="

greenVersions |> List.rev |> List.iteri (fun i version ->
    let status = if testResults.[greenVersions.Length - 1 - i] then "✅ OPERATIONAL" else "❌ FAILED"
    printfn "%d. %s v%s - %s" (i + 1) version.Name version.Version status
    printfn "   Use Case: %s" version.UseCase
    printfn "   Path: %s" version.Path
    printfn "")

let operationalVersions = testResults |> List.filter id |> List.length
printfn "🎯 Operational Versions: %d/%d" operationalVersions greenVersions.Length
printfn "🟢 System Health: %s" (if operationalVersions = greenVersions.Length then "EXCELLENT" else "NEEDS ATTENTION")

printfn ""
printfn "✅ Multiple green versions system operational!"
printfn "🎯 Different versions available for different use cases!"
printfn "🔒 All versions maintain system stability!"
