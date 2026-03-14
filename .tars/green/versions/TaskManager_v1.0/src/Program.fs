open System

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
