open System

type ExperimentalTask = { Id: int; Title: string; Experimental: bool }

let mutable tasks = []

let addExperimentalTask title =
    let id = List.length tasks + 1
    let task = { Id = id; Title = title; Experimental = true }
    tasks <- task :: tasks
    printfn "🔵 [GREEN] Added experimental task: %s" title

[<EntryPoint>]
let main argv =
    printfn "🟢 PROMOTED GREEN NODE"
    printfn "========================="
    printfn "✅ This is stable, promoted code!"
    
    addExperimentalTask "Test new feature"
    addExperimentalTask "Prototype UI"
    
    printfn "📋 Experimental tasks: %d" tasks.Length
    printfn "🔵 [GREEN] Experimental node operational"
    0
