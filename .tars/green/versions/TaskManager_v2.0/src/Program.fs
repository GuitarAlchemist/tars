open System

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
