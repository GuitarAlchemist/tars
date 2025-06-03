open System

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
