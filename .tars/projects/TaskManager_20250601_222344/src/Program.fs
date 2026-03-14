open System

type Task = { Id: int; Title: string; Done: bool }

let mutable tasks = []

let addTask title =
    let id = List.length tasks + 1
    let task = { Id = id; Title = title; Done = false }
    tasks <- task :: tasks
    printfn "✅ Added: %s" title

let showTasks () =
    printfn "📋 Tasks:"
    tasks |> List.iter (fun t ->
        let status = if t.Done then "✅" else "⏳"
        printfn "  %s %d. %s" status t.Id t.Title)

[<EntryPoint>]
let main argv =
    printfn "🚀 TASK MANAGER"
    printfn "==============="
    
    addTask "Write code"
    addTask "Test code"
    addTask "Deploy code"
    
    showTasks()
    
    printfn ""
    printfn "✅ TARS generated working code!"
    0
