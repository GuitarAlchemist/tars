open System

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
