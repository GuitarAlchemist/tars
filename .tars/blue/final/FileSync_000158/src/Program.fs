open System

type SyncFile = { Name: string; Size: int; Hash: string }

let mutable files = []

let addFile name content =
    let file = { Name = name; Size = content.Length; Hash = content.GetHashCode().ToString() }
    files <- file :: files
    printfn "🔵 [SYNC] Added: %s (%d bytes)" name content.Length

[<EntryPoint>]
let main argv =
    printfn "🔵 FILE SYNC - Blue Node"
    printfn "======================="
    
    addFile "doc1.txt" "Important document"
    addFile "config.json" "Configuration"
    
    printfn "📁 Files: %d" files.Length
    printfn "🔵 [BLUE] Sync operational!"
    0
