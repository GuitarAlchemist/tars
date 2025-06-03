open System

type Message = { From: string; Content: string; Time: DateTime }

let mutable messages = []

let sendMessage from content =
    let msg = { From = from; Content = content; Time = DateTime.UtcNow }
    messages <- msg :: messages
    printfn "🔵 [CHAT] %s: %s" from content

[<EntryPoint>]
let main argv =
    printfn "🔵 CHAT APPLICATION - Blue Node"
    printfn "=============================="
    
    sendMessage "Alice" "Hello world!"
    sendMessage "Bob" "Hi Alice!"
    
    printfn "💬 Messages: %d" messages.Length
    printfn "🔵 [BLUE] Chat operational!"
    0
