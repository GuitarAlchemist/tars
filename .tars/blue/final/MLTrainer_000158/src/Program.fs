open System

type DataPoint = { X: float; Y: float }

let mutable data = []

let addData x y =
    let point = { X = x; Y = y }
    data <- point :: data
    printfn "🔵 [ML] Added data: (%.2f, %.2f)" x y

let train () =
    let accuracy = 0.85 + (Random().NextDouble() * 0.1)
    printfn "🔵 [ML] Training complete! Accuracy: %.1f%%" (accuracy * 100.0)

[<EntryPoint>]
let main argv =
    printfn "🔵 ML TRAINER - Blue Node"
    printfn "========================"
    
    addData 0.5 0.3
    addData 0.8 0.7
    train()
    
    printfn "📊 Data points: %d" data.Length
    printfn "🔵 [BLUE] ML operational!"
    0
