let calculateValue x y =
    let intermediate = x * 2
    let result = intermediate + y
    result

/// <summary>TODO: Document main</summary>
let main () =
    let v1 = 10
    let v2 = 20
    let final = calculateValue v1 v2
    printfn "Result: %d" final
