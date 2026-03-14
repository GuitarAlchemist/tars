let rec fibonacci n =
    match n with
    | x when x < 0 -> failwith "Negative input not allowed"
    | 0 -> 0L
    | 1 -> 1L
    | _ -> 
        let mutable a, b = 0L, 1L
        for _ in 2 .. n do
            let temp = a + b
            a <- b
            b <- temp
        b

// Example usage:
let result = fibonacci 10
printfn "%d" result