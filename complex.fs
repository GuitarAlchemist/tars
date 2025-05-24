module Complex

// Recursive function
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)

// Pattern matching
let describe x =
    match x with
    | 0 -> "Zero"
    | 1 -> "One"
    | 2 -> "Two"
    | _ -> "Many"

// Higher-order function
let applyTwice f x = f (f x)

// Active pattern
let (|Even|Odd|) n = if n % 2 = 0 then Even else Odd

// Computation expression
let maybe = async {
    let! x = async { return 42 }
    return x * 2
}

[<EntryPoint>]
let main args =
    printfn "Factorial of 5: %d" (factorial 5)
    printfn "Description of 2: %s" (describe 2)
    printfn "Apply twice: %d" (applyTwice (fun x -> x * 2) 3)
    
    match 10 with
    | Even -> printfn "Even number"
    | Odd -> printfn "Odd number"
    
    let result = Async.RunSynchronously maybe
    printfn "Result: %d" result
    
    0
