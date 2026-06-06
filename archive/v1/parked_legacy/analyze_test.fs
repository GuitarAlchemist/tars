module AnalyzeTest

// Define a recursive function
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)

// Define a higher-order function
let applyTwice f x = f (f x)

// Use pattern matching
let isPositive x =
    match x with
    | x when x > 0 -> true
    | _ -> false

// Define an active pattern
let (|Even|Odd|) n = if n % 2 = 0 then Even else Odd

// Use a computation expression
let asyncExample = async {
    let! result = async { return 42 }
    return result * 2
}

// Define a complex function with high cyclomatic complexity
let complexFunction x =
    if x < 0 then
        if x < -10 then
            if x < -20 then
                -3
            else
                -2
        else
            -1
    else
        if x > 10 then
            if x > 20 then
                3
            else
                2
        else
            1

// Use list functions
let numbers = [1; 2; 3; 4; 5]
let doubled = List.map (fun x -> x * 2) numbers
let evens = List.filter (fun x -> x % 2 = 0) numbers
let sum = List.fold (+) 0 numbers
