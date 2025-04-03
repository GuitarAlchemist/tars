namespace TarsEngine.FSharpToRust

/// Examples for the F# to Rust transpiler
module Examples =
    
    /// Sample F# code for transpilation
    let sampleCode = """
module BasicFunctions

// Simple function to calculate factorial
let rec factorial (n: int) : int =
    if n <= 1 then
        1
    else
        n * factorial (n - 1)

// Function with pattern matching
let isPositive (x: int) : bool =
    match x with
    | n when n > 0 -> true
    | 0 -> false
    | _ -> false

// Function with tuple
let swap (a, b) = (b, a)

// Function with list processing
let sumList (numbers: int list) : int =
    List.fold (fun acc x -> acc + x) 0 numbers

// Function with option type
let safeDivide (a: int) (b: int) : int option =
    if b = 0 then
        None
    else
        Some (a / b)
"""

    /// Run the transpiler on the sample code
    let runExample () =
        let rustCode = FSharpToRustTranspiler.simpleTranspile sampleCode
        printfn "Transpiled Rust code:\n%s" rustCode
        rustCode
