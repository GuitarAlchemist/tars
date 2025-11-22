// Sample F# file for transpilation to Rust

// Simple function to calculate factorial
let rec factorial (n: int) : int =
    if n <= 1 then
        1
    else
        n * factorial (n - 1)

// Function to check if a number is even
let isEven (x: int) : bool =
    x % 2 = 0

// Function to double a number
let double (x: int) : int =
    x * 2

// Function to calculate the sum of a list
let sum (numbers: int list) : int =
    List.fold (fun acc x -> acc + x) 0 numbers

// Function to find the maximum value in a list
let findMax (numbers: int list) : int option =
    match numbers with
    | [] -> None
    | x::xs -> Some (List.fold max x xs)

// Function to filter even numbers
let filterEven (numbers: int list) : int list =
    List.filter isEven numbers

// Main function
let main () =
    let result = factorial 5
    printfn "Factorial of 5 is %d" result
    
    let numbers = [1; 2; 3; 4; 5]
    let sumResult = sum numbers
    printfn "Sum of numbers is %d" sumResult
    
    let maxResult = findMax numbers
    match maxResult with
    | Some max -> printfn "Maximum value is %d" max
    | None -> printfn "List is empty"
    
    let evenNumbers = filterEven numbers
    printfn "Even numbers: %A" evenNumbers
