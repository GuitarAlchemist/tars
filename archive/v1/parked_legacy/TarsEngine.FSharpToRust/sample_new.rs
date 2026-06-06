// Sample F# file for transpilation to Rust

// Simple function to calculate factorial
let rec factorial (n: i32) : i32 =
    if n <= 1 { 1 } else { n * factorial (n - 1)

// Function to check } if a number is even
fn isEven(x: /* Unmapped type: i32 */ ()) -> bool {
    x % 2 = 0

// Function to double a number
}fn double(x: /* Unmapped type: i32 */ ()) -> i32 {
    x * 2

// Function to calculate the sum of a list
}fn sum(numbers: /* Unmapped type: i32 list */ ()) -> i32 {
    List.fold (fun acc x -> acc + x) 0 numbers

// Function to find the maximum value in a list
}fn findMax(numbers: /* Unmapped type: i32 list */ ()) -> /* Unmapped type: i32 option */ () {
    match numbers {
[] -> None
    x:: /* Unmapped type: xs */ () => { Some (List.fold max x xs)

// Function to filter even numbers
}fn filterEven(numbers: /* Unmapped type: i32 list */ ()) -> /* Unmapped type: i32 list */ () {
    List.filter isEven numbers

// Main function
}fn main() -> /* unknown return type */ { },
}let result = factorial 5
    printfn "Factorial of 5 is %d" result
}fn numbers() -> /* unknown return type */ {
    [1; 2; 3; 4; 5]
}fn sumResult() -> /* unknown return type */ {
    sum numbers
    printfn "Sum of numbers is %d" sumResult
}fn maxResult() -> /* unknown return type */ {
    findMax numbers
    match maxResult {
Some max -> printfn "Maximum value is %d" max
    None => { printfn "List is empty"
}fn evenNumbers() -> /* unknown return type */ {
    filterEven numbers
    printfn "Even numbers: %A" evenNumbers
} },
}