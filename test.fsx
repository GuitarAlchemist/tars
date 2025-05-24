// This is a test F# script file

printfn "Hello from F# script!"

// Define a function
let add x y = x + y

// Use the function
let result = add 5 7
printfn "5 + 7 = %d" result

// Define a recursive function
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)

// Use the recursive function
let fact5 = factorial 5
printfn "Factorial of 5 = %d" fact5

// Define a higher-order function
let applyTwice f x = f (f x)

// Use the higher-order function
let double x = x * 2
let quadruple = applyTwice double
printfn "Quadruple of 3 = %d" (quadruple 3)

// Define a list
let numbers = [1; 2; 3; 4; 5]

// Use list functions
let sum = List.sum numbers
printfn "Sum of numbers = %d" sum

let doubled = List.map double numbers
printfn "Doubled numbers = %A" doubled

let evens = List.filter (fun x -> x % 2 = 0) numbers
printfn "Even numbers = %A" evens
