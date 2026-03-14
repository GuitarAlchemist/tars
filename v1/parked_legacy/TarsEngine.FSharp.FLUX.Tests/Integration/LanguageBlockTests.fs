namespace TarsEngine.FSharp.FLUX.Tests.Integration

open System
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open Unquote
open TarsEngine.FSharp.FLUX.Ast.FluxAst
open TarsEngine.FSharp.FLUX.FluxEngine
open TarsEngine.FSharp.FLUX.Tests.TestHelpers

/// Integration tests for Language Block execution
module LanguageBlockTests =
    
    [<Fact>]
    let ``F# language block executes successfully`` () =
        task {
            // Arrange
            let script = """
META {
    title: "F# Test"
    version: "1.0.0"
}

FSHARP {
    let greeting = "Hello from F#!"
    let numbers = [1; 2; 3; 4; 5]
    let sum = numbers |> List.sum
    printfn "%s" greeting
    printfn "Sum of numbers: %d" sum
    
    // Test function definition
    let multiply x y = x * y
    let result = multiply 6 7
    printfn "6 * 7 = %d" result
}
"""
            let engine = FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 2 // META + FSHARP
        }
    
    [<Fact>]
    let ``Python language block executes successfully`` () =
        task {
            // Arrange
            let script = """
META {
    title: "Python Test"
    version: "1.0.0"
}

PYTHON {
    greeting = "Hello from Python!"
    numbers = [1, 2, 3, 4, 5]
    total = sum(numbers)
    print(greeting)
    print(f"Sum of numbers: {total}")
    
    # Test function definition
    def multiply(x, y):
        return x * y
    
    result = multiply(6, 7)
    print(f"6 * 7 = {result}")
}
"""
            let engine = FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 2 // META + PYTHON
        }
    
    [<Fact>]
    let ``JavaScript language block executes successfully`` () =
        task {
            // Arrange
            let script = """
META {
    title: "JavaScript Test"
    version: "1.0.0"
}

JAVASCRIPT {
    const greeting = "Hello from JavaScript!";
    const numbers = [1, 2, 3, 4, 5];
    const sum = numbers.reduce((a, b) => a + b, 0);
    console.log(greeting);
    console.log(`Sum of numbers: ${sum}`);
    
    // Test function definition
    function multiply(x, y) {
        return x * y;
    }
    
    const result = multiply(6, 7);
    console.log(`6 * 7 = ${result}`);
}
"""
            let engine = FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 2 // META + JAVASCRIPT
        }
    
    [<Fact>]
    let ``C# language block executes successfully`` () =
        task {
            // Arrange
            let script = """
META {
    title: "C# Test"
    version: "1.0.0"
}

CSHARP {
    using System;
    using System.Linq;
    
    var greeting = "Hello from C#!";
    var numbers = new[] { 1, 2, 3, 4, 5 };
    var sum = numbers.Sum();
    Console.WriteLine(greeting);
    Console.WriteLine($"Sum of numbers: {sum}");
    
    // Test function definition
    int Multiply(int x, int y) => x * y;
    
    var result = Multiply(6, 7);
    Console.WriteLine($"6 * 7 = {result}");
}
"""
            let engine = FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 2 // META + CSHARP
        }
    
    [<Fact>]
    let ``Multiple language blocks execute in sequence`` () =
        task {
            // Arrange
            let script = """
META {
    title: "Multi-Language Test"
    version: "1.0.0"
}

FSHARP {
    let fsharpValue = 42
    printfn "F# value: %d" fsharpValue
}

PYTHON {
    python_value = 24
    print(f"Python value: {python_value}")
}

JAVASCRIPT {
    const jsValue = 84;
    console.log(`JavaScript value: ${jsValue}`);
}
"""
            let engine = FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 4 // META + 3 language blocks
        }
    
    [<Fact>]
    let ``Language blocks can handle complex computations`` () =
        task {
            // Arrange
            let script = """
META {
    title: "Complex Computation Test"
    version: "1.0.0"
}

FSHARP {
    // Fibonacci sequence
    let rec fibonacci n =
        match n with
        | 0 | 1 -> n
        | _ -> fibonacci (n - 1) + fibonacci (n - 2)
    
    let fibResult = fibonacci 10
    printfn "Fibonacci(10) = %d" fibResult
    
    // List processing
    let data = [1..100]
    let evenNumbers = data |> List.filter (fun x -> x % 2 = 0)
    let evenSum = evenNumbers |> List.sum
    printfn "Sum of even numbers 1-100: %d" evenSum
}

PYTHON {
    # Prime number calculation
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [n for n in range(2, 50) if is_prime(n)]
    print(f"Primes under 50: {primes}")
    print(f"Count of primes: {len(primes)}")
}
"""
            let engine = FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 3 // META + 2 language blocks
        }
    
    [<Fact>]
    let ``Language blocks handle errors gracefully`` () =
        task {
            // Arrange
            let script = """
META {
    title: "Error Handling Test"
    version: "1.0.0"
}

FSHARP {
    try
        let result = 10 / 0  // This will cause an error
        printfn "Result: %d" result
    with
    | ex -> printfn "Caught exception: %s" ex.Message
}

PYTHON {
    try:
        result = 10 / 0  # This will cause an error
        print(f"Result: {result}")
    except Exception as e:
        print(f"Caught exception: {e}")
}
"""
            let engine = FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(script)
            
            // Assert
            // The execution should succeed even with handled errors
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 3 // META + 2 language blocks
        }
    
    [<Fact>]
    let ``Language blocks can use external libraries`` () =
        task {
            // Arrange
            let script = """
META {
    title: "External Library Test"
    version: "1.0.0"
}

FSHARP {
    open System
    open System.IO
    
    // Use System libraries
    let currentTime = DateTime.Now
    printfn "Current time: %A" currentTime
    
    let tempPath = Path.GetTempPath()
    printfn "Temp path: %s" tempPath
}

PYTHON {
    import datetime
    import os
    
    # Use standard libraries
    current_time = datetime.datetime.now()
    print(f"Current time: {current_time}")
    
    temp_path = os.path.dirname(os.path.realpath(__file__))
    print(f"Current directory: {temp_path}")
}
"""
            let engine = FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 3 // META + 2 language blocks
        }

printfn "🧪 Language Block Integration Tests Loaded"
printfn "==========================================="
printfn "✅ F# execution tests"
printfn "✅ Python execution tests"
printfn "✅ JavaScript execution tests"
printfn "✅ C# execution tests"
printfn "✅ Multi-language tests"
printfn "✅ Complex computation tests"
printfn "✅ Error handling tests"
printfn "✅ External library tests"
