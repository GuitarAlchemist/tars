# TARS Metascript Examples

This directory contains example metascripts that demonstrate various features of the TARS metascript system.

## Basic Examples

### example_metascript.tars

A basic example that demonstrates the core features of the TARS metascript system, including:
- Variable declarations
- F# code blocks
- Function calls
- Conditional expressions
- Loops
- Binary operations

### variable_interpolation_example.tars

An example that demonstrates variable interpolation in metascripts, including:
- Variable interpolation in string literals
- Variable interpolation in property values
- Variable interpolation in function arguments
- Variable interpolation in F# code blocks
- Variable interpolation in nested structures

## Advanced Examples

### tars_integration_example.tars

A more advanced example that demonstrates integration with the TARS engine, including:
- Code analysis
- Test generation
- Report generation
- Integration with TARS engine services

## Usage

To execute these examples, use the `MetascriptService` class:

```fsharp
// Create a metascript service
let logger = // ... create a logger
let metascriptService = MetascriptService(logger)

// Read the metascript file
let metascript = File.ReadAllText("path/to/example.tars")

// Create a context with variables
let context = {
    Variables = Map.ofList [
        "name", "World" :> obj
        "count", 5 :> obj
    ]
    Functions = Map.empty
    WorkingDirectory = Directory.GetCurrentDirectory()
    Metadata = Map.empty
}

// Execute the metascript
let result = metascriptService.ExecuteMetascriptAsync(metascript, context).Result

// Check the result
if result.Success then
    printfn "Metascript executed successfully"
    printfn "Result: %A" result.Result
    printfn "Output: %s" result.Output
    printfn "Modified variables:"
    for KeyValue(name, value) in result.ModifiedVariables do
        printfn "  %s = %A" name value
else
    printfn "Metascript execution failed"
    printfn "Errors:"
    for error in result.Errors do
        printfn "  %s" error
```
