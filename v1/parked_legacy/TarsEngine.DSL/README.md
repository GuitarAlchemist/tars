# TarsEngine.DSL

This project contains the parser and interpreter for the TARS Domain Specific Language (DSL).

## Overview

The TARS DSL is a block-based language that allows you to define and execute AI workflows in a simple and intuitive way. Each block represents a specific operation or configuration, and blocks are executed in sequence.

## Features

- **Block-based syntax**: Define workflows using blocks like CONFIG, PROMPT, ACTION, etc.
- **Property-based configuration**: Configure blocks using properties
- **Extensible**: Easy to add new block types and properties
- **Integration with TARS CLI**: Run DSL files using the TARS CLI

## Usage

### Parse a DSL File

```fsharp
open TarsEngine.DSL

// Parse a DSL file
let program = Parser.parseFile "path/to/file.tars"

// Access the blocks
for block in program.Blocks do
    printfn "Block type: %A" block.Type
    
    // Access the properties
    for KeyValue(key, value) in block.Properties do
        printfn "Property: %s = %A" key value
```

### Execute a DSL Program

```fsharp
open TarsEngine.DSL

// Parse a DSL file
let program = Parser.parseFile "path/to/file.tars"

// Execute the program
let result = Interpreter.execute program

// Handle the result
match result with
| ExecutionResult.Success value -> printfn "Success: %A" value
| ExecutionResult.Error error -> printfn "Error: %s" error
```

## Block Types

The TARS DSL supports the following block types:

- `CONFIG`: Define global configuration for the workflow
- `PROMPT`: Define a prompt to send to the AI model
- `ACTION`: Define an action to perform
- `TASK`: Define a task to perform
- `AGENT`: Define an agent that can perform tasks
- `AUTO_IMPROVE`: Define an auto-improvement task

## Documentation

For more information, see the [TARS DSL Guide](../docs/DSL/DSL-Guide.md).
