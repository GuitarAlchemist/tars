# TARS DSL Common Parsing Patterns

This document provides examples of common parsing patterns for the TARS DSL.

## Basic Parsing

### Parsing a TARS Program from a String

```fsharp
open TarsEngine.DSL

// Parse a TARS program from a string
let code = """
CONFIG {
    name: "My Program",
    version: "1.0"
}

VARIABLE x {
    value: 42
}
"""

// Parse with the original parser
let program = Parser.parse code

// Parse with the FParsec-based parser
let program = FParsecParser.parse code

// Parse with the unified parser
let program = UnifiedParser.parseWithCurrentConfig code
```

### Parsing a TARS Program from a File

```fsharp
open TarsEngine.DSL

// Parse a TARS program from a file
let program = Parser.parseFile "path/to/file.tars"

// Parse with the FParsec-based parser
let program = FParsecParser.parseFile "path/to/file.tars"

// Parse with the unified parser
let program = UnifiedParser.parseFileWithCurrentConfig "path/to/file.tars"
```

## Accessing Parsed Data

### Accessing Blocks

```fsharp
open TarsEngine.DSL

// Parse a TARS program
let program = Parser.parseFile "path/to/file.tars"

// Access the blocks
for block in program.Blocks do
    printfn "Block type: %A" block.Type
    
    // Access the block name
    match block.Name with
    | Some name -> printfn "Block name: %s" name
    | None -> printfn "Block has no name"
    
    // Access the block content
    if block.Content <> "" then
        printfn "Block content: %s" block.Content
    
    // Access the block properties
    for KeyValue(key, value) in block.Properties do
        printfn "Property: %s = %A" key value
    
    // Access the nested blocks
    for nestedBlock in block.NestedBlocks do
        printfn "Nested block type: %A" nestedBlock.Type
```

### Accessing Properties

```fsharp
open TarsEngine.DSL

// Parse a TARS program
let program = Parser.parseFile "path/to/file.tars"

// Find a block by type
let configBlock = program.Blocks |> List.find (fun block -> block.Type = BlockType.Config)

// Access a property
match configBlock.Properties.TryFind "name" with
| Some (StringValue name) -> printfn "Name: %s" name
| _ -> printfn "Name property not found or not a string"

// Access a nested property
match configBlock.Properties.TryFind "settings" with
| Some (ObjectValue settings) ->
    match settings.TryFind "debug" with
    | Some (BoolValue debug) -> printfn "Debug: %b" debug
    | _ -> printfn "Debug property not found or not a boolean"
| _ -> printfn "Settings property not found or not an object"

// Access a list property
match configBlock.Properties.TryFind "tags" with
| Some (ListValue tags) ->
    for tag in tags do
        match tag with
        | StringValue tag -> printfn "Tag: %s" tag
        | _ -> printfn "Tag is not a string"
| _ -> printfn "Tags property not found or not a list"
```

### Accessing Variable References

```fsharp
open TarsEngine.DSL

// Parse a TARS program
let program = Parser.parseFile "path/to/file.tars"

// Find a block by type and name
let variableBlock = 
    program.Blocks 
    |> List.find (fun block -> 
        block.Type = BlockType.Variable && 
        block.Name = Some "y")

// Access a variable reference
match variableBlock.Properties.TryFind "value" with
| Some (VariableReference name) -> printfn "References variable: %s" name
| _ -> printfn "Value property not found or not a variable reference"
```

### Accessing Expressions

```fsharp
open TarsEngine.DSL

// Parse a TARS program
let program = Parser.parseFile "path/to/file.tars"

// Find a block by type and name
let variableBlock = 
    program.Blocks 
    |> List.find (fun block -> 
        block.Type = BlockType.Variable && 
        block.Name = Some "z")

// Access an expression
match variableBlock.Properties.TryFind "value" with
| Some (ExpressionValue expr) ->
    match expr with
    | BinaryOp(left, op, right) -> 
        printfn "Binary operation: %A %s %A" left op right
    | UnaryOp(op, value) -> 
        printfn "Unary operation: %s %A" op value
    | _ -> printfn "Expression is not a binary or unary operation"
| _ -> printfn "Value property not found or not an expression"
```

## Error Handling

### Handling Parsing Errors

```fsharp
open TarsEngine.DSL
open System

// Try to parse a TARS program
try
    let code = """
    CONFIG {
        name: "My Program",
        version: "1.0"
    }
    
    VARIABLE x {
        value: 42
    }
    """
    
    let program = FParsecParser.parse code
    
    // Process the program
    printfn "Program has %d blocks" program.Blocks.Length
catch
| ex ->
    printfn "Error parsing TARS program: %s" ex.Message
```

### Validating a Parsed Program

```fsharp
open TarsEngine.DSL

// Parse a TARS program
let program = Parser.parseFile "path/to/file.tars"

// Validate the program
let validateProgram (program: TarsProgram) =
    // Check if the program has a CONFIG block
    let hasConfigBlock = 
        program.Blocks 
        |> List.exists (fun block -> block.Type = BlockType.Config)
    
    if not hasConfigBlock then
        printfn "Warning: Program does not have a CONFIG block"
    
    // Check if all VARIABLE blocks have a name
    let variableBlocksWithoutName = 
        program.Blocks 
        |> List.filter (fun block -> 
            block.Type = BlockType.Variable && 
            block.Name = None)
    
    if not (List.isEmpty variableBlocksWithoutName) then
        printfn "Warning: Program has %d VARIABLE blocks without a name" variableBlocksWithoutName.Length
    
    // Check if all FUNCTION blocks have a name
    let functionBlocksWithoutName = 
        program.Blocks 
        |> List.filter (fun block -> 
            block.Type = BlockType.Function && 
            block.Name = None)
    
    if not (List.isEmpty functionBlocksWithoutName) then
        printfn "Warning: Program has %d FUNCTION blocks without a name" functionBlocksWithoutName.Length
    
    // Check if all FUNCTION blocks have a RETURN block
    let functionBlocksWithoutReturn = 
        program.Blocks 
        |> List.filter (fun block -> 
            block.Type = BlockType.Function && 
            not (block.NestedBlocks |> List.exists (fun nestedBlock -> nestedBlock.Type = BlockType.Return)))
    
    if not (List.isEmpty functionBlocksWithoutReturn) then
        printfn "Warning: Program has %d FUNCTION blocks without a RETURN block" functionBlocksWithoutReturn.Length

// Validate the program
validateProgram program
```

## Advanced Parsing

### Parsing with Configuration

```fsharp
open TarsEngine.DSL
open TarsEngine.DSL.ParserConfiguration

// Create a custom parser configuration
let config = {
    ParserType = ParserType.FParsec
    ResolveImportsAndIncludes = true
    ValidateProgram = true
    OptimizeProgram = false
}

// Parse a TARS program with the custom configuration
let program = UnifiedParser.parseFile "path/to/file.tars" (Some config)
```

### Parsing with Imports and Includes

```fsharp
open TarsEngine.DSL
open TarsEngine.DSL.ParserConfiguration

// Set the parser type to FParsec
ParserConfiguration.setParserType ParserType.FParsec

// Set whether to resolve imports and includes
ParserConfiguration.setResolveImportsAndIncludes true

// Parse a TARS program with imports and includes
let program = UnifiedParser.parseFileWithCurrentConfig "path/to/file.tars"
```

### Parsing with Error Recovery

```fsharp
open TarsEngine.DSL
open TarsEngine.DSL.ParserConfiguration

// Set the parser type to FParsec
ParserConfiguration.setParserType ParserType.FParsec

// Parse a TARS program with error recovery
let code = """
CONFIG {
    name: "My Program",
    version: "1.0"
}

VARIABLE x {
    value: 42
}

VARIABLE y {
    value: @x + @z  // Error: z is not defined
}
"""

let program = UnifiedParser.parseWithCurrentConfig code

// Check if the program has any blocks
if program.Blocks.Length > 0 then
    printfn "Program has %d blocks" program.Blocks.Length
else
    printfn "Program has no blocks"
```

## Common Error Messages and Solutions

### Unclosed String Literal

```
Error in Ln: 5 Col: 10
    name: "My Program
         ^
Note: The error occurred at the end of the line.
Expected '"'.
```

**Solution**: Add a closing double quote to the string literal.

```
name: "My Program"
```

### Unclosed Block

```
Error in Ln: 5 Col: 1
VARIABLE x {
^
Note: The error occurred at the end of the line.
Expected '}'.
```

**Solution**: Add a closing brace to the block.

```
VARIABLE x {
    value: 42
}
```

### Invalid Property Value

```
Error in Ln: 5 Col: 12
    value: @
           ^
Note: The error occurred at the end of the line.
Expected identifier.
```

**Solution**: Add an identifier after the @ symbol.

```
value: @x
```

### Invalid Block Type

```
Error in Ln: 5 Col: 1
VARIABEL x {
^
Note: The error occurred at the end of the line.
Unknown block type: VARIABEL.
```

**Solution**: Fix the block type.

```
VARIABLE x {
    value: 42
}
```

### Invalid Expression

```
Error in Ln: 5 Col: 12
    value: x +
           ^
Note: The error occurred at the end of the line.
Expected expression.
```

**Solution**: Complete the expression.

```
value: x + y
```

### Invalid Import Path

```
Error in Ln: 5 Col: 12
IMPORT {
    "path/to/file.tars
    ^
Note: The error occurred at the end of the line.
Expected '"'.
```

**Solution**: Add a closing double quote to the import path.

```
IMPORT {
    "path/to/file.tars"
}
```

### File Not Found

```
Error: File not found: path/to/file.tars
```

**Solution**: Check that the file exists and the path is correct.

```
IMPORT {
    "correct/path/to/file.tars"
}
```
