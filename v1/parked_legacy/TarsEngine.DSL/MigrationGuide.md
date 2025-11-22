# Migration Guide: Original Parser to FParsec-based Parser

This guide will help you migrate from the original TARS DSL parser to the new FParsec-based parser.

## Overview

The FParsec-based parser is a drop-in replacement for the original parser, with several advantages:

- **Better error messages**: Error messages with line and column information
- **String interpolation**: Support for string interpolation with `${...}` syntax
- **Raw string literals**: Support for raw string literals with triple-quote delimiters
- **Content blocks**: Support for raw content blocks with triple-backtick delimiters
- **Comments**: Support for single-line (`//`) and multi-line (`/* */`) comments
- **Variable references**: Support for referencing variables with `@variable_name` syntax
- **Expressions**: Support for mathematical expressions with operators like `+`, `-`, `*`, `/`, `%`, `^`
- **Imports and includes**: Support for importing and including other TARS files
- **Templates**: Support for defining and using templates
- **Error recovery**: Support for recovering from errors and continuing parsing
- **Error suggestions**: Support for suggesting fixes for common errors

## Migration Steps

### Step 1: Update Your Code to Use the FParsec-based Parser

#### Option 1: Use the FParsec-based Parser Directly

```fsharp
// Before
open TarsEngine.DSL

let program = Parser.parseFile "path/to/file.tars"

// After
open TarsEngine.DSL

let program = FParsecParser.parseFile "path/to/file.tars"
```

#### Option 2: Use the Unified Parser with Configuration

```fsharp
// Before
open TarsEngine.DSL

let program = Parser.parseFile "path/to/file.tars"

// After
open TarsEngine.DSL
open TarsEngine.DSL.ParserConfiguration

// Set the parser type to use
ParserConfiguration.setParserType ParserType.FParsec

// Parse a DSL file with the current configuration
let program = UnifiedParser.parseFileWithCurrentConfig "path/to/file.tars"
```

### Step 2: Update Your TARS DSL Code (Optional)

The FParsec-based parser is backward compatible with the original parser, so your existing TARS DSL code should work without changes. However, you may want to take advantage of the new features offered by the FParsec-based parser.

#### String Interpolation

```
// Before
VARIABLE message {
    value: "Hello, " + @name + "!"
}

// After
VARIABLE message {
    value: "Hello, ${name}!"
}
```

#### Raw String Literals

```
// Before
VARIABLE message {
    value: "This is a multi-line string.\nIt can contain special characters like \\, \", and \n."
}

// After
VARIABLE message {
    value: """
This is a multi-line string.
It can contain special characters like \, ", and newlines.
"""
}
```

#### Comments

```
// Before
VARIABLE x {
    value: 42
}

// After
// This is a single-line comment
VARIABLE x {
    value: 42, // This is a comment after a property
    /* This is a multi-line comment
       that spans multiple lines */
    description: "Variable x"
}
```

#### Imports and Includes

```
// Before
// No direct support for imports and includes

// After
// Import another TARS file
IMPORT {
    "path/to/file.tars"
}

// Include another TARS file
INCLUDE {
    "path/to/file.tars"
}
```

#### Templates

```
// Before
// No direct support for templates

// After
// Define a template
TEMPLATE button {
    type: "button",
    style: "primary",
    text: "Click me",
    action: "submit"
}

// Use the template
USE_TEMPLATE button {
    text: "Submit",
    action: "submit_form"
}
```

### Step 3: Handle Error Messages

The FParsec-based parser provides more detailed error messages than the original parser. If your code handles error messages from the parser, you may need to update it to handle the new format.

```fsharp
// Before
try
    let program = Parser.parseFile "path/to/file.tars"
    // Process the program
with
| ex ->
    printfn "Error parsing TARS program: %s" ex.Message

// After
try
    let program = FParsecParser.parseFile "path/to/file.tars"
    // Process the program
with
| ex ->
    printfn "Error parsing TARS program: %s" ex.Message
    // The error message will include line and column information
    // and suggestions for fixing the error
```

## Examples

### Example 1: Basic Parsing

```fsharp
// Before
open TarsEngine.DSL

let program = Parser.parseFile "path/to/file.tars"

for block in program.Blocks do
    printfn "Block type: %A" block.Type

// After
open TarsEngine.DSL

let program = FParsecParser.parseFile "path/to/file.tars"

for block in program.Blocks do
    printfn "Block type: %A" block.Type
```

### Example 2: Error Handling

```fsharp
// Before
open TarsEngine.DSL

try
    let program = Parser.parseFile "path/to/file.tars"
    // Process the program
with
| ex ->
    printfn "Error parsing TARS program: %s" ex.Message

// After
open TarsEngine.DSL

try
    let program = FParsecParser.parseFile "path/to/file.tars"
    // Process the program
with
| ex ->
    printfn "Error parsing TARS program: %s" ex.Message
```

### Example 3: Using the Unified Parser

```fsharp
// Before
open TarsEngine.DSL

let program = Parser.parseFile "path/to/file.tars"

// After
open TarsEngine.DSL
open TarsEngine.DSL.ParserConfiguration

// Set the parser type to use
ParserConfiguration.setParserType ParserType.FParsec

// Parse a DSL file with the current configuration
let program = UnifiedParser.parseFileWithCurrentConfig "path/to/file.tars"
```

### Example 4: Using the Unified Parser with Configuration

```fsharp
// Before
open TarsEngine.DSL

let program = Parser.parseFile "path/to/file.tars"

// After
open TarsEngine.DSL
open TarsEngine.DSL.ParserConfiguration

// Create a custom parser configuration
let config = {
    ParserType = ParserType.FParsec
    ResolveImportsAndIncludes = true
    ValidateProgram = true
    OptimizeProgram = false
}

// Parse a DSL file with the custom configuration
let program = UnifiedParser.parseFile "path/to/file.tars" (Some config)
```

## Troubleshooting

### Issue: The FParsec-based parser reports errors in my TARS DSL code that the original parser didn't

The FParsec-based parser is more strict than the original parser and may report errors in code that the original parser accepted. This is usually a good thing, as it helps you catch errors in your code that the original parser missed.

If you encounter errors in your TARS DSL code when using the FParsec-based parser, check the error messages for suggestions on how to fix the errors. The FParsec-based parser provides detailed error messages with line and column information and suggestions for fixing common errors.

### Issue: The FParsec-based parser doesn't support a feature that I'm using

The FParsec-based parser is designed to be backward compatible with the original parser, so it should support all the features of the original parser. If you encounter a feature that the FParsec-based parser doesn't support, please report it as an issue.

### Issue: The FParsec-based parser is slower than the original parser

The FParsec-based parser is generally faster than the original parser, especially for large TARS DSL files. However, if you encounter performance issues, you can try the following:

- Use the original parser for performance-critical code
- Use the FParsec-based parser with error recovery disabled
- Use the FParsec-based parser with validation disabled

```fsharp
open TarsEngine.DSL
open TarsEngine.DSL.ParserConfiguration

// Create a custom parser configuration
let config = {
    ParserType = ParserType.FParsec
    ResolveImportsAndIncludes = false
    ValidateProgram = false
    OptimizeProgram = true
}

// Parse a DSL file with the custom configuration
let program = UnifiedParser.parseFile "path/to/file.tars" (Some config)
```

## Conclusion

The FParsec-based parser is a drop-in replacement for the original parser, with several advantages. It is backward compatible with the original parser, so your existing TARS DSL code should work without changes. However, you may want to take advantage of the new features offered by the FParsec-based parser.

If you encounter any issues when migrating from the original parser to the FParsec-based parser, please report them as issues.
