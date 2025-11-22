# FParsec Parser Implementation Plan

## 1. Fix Property Parsing

### 1.1 Trailing Comma Handling
- [ ] Modify the property parser to properly handle trailing commas
- [ ] Update the property parser to ignore whitespace after commas
- [ ] Add test cases for properties with and without trailing commas
- [ ] Ensure consistent behavior with the original parser

### 1.2 Property Value Parsing
- [ ] Fix string literal parsing to handle escaped quotes
- [ ] Improve number parsing to handle scientific notation
- [ ] Add support for null values in properties
- [ ] Ensure property names are correctly parsed with special characters

## 2. Fix Nested Block Parsing

### 2.1 Block Type Recognition
- [ ] Update the block type parser to handle all known block types
- [ ] Add support for custom block types with validation
- [ ] Ensure block names are correctly parsed
- [ ] Fix parsing of empty blocks

### 2.2 Nested Block Structure
- [ ] Fix the recursive parsing of nested blocks
- [ ] Ensure correct parent-child relationships
- [ ] Handle deeply nested block structures
- [ ] Add proper indentation handling for nested blocks

## 3. Add Content Block Support

### 3.1 Raw Content Parsing
- [ ] Implement a parser for raw content blocks
- [ ] Add support for different content delimiters
- [ ] Handle whitespace preservation in content blocks
- [ ] Support mixed content and property blocks

### 3.2 Content Processing
- [ ] Add content preprocessing options
- [ ] Implement content trimming options
- [ ] Support for content interpolation
- [ ] Add content validation hooks

## 4. Improve Error Messages

### 4.1 Error Context
- [ ] Add line and column information to error messages
- [ ] Include snippet of the problematic code in errors
- [ ] Show caret pointing to the exact error location
- [ ] Add error codes for common parsing issues

### 4.2 Error Recovery
- [ ] Implement basic error recovery to continue parsing after errors
- [ ] Add suggestions for fixing common errors
- [ ] Group related errors to avoid overwhelming the user
- [ ] Provide context-aware error messages

## 5. Add Comment Support

### 5.1 Single-Line Comments
- [ ] Implement parser for single-line comments (// style)
- [ ] Handle comments at the end of lines
- [ ] Preserve comments in the AST (optional)
- [ ] Add tests for various comment patterns

### 5.2 Multi-Line Comments
- [ ] Implement parser for multi-line comments (/* */ style)
- [ ] Handle nested multi-line comments
- [ ] Support for documentation comments
- [ ] Add tests for complex comment scenarios

## Implementation Details

### Parser Combinators
```fsharp
// Single-line comment parser
let singleLineComment = 
    pstring "//" >>. manyCharsTill anyChar (newline <|> eof) |>> ignore

// Multi-line comment parser
let multiLineComment =
    between (pstring "/*") (pstring "*/") 
        (manyCharsTill anyChar (attempt (pstring "*/"))) |>> ignore

// Comment parser (either single-line or multi-line)
let comment = (attempt singleLineComment <|> attempt multiLineComment) .>> ws

// Updated property parser with comment support
let property =
    identifier .>> ws .>> many comment .>> str_ws ":" .>>. 
    propertyValue .>> optional (str_ws ",") .>> ws .>> many comment
```

### Error Handling Improvements
```fsharp
// Enhanced error reporting
let withError errorMsg p =
    p <?> errorMsg

// Parser with position information for better error messages
let withPos p =
    getPosition .>>. p .>>. getPosition
    |>> fun ((startPos, result), endPos) -> 
        { Result = result; StartPos = startPos; EndPos = endPos }

// Example usage
let propertyWithError =
    withError "Expected property in format 'name: value'" 
        (withPos property)
```

## Testing Strategy

1. Create unit tests for each parser component
2. Add integration tests for complete program parsing
3. Compare results with the original parser
4. Test with real-world TARS programs
5. Add edge case tests for error handling
6. Benchmark performance against the original parser
