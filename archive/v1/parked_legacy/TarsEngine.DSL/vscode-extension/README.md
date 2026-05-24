# TARS DSL for Visual Studio Code

This extension provides syntax highlighting for the TARS Domain Specific Language (DSL).

## Features

- Syntax highlighting for TARS DSL files
- Support for block types, properties, strings, numbers, booleans, null, variables, expressions, and content blocks
- Support for single-line and multi-line comments
- Support for string interpolation
- Support for raw string literals (triple-quoted strings)
- Support for variable references
- Support for expressions
- Support for content blocks

## Installation

1. Open Visual Studio Code
2. Press `Ctrl+Shift+X` to open the Extensions view
3. Search for "TARS DSL"
4. Click Install

## Usage

1. Create a new file with the `.tars` extension
2. Start writing TARS DSL code
3. Enjoy syntax highlighting!

## Example

```tars
CONFIG {
    name: "My Config",
    version: "1.0",
    description: "A sample configuration"
}

FUNCTION add {
    parameters: "a, b",
    description: "Adds two numbers",
    
    VARIABLE result {
        value: @a + @b,
        description: "The result of adding a and b"
    }
    
    RETURN {
        value: @result
    }
}

PROMPT {
```
This is a content block.
It can contain multiple lines of text.
It can also contain special characters like { and }.
```
}
```

## License

This extension is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Release Notes

### 0.0.1

Initial release of TARS DSL for Visual Studio Code.
