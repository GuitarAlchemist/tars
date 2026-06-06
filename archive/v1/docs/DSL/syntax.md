# TARS DSL Syntax Guide

This guide provides an overview of the basic syntax and structure of the TARS Domain Specific Language (DSL).

## Basic Structure

TARS DSL uses a block-based syntax with curly braces, similar to languages like C# and JavaScript. Each block has a type, an optional name, and content that can include properties, statements, and nested blocks.

```
BLOCK_TYPE [name] {
    property1: value1;
    property2: value2;
    
    NESTED_BLOCK {
        nestedProperty: nestedValue;
    }
}
```

## Comments

TARS DSL supports both single-line and multi-line comments:

```
// This is a single-line comment

/*
   This is a
   multi-line comment
*/
```

## Properties

Properties are key-value pairs that define attributes of a block:

```
property: value;
```

The value can be:
- String: `"Hello, World!"`
- Number: `42` or `3.14`
- Boolean: `true` or `false`
- Array: `[1, 2, 3]` or `["a", "b", "c"]`
- Object: `{ key1: value1, key2: value2 }`
- Identifier: `variableName`

## Statements

Statements are commands that perform actions:

```
let x = 42;
print("Hello, World!");
if (condition) {
    // Do something
}
```

## Blocks

Blocks are the primary organizational units in TARS DSL. Each block has a specific purpose and can contain properties, statements, and nested blocks.

### Top-Level Blocks

```
CONFIG {
    version: "1.0";
    author: "John Doe";
}

PROMPT {
    text: "Generate a list of ideas";
    model: "gpt-4";
}

ACTION {
    let result = processFile("example.cs");
    print(result);
}
```

### Nested Blocks

```
AGENT {
    name: "CodeAnalyzer";
    
    TASK {
        name: "AnalyzeCode";
        
        ACTION {
            let code = readFile("example.cs");
            return analyzeCode(code);
        }
    }
}
```

## Expressions

Expressions are combinations of values, variables, operators, and function calls that evaluate to a value:

```
let x = 10 + 20;
let y = x * 2;
let z = (x + y) / 2;
let isGreater = x > y;
let combined = "Value: " + x;
```

## Control Flow

TARS DSL supports standard control flow constructs:

### If Statements

```
if (condition) {
    // Do something
} else if (anotherCondition) {
    // Do something else
} else {
    // Default action
}
```

### For Loops

```
for (let i = 0; i < 10; i++) {
    // Do something
}
```

### While Loops

```
while (condition) {
    // Do something
}
```

## Function Calls

Function calls invoke built-in or user-defined functions:

```
let result = functionName(arg1, arg2);
print("Result: " + result);
```

## Variables

Variables store values that can be used later:

```
let name = "John";
let age = 30;
let isActive = true;
```

## Semicolons

Semicolons are optional at the end of statements and property definitions:

```
let x = 42;  // With semicolon
let y = 24   // Without semicolon
```

## Case Sensitivity

TARS DSL is case-sensitive. `variable`, `Variable`, and `VARIABLE` are all different identifiers.

## Whitespace

Whitespace (spaces, tabs, newlines) is generally ignored except when it's used to separate tokens:

```
let x=42;  // Valid
let y = 24; // Valid and more readable
```

## Naming Conventions

- Block types are in UPPERCASE: `CONFIG`, `PROMPT`, `ACTION`
- Block names are in PascalCase: `MyBlock`, `CodeAnalyzer`
- Properties and variables are in camelCase: `myProperty`, `userName`
- Constants are in UPPERCASE with underscores: `MAX_RETRY_COUNT`

## Example

Here's a complete example of a TARS DSL program:

```
CONFIG {
    version: "1.0";
    author: "John Doe";
    description: "Example TARS program";
}

PROMPT {
    text: "Generate a list of 5 ideas for improving code quality.";
    model: "gpt-4";
    temperature: 0.7;
}

ACTION {
    let result = processFile("example.cs");
    print(result);
    
    if (result.issues.length > 0) {
        for (let i = 0; i < result.issues.length; i++) {
            print("Issue " + (i + 1) + ": " + result.issues[i]);
        }
    } else {
        print("No issues found!");
    }
}
```

This example defines a configuration, a prompt for an AI model, and an action that processes a file and prints the results.
