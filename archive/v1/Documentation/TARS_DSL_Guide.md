# TARS DSL Guide

## Introduction

The TARS Domain-Specific Language (DSL) is a powerful scripting language designed for the TARS system. It allows users to create metascripts that can perform a wide range of operations, from simple text processing to complex code transformations and integrations with other systems.

This guide provides a comprehensive overview of the TARS DSL, including its syntax, capabilities, and examples.

## Basic Structure

A TARS metascript consists of a series of blocks, each with a specific type and purpose. The basic structure of a metascript is as follows:

```
DESCRIBE {
    name: "My Metascript"
    version: "1.0"
    description: "A simple metascript"
    author: "TARS Team"
    date: "2025-04-02"
}

CONFIG {
    model: "llama3"
    temperature: 0.7
    max_tokens: 1000
}

// Rest of the metascript...
```

## Block Types

The TARS DSL supports the following block types:

### DESCRIBE

The `DESCRIBE` block provides metadata about the metascript, including its name, version, description, author, and date.

```
DESCRIBE {
    name: "My Metascript"
    version: "1.0"
    description: "A simple metascript"
    author: "TARS Team"
    date: "2025-04-02"
}
```

### CONFIG

The `CONFIG` block specifies configuration settings for the metascript, such as the model to use, temperature, and maximum tokens.

```
CONFIG {
    model: "llama3"
    temperature: 0.7
    max_tokens: 1000
}
```

### VARIABLE

The `VARIABLE` block defines a variable that can be used throughout the metascript.

```
VARIABLE greeting {
    value: "Hello, World!"
}
```

Variables can be referenced using the `${variable_name}` syntax.

### ACTION

The `ACTION` block performs an action, such as logging a message, reading or writing a file, or making an HTTP request.

```
ACTION {
    type: "log"
    message: "Hello, World!"
}
```

#### Supported Action Types

- `log`: Logs a message to the console
- `file_read`: Reads a file and stores its content in a variable
- `file_write`: Writes content to a file
- `file_delete`: Deletes a file
- `http_request`: Makes an HTTP request
- `mcp_send`: Sends a message to an MCP target
- `mcp_receive`: Receives a message from an MCP target

### PROMPT

The `PROMPT` block sends a prompt to the language model and stores the response.

```
PROMPT {
    text: "What is the capital of France?"
    result_variable: "capital"
}
```

### IF/ELSE

The `IF` and `ELSE` blocks provide conditional execution based on a condition.

```
IF {
    condition: "${count > 3}"
    
    ACTION {
        type: "log"
        message: "Count is greater than 3"
    }
}
ELSE {
    ACTION {
        type: "log"
        message: "Count is not greater than 3"
    }
}
```

### FOR

The `FOR` block iterates over a collection or a range of values.

```
// Iterate over a collection
FOR {
    item: "fruit"
    collection: "Apple,Banana,Cherry,Date,Elderberry"
    
    ACTION {
        type: "log"
        message: "Fruit: ${fruit}"
    }
}

// Iterate over a range
FOR {
    item: "i"
    collection: "1,2,3,4,5"
    
    ACTION {
        type: "log"
        message: "Count: ${i}"
    }
}
```

### WHILE

The `WHILE` block executes a set of blocks repeatedly as long as a condition is true.

```
VARIABLE count {
    value: 0
}

WHILE {
    condition: "${count < 5}"
    
    ACTION {
        type: "log"
        message: "Count: ${count}"
    }
    
    VARIABLE count {
        value: "${count + 1}"
    }
}
```

### FUNCTION

The `FUNCTION` block defines a function that can be called from other parts of the metascript.

```
FUNCTION calculate_factorial {
    parameters: "n"
    
    VARIABLE result {
        value: 1
    }
    
    FOR {
        item: "i"
        collection: "1,2,3,4,5"
        
        VARIABLE result {
            value: "${result * i}"
        }
    }
    
    RETURN {
        value: "${result}"
    }
}
```

### CALL

The `CALL` block calls a function defined with the `FUNCTION` block.

```
CALL {
    function: "calculate_factorial"
    arguments: {
        n: 5
    }
    result_variable: "factorial_result"
}
```

### TRY/CATCH

The `TRY` and `CATCH` blocks provide error handling.

```
TRY {
    ACTION {
        type: "file_read"
        path: "non_existent_file.txt"
        result_variable: "file_content"
    }
}
CATCH {
    ACTION {
        type: "log"
        message: "Error reading file"
    }
}
```

### FSHARP

The `FSHARP` block allows embedding F# code within the metascript.

```
FSHARP {
    // Define a function
    let square x = x * x
    
    // Get a variable from the environment
    let count = environment.["count"].ToString() |> int
    
    // Calculate the square
    let result = square count
    
    // Return the result
    sprintf "The square of %d is %d" count result
}
```

The result of the F# code is stored in the special variable `_last_result`.

## File Operations

The TARS DSL provides several actions for working with files:

### Reading Files

```
ACTION {
    type: "file_read"
    path: "input.txt"
    result_variable: "file_content"
}
```

### Writing Files

```
ACTION {
    type: "file_write"
    path: "output.txt"
    content: "Hello, World!"
}
```

### Deleting Files

```
ACTION {
    type: "file_delete"
    path: "temp.txt"
}
```

## HTTP Requests

The TARS DSL allows making HTTP requests:

```
ACTION {
    type: "http_request"
    url: "https://api.example.com/data"
    method: "GET"
    headers: {
        "Content-Type": "application/json"
        "Authorization": "Bearer ${token}"
    }
    result_variable: "response"
}
```

## MCP Integration

The TARS DSL provides actions for integrating with the Model Context Protocol (MCP):

### Sending MCP Messages

```
ACTION {
    type: "mcp_send"
    target: "augment"
    action: "code_generation"
    data: {
        prompt: "Generate a function to calculate factorial"
        language: "csharp"
    }
    result_variable: "generated_code"
}
```

### Receiving MCP Messages

```
ACTION {
    type: "mcp_receive"
    timeout: 30
    result_variable: "mcp_request"
}
```

## Code Transformation

The TARS DSL can be used to transform code using metascript rules:

```
// Define a transformation rule
VARIABLE rule_content {
    value: "rule AddNullCheck {
    match: \"public int Divide\\(int a, int b\\)\\s*{\\s*return a / b;\\s*}\"
    replace: \"public int Divide(int a, int b)\\n    {\\n        if (b == 0)\\n        {\\n            throw new System.DivideByZeroException(\\\"Cannot divide by zero\\\");\\n        }\\n        return a / b;\\n    }\"
    requires: \"System\"
}"
}

// Save the rule to a file
ACTION {
    type: "file_write"
    path: "rule.meta"
    content: "${rule_content}"
}

// Apply the transformation using F#
FSHARP {
    // Load the rule
    let rulePath = "rule.meta"
    let rules = TarsEngineFSharp.MetascriptEngine.loadRules(rulePath)
    
    // Get the sample code
    let code = environment.["sample_code"].ToString()
    
    // Apply the transformation
    let transformedCode = 
        rules 
        |> List.fold (fun c rule -> TarsEngineFSharp.MetascriptEngine.applyRule rule c) code
    
    // Return the transformed code
    transformedCode
}
```

## Examples

### Hello World

```
DESCRIBE {
    name: "Hello World"
    version: "1.0"
    description: "A simple hello world example"
}

CONFIG {
    model: "llama3"
    temperature: 0.7
    max_tokens: 1000
}

VARIABLE message {
    value: "Hello, World!"
}

ACTION {
    type: "log"
    message: "${message}"
}
```

### File Operations

```
DESCRIBE {
    name: "File Operations"
    version: "1.0"
    description: "A demonstration of file operations"
}

CONFIG {
    model: "llama3"
    temperature: 0.7
    max_tokens: 1000
}

// Create a file
ACTION {
    type: "file_write"
    path: "temp.txt"
    content: "Hello, World!"
}

// Read the file
ACTION {
    type: "file_read"
    path: "temp.txt"
    result_variable: "file_content"
}

// Display the file content
ACTION {
    type: "log"
    message: "File content: ${file_content}"
}

// Delete the file
ACTION {
    type: "file_delete"
    path: "temp.txt"
}
```

### F# Integration

```
DESCRIBE {
    name: "F# Integration"
    version: "1.0"
    description: "A demonstration of F# integration"
}

CONFIG {
    model: "llama3"
    temperature: 0.7
    max_tokens: 1000
}

VARIABLE count {
    value: 5
}

FSHARP {
    // Define a function
    let square x = x * x
    
    // Get the count from the environment
    let count = environment.["count"].ToString() |> int
    
    // Calculate the square
    let result = square count
    
    // Return the result
    sprintf "The square of %d is %d" count result
}

VARIABLE result {
    value: "${_last_result}"
}

ACTION {
    type: "log"
    message: "${result}"
}
```

## Conclusion

The TARS DSL is a powerful tool for creating metascripts that can perform a wide range of operations. This guide provides a comprehensive overview of its capabilities, but there is much more to explore. Experiment with the examples and create your own metascripts to discover the full potential of the TARS DSL.
