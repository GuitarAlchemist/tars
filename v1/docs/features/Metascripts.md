# TARS Metascripts

TARS Metascripts provide a powerful way to create complex workflows that combine multiple AI capabilities, including collaboration with other AI systems like Augment Code via MCP.

## Overview

Metascripts are written in the TARS Domain Specific Language (DSL) and allow you to:

- Define variables and configurations
- Execute prompts with LLMs
- Perform conditional logic (IF/ELSE, WHILE, FOR)
- Define and call functions
- Handle errors with TRY/CATCH blocks
- Interact with other AI systems via MCP
- Perform file operations (read/write)
- Make HTTP requests
- Execute shell commands
- Log information and results

## Metascript Structure

A typical metascript consists of several block types:

```
DESCRIBE {
    name: "Example Metascript"
    version: "1.0"
    description: "A simple example metascript"
    author: "TARS Team"
    date: "2025-03-31"
}

CONFIG {
    model: "llama3"
    temperature: 0.7
    max_tokens: 1000
}

VARIABLE task {
    value: "Example task"
}

PROMPT {
    text: "Perform the following task: ${task}"
    result_variable: "prompt_result"
}

ACTION {
    type: "log"
    message: "Task result: ${prompt_result}"
}
```

## Block Types

### DESCRIBE

The `DESCRIBE` block provides metadata about the metascript:

```
DESCRIBE {
    name: "Example Metascript"
    version: "1.0"
    description: "A simple example metascript"
    author: "TARS Team"
    date: "2025-03-31"
}
```

### CONFIG

The `CONFIG` block sets configuration parameters for the metascript execution:

```
CONFIG {
    model: "llama3"
    temperature: 0.7
    max_tokens: 1000
}
```

### VARIABLE

The `VARIABLE` block defines variables that can be used throughout the metascript:

```
VARIABLE task {
    value: "Example task"
}
```

### PROMPT

The `PROMPT` block sends a prompt to an LLM and stores the result:

```
PROMPT {
    text: "Perform the following task: ${task}"
    result_variable: "prompt_result"
}
```

### ACTION

The `ACTION` block performs various actions:

```
ACTION {
    type: "log"
    message: "Task result: ${prompt_result}"
}
```

Action types include:
- `log`: Log a message
- `mcp_send`: Send a request to another AI system via MCP
- `mcp_receive`: Receive a request from another AI system via MCP
- `file_read`: Read a file and store its content in a variable
- `file_write`: Write content to a file
- `http_request`: Make an HTTP request and store the response in a variable
- `shell_execute`: Execute a shell command and store the output in a variable

### IF/ELSE

The `IF` and `ELSE` blocks provide conditional logic:

```
IF {
    condition: "${prompt_result != ''}"

    ACTION {
        type: "log"
        message: "Prompt returned a result"
    }
}
ELSE {
    ACTION {
        type: "log"
        message: "Prompt returned no result"
    }
}
```

### WHILE

The `WHILE` block provides loop functionality based on a condition:

```
VARIABLE counter {
    value: 0
}

WHILE {
    condition: "${counter < 5}"

    ACTION {
        type: "log"
        message: "Counter: ${counter}"
    }

    VARIABLE counter {
        value: "${counter + 1}"
    }
}
```

### FOR

The `FOR` block provides loop functionality with a variable, start, end, and step values:

```
FOR {
    variable: "i"
    from: 1
    to: 5
    step: 1

    ACTION {
        type: "log"
        message: "Number: ${i}"
    }
}
```

### FUNCTION

The `FUNCTION` block defines a reusable function:

```
FUNCTION add {
    parameters: "a, b"

    RETURN {
        value: "${a + b}"
    }
}
```

### CALL

The `CALL` block calls a defined function:

```
CALL {
    function: "add"
    arguments: {
        a: 2
        b: 3
    }
    result_variable: "sum"
}
```

### TRY/CATCH

The `TRY` and `CATCH` blocks provide error handling:

```
TRY {
    ACTION {
        type: "file_read"
        path: "non_existent_file.txt"
        result_variable: "file_content"
    }

    CATCH {
        ACTION {
            type: "log"
            message: "Caught an error: ${error}"
        }
    }
}
```

### RETURN

The `RETURN` block returns a value from a function:

```
RETURN {
    value: "${result}"
}
```

## Variable Substitution

Variables can be referenced using the `${variable_name}` syntax. This works in most string values throughout the metascript.

## MCP Integration

Metascripts can interact with other AI systems like Augment Code via MCP:

```
ACTION {
    type: "mcp_send"
    target: "augment"
    action: "code_generation"
    parameters: {
        language: "typescript"
        task: "${task}"
    }
    result_variable: "code_result"
}
```

## Examples

See the `Examples/metascripts` directory for example metascripts, including:

- `hello_world.tars`: A simple hello world example
- `tars_augment_collaboration.tars`: An example of TARS-Augment collaboration via MCP

## CLI Commands

The TARS CLI provides commands for working with metascripts:

```
tarscli metascript validate <path>   # Validate a metascript
tarscli metascript execute <path>    # Execute a metascript
```

You can also run a demo of the metascript capabilities:

```
tarscli demo --type metascript
```

## Future Enhancements

Future enhancements to the metascript system may include:

- More action types
- More complex control flow (loops, functions)
- Error handling blocks
- Integration with more external systems
- Visual metascript editor
