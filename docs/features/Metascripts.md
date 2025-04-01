# TARS Metascripts

TARS Metascripts provide a powerful way to create complex workflows that combine multiple AI capabilities, including collaboration with other AI systems like Augment Code via MCP.

## Overview

Metascripts are written in the TARS Domain Specific Language (DSL) and allow you to:

- Define variables and configurations
- Execute prompts with LLMs
- Perform conditional logic (IF/ELSE)
- Interact with other AI systems via MCP
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
