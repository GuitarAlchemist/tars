# TARS DSL Guide

The TARS Domain Specific Language (DSL) is designed to define AI workflows, agent behaviors, and self-improvement processes. This guide provides an overview of the language and its usage.

## Basic Structure

TARS DSL is a block-based language. Each block has a type, an optional name, and a set of properties and nested blocks.

```
BLOCK_TYPE [name] {
    property1: value1
    property2: value2

    NESTED_BLOCK {
        nested_property: value
    }
}
```

## Block Types

The TARS DSL supports the following block types:

### CONFIG

Defines configuration settings for the TARS environment.

```
CONFIG {
    model: "llama3"
    temperature: 0.7
    max_tokens: 1000
}
```

### PROMPT

Defines a prompt to be sent to an AI model.

```
PROMPT {
    text: "What is the capital of France?"
    role: "user"  // Optional, can be "system", "user", or "assistant"
}
```

### ACTION

Defines an action to be executed.

```
ACTION {
    type: "generate"  // Can be "generate", "chat", "execute", etc.
    model: "llama3"   // Optional, defaults to the model in CONFIG
}
```

### TASK

Defines a task to be performed.

```
TASK FindInformation {
    description: "Find information about the history of Paris"
    agent: "researcher"  // Optional, the agent to perform the task
}
```

### AGENT

Defines an agent with capabilities.

```
AGENT researcher {
    description: "A research agent that can find information on the web"
    capabilities: ["search", "summarize", "analyze"]
}
```

### AUTO_IMPROVE

Defines an auto-improvement process.

```
AUTO_IMPROVE {
    target: "code"  // What to improve
    method: "analyze"  // How to improve it
    iterations: 3  // How many iterations to run
}
```

### DESCRIBE

Provides metadata about the TARS DSL script.

```
DESCRIBE {
    name: "My TARS Script"
    version: "1.0"
    author: "TARS User"
    description: "A script that does something useful"
}
```

### SPAWN_AGENT

Creates a new agent instance.

```
SPAWN_AGENT {
    id: "agent1"
    type: "researcher"
}
```

### MESSAGE

Sends a message to an agent.

```
MESSAGE {
    agent: "agent1"
    text: "Find information about the history of Paris"
}
```

### SELF_IMPROVE

Triggers self-improvement for an agent.

```
SELF_IMPROVE {
    agent: "agent1"
    instructions: "Improve your search capabilities"
}
```

### TARS

A container block for TARS-specific operations.

```
TARS {
    // Nested blocks
}
```

### COMMUNICATION

Defines communication settings.

```
COMMUNICATION {
    protocol: "http"
    endpoint: "http://localhost:8080"
}
```

### VARIABLE

Defines a variable with a name and value.

```
VARIABLE my_var {
    value: "Hello, world!"
}
```

### IF

Executes nested blocks if a condition is true.

```
IF {
    condition: true

    PROMPT {
        text: "This will be executed if the condition is true"
    }
}
```

### ELSE

Executes nested blocks if the preceding IF block's condition is false.

```
ELSE {
    PROMPT {
        text: "This will be executed if the condition is false"
    }
}
```

### FOR

Executes nested blocks for each value in a range.

```
FOR {
    variable: "item"
    range: ["a", "b", "c"]

    PROMPT {
        text: "Processing item"
    }
}
```

### WHILE

Executes nested blocks while a condition is true.

```
WHILE {
    condition: true

    PROMPT {
        text: "This will be executed while the condition is true"
    }
}
```

### FUNCTION

Defines a reusable function.

```
FUNCTION process_data {
    parameters: ["data"]

    PROMPT {
        text: "Processing data"
    }

    RETURN {
        value: "Processed data"
    }
}
```

### RETURN

Returns a value from a function.

```
RETURN {
    value: "Result"
}
```

### IMPORT

Imports a module.

```
IMPORT {
    module: "utils"
}
```

### EXPORT

Exports a value.

```
EXPORT {
    name: "my_function"
}
```

## Nested Blocks

Blocks can be nested to create hierarchical structures. For example, an AGENT block can contain TASK blocks, which can contain ACTION blocks.

```
AGENT researcher {
    description: "A research agent"

    TASK FindInformation {
        description: "Find information"

        ACTION {
            type: "web_search"
            query: "history of Paris"
        }
    }
}
```

## Property Values

Properties can have the following types of values:

- **String**: Enclosed in double quotes, e.g., `"hello"`
- **Number**: Integer or floating-point, e.g., `42` or `3.14`
- **Boolean**: `true` or `false`
- **Array**: Enclosed in square brackets, e.g., `["a", "b", "c"]`
- **Object**: Enclosed in curly braces, e.g., `{ key: "value" }`

## Examples

### Basic Example

```
DESCRIBE {
    name: "TARS Basic Example"
    version: "1.0"
    author: "TARS User"
    description: "A basic example of the TARS DSL"
}

CONFIG {
    model: "llama3"
    temperature: 0.7
    max_tokens: 1000
}

PROMPT {
    text: "What is the capital of France?"
}

ACTION {
    type: "generate"
    model: "llama3"
}
```

### Chat Example

```
DESCRIBE {
    name: "TARS Chat Example"
    version: "1.0"
    author: "TARS User"
    description: "A chat example of the TARS DSL"
}

CONFIG {
    model: "llama3"
    temperature: 0.7
    max_tokens: 1000
}

TARS {
    PROMPT {
        text: "You are a helpful assistant. Answer the user's questions accurately and concisely."
        role: "system"
    }

    PROMPT {
        text: "What is the capital of France?"
        role: "user"
    }

    ACTION {
        type: "chat"
        model: "llama3"
    }
}
```

### Agent Example

```
CONFIG {
    model: "llama3"
    temperature: 0.7
    max_tokens: 1000
}

AGENT researcher {
    description: "A research agent that can find information on the web"
    capabilities: ["search", "summarize", "analyze"]

    TASK FindInformation {
        description: "Find information about the history of Paris"

        ACTION {
            type: "web_search"
            query: "history of Paris"
        }
    }
}

ACTION {
    type: "execute"
    task: "FindInformation"
    agent: "researcher"
}
```

## CLI Usage

The TARS CLI provides commands for working with DSL files:

```
# Run a DSL file
tarscli dsl run --file my_script.tars

# Validate a DSL file
tarscli dsl validate --file my_script.tars

# Generate a DSL template
tarscli dsl generate --type basic --output my_script.tars

# Generate EBNF for the TARS DSL
tarscli dsl ebnf --output tars_dsl.ebnf
```
