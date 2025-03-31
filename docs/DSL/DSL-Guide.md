# TARS DSL Guide

The TARS Domain Specific Language (DSL) is a powerful way to define and execute AI workflows. This guide explains how to use the TARS DSL to create and run AI workflows.

## Overview

The TARS DSL is a block-based language that allows you to define AI workflows in a simple and intuitive way. Each block represents a specific operation or configuration, and blocks are executed in sequence.

The basic structure of a TARS DSL file is as follows:

```
BLOCK_TYPE {
    property1: value1
    property2: value2
    ...
}
```

## Block Types

The TARS DSL supports the following block types:

### CONFIG

The `CONFIG` block defines global configuration for the workflow.

```
CONFIG {
    model: "llama3"
    temperature: 0.7
    max_tokens: 1000
}
```

Common properties:
- `model`: The AI model to use (e.g., "llama3", "codellama", etc.)
- `temperature`: The temperature parameter for the AI model (0.0 to 1.0)
- `max_tokens`: The maximum number of tokens to generate
- `top_p`: The top-p parameter for the AI model (0.0 to 1.0)
- `top_k`: The top-k parameter for the AI model (integer)

### PROMPT

The `PROMPT` block defines a prompt to send to the AI model.

```
PROMPT {
    text: "What is the capital of France?"
    role: "user"
}
```

Common properties:
- `text`: The prompt text
- `role`: The role of the prompt (e.g., "user", "system", "assistant")
- `model`: Override the model for this prompt

### ACTION

The `ACTION` block defines an action to perform.

```
ACTION {
    type: "generate"
    model: "llama3"
}
```

Common properties:
- `type`: The type of action to perform (e.g., "generate", "chat", "execute")
- `model`: Override the model for this action

### TASK

The `TASK` block defines a task to perform.

```
TASK {
    description: "Find information about the history of Paris"
    agent: "researcher"
}
```

Common properties:
- `description`: The description of the task
- `agent`: The agent to assign the task to

### AGENT

The `AGENT` block defines an agent that can perform tasks.

```
AGENT {
    name: "researcher"
    description: "A research agent that can find information on the web"
    capabilities: ["search", "summarize", "analyze"]
}
```

Common properties:
- `name`: The name of the agent
- `description`: The description of the agent
- `capabilities`: The capabilities of the agent

### AUTO_IMPROVE

The `AUTO_IMPROVE` block defines an auto-improvement task.

```
AUTO_IMPROVE {
    target: "code.cs"
    model: "codellama"
}
```

Common properties:
- `target`: The target to auto-improve
- `model`: Override the model for this auto-improvement task

## Examples

### Basic Example

```
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
}
```

### Chat Example

```
CONFIG {
    model: "llama3"
    temperature: 0.7
    max_tokens: 1000
}

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
}
```

### Agent Example

```
CONFIG {
    model: "llama3"
    temperature: 0.7
    max_tokens: 1000
}

AGENT {
    name: "researcher"
    description: "A research agent that can find information on the web"
    capabilities: ["search", "summarize", "analyze"]
}

TASK {
    description: "Find information about the history of Paris"
    agent: "researcher"
}

ACTION {
    type: "execute"
    task: "Find information about the history of Paris"
}
```

## Using the TARS CLI

The TARS CLI provides commands for working with DSL files:

### Validate a DSL File

```
tarscli dsl validate --file path/to/file.tars
```

### Run a DSL File

```
tarscli dsl run --file path/to/file.tars
```

Add the `--verbose` flag to see detailed output:

```
tarscli dsl run --file path/to/file.tars --verbose
```

### Generate a DSL Template

```
tarscli dsl generate --output path/to/output.tars --template basic
```

Available templates:
- `basic`: A basic template with a CONFIG, PROMPT, and ACTION block
- `chat`: A chat template with a system prompt and a user prompt
- `agent`: An agent template with an AGENT, TASK, and ACTION block
