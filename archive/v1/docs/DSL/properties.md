# TARS DSL Properties

This document describes the property types and syntax in the TARS Domain Specific Language (DSL).

## Property Syntax

Properties in TARS DSL are defined as key-value pairs:

```
property: value;
```

The semicolon at the end is optional but recommended for clarity.

## Property Types

TARS DSL supports various property types:

### String Properties

String properties are enclosed in double quotes:

```
name: "John Doe";
description: "This is a description";
```

Strings can span multiple lines:

```
text: "This is a
multi-line
string";
```

### Number Properties

Number properties can be integers or floating-point numbers:

```
age: 30;
temperature: 0.7;
maxTokens: 1024;
```

### Boolean Properties

Boolean properties can be `true` or `false`:

```
enabled: true;
verbose: false;
```

### Array Properties

Array properties are enclosed in square brackets:

```
tags: ["tag1", "tag2", "tag3"];
numbers: [1, 2, 3, 4, 5];
mixed: ["string", 42, true];
```

Arrays can span multiple lines:

```
capabilities: [
    "code-analysis",
    "refactoring",
    "documentation"
];
```

### Object Properties

Object properties are enclosed in curly braces:

```
metadata: {
    author: "John Doe",
    version: "1.0",
    created: "2025-03-29"
};
```

Objects can be nested:

```
config: {
    server: {
        host: "localhost",
        port: 8080
    },
    database: {
        url: "mongodb://localhost:27017",
        name: "tars"
    }
};
```

### Identifier Properties

Identifier properties reference other blocks or variables:

```
dependsOn: TaskA;
model: DefaultModel;
```

## Property Inheritance

Properties can be inherited from parent blocks:

```
AGENT {
    model: "gpt-4";
    
    TASK {
        // Inherits model: "gpt-4" from parent
    }
}
```

## Property Overriding

Properties can be overridden in child blocks:

```
AGENT {
    model: "gpt-4";
    
    TASK {
        model: "codellama"; // Overrides the parent's model
    }
}
```

## Property Access

Properties can be accessed in expressions using dot notation:

```
AGENT {
    name: "CodeAnalyzer";
    
    ACTION {
        print("Agent name: " + this.name);
    }
}
```

## Common Properties

### Common to All Blocks

- `id`: A unique identifier for the block
- `name`: A name for the block
- `description`: A description of the block
- `tags`: Tags for categorizing the block

### CONFIG Block

- `version`: The version of the program
- `author`: The author of the program
- `license`: The license of the program
- `created`: The creation date of the program
- `updated`: The last update date of the program

### PROMPT Block

- `text`: The prompt text
- `model`: The AI model to use
- `temperature`: The temperature parameter for the model
- `max_tokens`: The maximum number of tokens to generate
- `stop`: Sequences where the model should stop generating
- `top_p`: The top-p parameter for the model
- `frequency_penalty`: The frequency penalty parameter for the model
- `presence_penalty`: The presence penalty parameter for the model

### TASK Block

- `priority`: The priority of the task
- `dependencies`: Other tasks that this task depends on
- `timeout`: The maximum time the task can run
- `retries`: The number of times to retry the task if it fails
- `retry_delay`: The delay between retries

### AGENT Block

- `capabilities`: A list of capabilities the agent has
- `model`: The AI model the agent uses
- `memory`: The memory configuration for the agent
- `learning_rate`: The learning rate for the agent
- `exploration_rate`: The exploration rate for the agent

### AUTO_IMPROVE Block

- `target`: The target of the self-improvement process
- `frequency`: How often the self-improvement process should run
- `metrics`: Metrics to track for the self-improvement process
- `threshold`: A threshold for triggering the self-improvement process
- `max_iterations`: The maximum number of iterations for the self-improvement process

## Best Practices

- Use meaningful names for properties
- Use consistent naming conventions
- Use appropriate property types
- Document properties with comments
- Group related properties together
- Use inheritance to avoid duplication
- Override properties only when necessary
