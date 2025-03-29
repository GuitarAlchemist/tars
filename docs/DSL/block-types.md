# TARS DSL Block Types

This document describes the available block types in the TARS Domain Specific Language (DSL) and their purposes.

## Overview

Blocks are the primary organizational units in TARS DSL. Each block has a specific purpose and can contain properties, statements, and nested blocks.

## Available Block Types

### CONFIG

The CONFIG block defines configuration settings for the TARS program.

```
CONFIG {
    version: "1.0";
    author: "John Doe";
    description: "Example TARS program";
}
```

Common properties:
- `version`: The version of the program
- `author`: The author of the program
- `description`: A description of the program
- `license`: The license of the program

### PROMPT

The PROMPT block defines a prompt to be sent to an AI model.

```
PROMPT {
    text: "Generate a list of 5 ideas for improving code quality.";
    model: "gpt-4";
    temperature: 0.7;
}
```

Common properties:
- `text`: The prompt text
- `model`: The AI model to use
- `temperature`: The temperature parameter for the model
- `max_tokens`: The maximum number of tokens to generate
- `stop`: Sequences where the model should stop generating

### ACTION

The ACTION block defines a set of actions to be performed.

```
ACTION {
    let result = processFile("example.cs");
    print(result);
}
```

The ACTION block can contain any valid statements, including:
- Variable declarations
- Function calls
- Control flow statements
- Return statements

### TASK

The TASK block defines a task to be performed, which can include properties and actions.

```
TASK {
    id: "task_001";
    description: "Process a file and print the result";
    
    ACTION {
        let result = processFile("example.cs");
        print(result);
    }
}
```

Common properties:
- `id`: A unique identifier for the task
- `name`: A name for the task
- `description`: A description of the task
- `priority`: The priority of the task
- `dependencies`: Other tasks that this task depends on

### AGENT

The AGENT block defines an AI agent with capabilities, tasks, and communication settings.

```
AGENT {
    id: "agent_001";
    name: "CodeAnalyzer";
    capabilities: ["code-analysis", "refactoring"];
    
    TASK {
        id: "task_001";
        description: "Analyze code quality";
    }
    
    COMMUNICATION {
        protocol: "HTTP";
        endpoint: "http://localhost:8080";
    }
}
```

Common properties:
- `id`: A unique identifier for the agent
- `name`: A name for the agent
- `capabilities`: A list of capabilities the agent has
- `description`: A description of the agent
- `model`: The AI model the agent uses

### AUTO_IMPROVE

The AUTO_IMPROVE block defines settings and actions for self-improvement processes.

```
AUTO_IMPROVE {
    target: "code_quality";
    frequency: "daily";
    
    ACTION {
        let files = findFiles("*.cs");
        foreach(file in files) {
            analyzeAndImprove(file);
        }
    }
}
```

Common properties:
- `target`: The target of the self-improvement process
- `frequency`: How often the self-improvement process should run
- `metrics`: Metrics to track for the self-improvement process
- `threshold`: A threshold for triggering the self-improvement process

### DATA

The DATA block defines data sources and operations.

```
DATA {
    let fileData = FILE("data/sample.csv");
    let apiData = API("https://api.example.com/data");
    let combined = combineData(fileData, apiData);
}
```

The DATA block can contain any valid statements related to data operations, including:
- Loading data from files
- Fetching data from APIs
- Transforming and combining data
- Storing data

### TOOLING

The TOOLING block defines tools and utilities for working with TARS DSL.

```
TOOLING {
    GENERATE_GRAMMAR {
        format: "BNF";
        output: "tars_grammar.bnf";
    }
    
    DIAGNOSTICS {
        level: "detailed";
        output: "diagnostics.log";
    }
}
```

The TOOLING block can contain nested blocks for specific tools:
- `GENERATE_GRAMMAR`: Generate grammar specifications
- `DIAGNOSTICS`: Configure diagnostics
- `INSTRUMENTATION`: Configure instrumentation
- `VISUALIZATION`: Configure visualization tools

## Nested Blocks

Blocks can be nested within other blocks to create hierarchical structures:

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
    
    COMMUNICATION {
        protocol: "HTTP";
        endpoint: "http://localhost:8080";
    }
}
```

## Block Names

Blocks can have optional names, which are useful for referencing them elsewhere:

```
PROMPT CodeQualityPrompt {
    text: "Generate a list of 5 ideas for improving code quality.";
    model: "gpt-4";
}

ACTION {
    let ideas = executePrompt(CodeQualityPrompt);
    print(ideas);
}
```

## Block References

Blocks can reference other blocks by name:

```
TASK AnalyzeCode {
    description: "Analyze code quality";
    
    ACTION {
        let code = readFile("example.cs");
        return analyzeCode(code);
    }
}

TASK RefactorCode {
    description: "Refactor code based on analysis";
    dependencies: [AnalyzeCode];
    
    ACTION {
        let analysis = getTaskResult(AnalyzeCode);
        let code = readFile("example.cs");
        let refactored = refactorCode(code, analysis);
        writeFile("example_refactored.cs", refactored);
    }
}
```

## Best Practices

- Use meaningful names for blocks
- Keep blocks focused on a single responsibility
- Use nested blocks to organize related functionality
- Use properties to configure blocks
- Use actions to define behavior
- Use references to connect blocks
