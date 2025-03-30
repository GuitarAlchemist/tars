# TARS DSL Documentation

*Generated on: 2025-03-29 20:13:39 UTC*

## Introduction

TARS DSL (Domain Specific Language) is a language designed for defining AI workflows, agent behaviors, and self-improvement processes. It provides a structured way to define prompts, actions, tasks, and agents within the TARS system.

## Syntax Overview

TARS DSL uses a block-based syntax with curly braces. Each block has a type, an optional name, and content. The content can include properties, statements, and nested blocks.

```
BLOCK_TYPE [name] {
    property1: value1;
    property2: value2;
    
    NESTED_BLOCK {
        nestedProperty: nestedValue;
    }
}
```

## Block Types

### CONFIG

The CONFIG block defines configuration settings for the TARS program.

```
CONFIG {
    version: "1.0";
    author: "John Doe";
    description: "Example TARS program";
}
```

### PROMPT

The PROMPT block defines a prompt to be sent to an AI model.

```
PROMPT {
    text: "Generate a list of 5 ideas for improving code quality.";
    model: "gpt-4";
    temperature: 0.7;
}
```

### ACTION

The ACTION block defines a set of actions to be performed.

```
ACTION {
    let result = processFile("example.cs");
    print(result);
}
```

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

### DATA

The DATA block defines data sources and operations.

```
DATA {
    let fileData = FILE("data/sample.csv");
    let apiData = API("https://api.example.com/data");
    let combined = combineData(fileData, apiData);
}
```

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

## Complete Examples

### Example 1: Simple Code Analysis

```
CONFIG {
    version: "1.0";
    description: "Simple code analysis example";
}

PROMPT {
    text: "Analyze the following code for potential improvements:";
    model: "gpt-4";
}

ACTION {
    let code = readFile("example.cs");
    let analysis = analyzeCode(code);
    print(analysis);
}
```

### Example 2: Agent-Based Workflow

```
CONFIG {
    version: "1.0";
    description: "Agent-based workflow example";
}

AGENT CodeAnalyzer {
    capabilities: ["code-analysis"];
    
    TASK AnalyzeCode {
        description: "Analyze code quality";
        
        ACTION {
            let code = readFile("example.cs");
            return analyzeCode(code);
        }
    }
}

AGENT CodeRefactorer {
    capabilities: ["refactoring"];
    
    TASK RefactorCode {
        description: "Refactor code based on analysis";
        
        ACTION {
            let analysis = getTaskResult("AnalyzeCode");
            let code = readFile("example.cs");
            let refactored = refactorCode(code, analysis);
            writeFile("example_refactored.cs", refactored);
        }
    }
}
```
