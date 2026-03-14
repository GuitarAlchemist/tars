# TARS DSL JSON Schema

This document provides a JSON Schema for the TARS Domain Specific Language (DSL). The schema can be used to validate TARS DSL programs represented in JSON format.

## JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "TARS DSL Schema",
  "description": "JSON Schema for TARS DSL",
  "type": "object",
  "properties": {
    "blocks": {
      "type": "array",
      "description": "List of blocks in the TARS program",
      "items": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": ["CONFIG", "PROMPT", "ACTION", "TASK", "AGENT", "AUTO_IMPROVE", "DATA", "TOOLING"],
            "description": "Type of the block"
          },
          "name": {
            "type": "string",
            "description": "Optional name of the block"
          },
          "content": {
            "type": "object",
            "description": "Content of the block",
            "additionalProperties": true
          }
        },
        "required": ["type", "content"]
      }
    }
  },
  "required": ["blocks"]
}
```

## Block-Specific Schemas

### CONFIG Block

```json
{
  "type": "object",
  "properties": {
    "version": {
      "type": "string",
      "description": "Version of the program"
    },
    "author": {
      "type": "string",
      "description": "Author of the program"
    },
    "description": {
      "type": "string",
      "description": "Description of the program"
    },
    "license": {
      "type": "string",
      "description": "License of the program"
    }
  }
}
```

### PROMPT Block

```json
{
  "type": "object",
  "properties": {
    "text": {
      "type": "string",
      "description": "Prompt text"
    },
    "model": {
      "type": "string",
      "description": "AI model to use"
    },
    "temperature": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "description": "Temperature parameter for the model"
    },
    "max_tokens": {
      "type": "integer",
      "minimum": 1,
      "description": "Maximum number of tokens to generate"
    },
    "stop": {
      "oneOf": [
        {
          "type": "string",
          "description": "Sequence where the model should stop generating"
        },
        {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Sequences where the model should stop generating"
        }
      ]
    }
  },
  "required": ["text"]
}
```

### ACTION Block

```json
{
  "type": "object",
  "properties": {
    "statements": {
      "type": "array",
      "description": "List of statements in the action",
      "items": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": ["assignment", "function-call", "if-statement", "for-loop", "while-loop", "return-statement"],
            "description": "Type of the statement"
          },
          "content": {
            "type": "object",
            "description": "Content of the statement",
            "additionalProperties": true
          }
        },
        "required": ["type", "content"]
      }
    }
  }
}
```

### TASK Block

```json
{
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "description": "Unique identifier for the task"
    },
    "name": {
      "type": "string",
      "description": "Name of the task"
    },
    "description": {
      "type": "string",
      "description": "Description of the task"
    },
    "priority": {
      "type": "integer",
      "minimum": 0,
      "description": "Priority of the task"
    },
    "dependencies": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Other tasks that this task depends on"
    },
    "actions": {
      "type": "array",
      "items": {
        "$ref": "#/definitions/action"
      },
      "description": "Actions to perform for this task"
    }
  }
}
```

### AGENT Block

```json
{
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "description": "Unique identifier for the agent"
    },
    "name": {
      "type": "string",
      "description": "Name of the agent"
    },
    "capabilities": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Capabilities of the agent"
    },
    "description": {
      "type": "string",
      "description": "Description of the agent"
    },
    "model": {
      "type": "string",
      "description": "AI model the agent uses"
    },
    "tasks": {
      "type": "array",
      "items": {
        "$ref": "#/definitions/task"
      },
      "description": "Tasks the agent can perform"
    },
    "communication": {
      "type": "object",
      "properties": {
        "protocol": {
          "type": "string",
          "description": "Communication protocol"
        },
        "endpoint": {
          "type": "string",
          "description": "Communication endpoint"
        }
      },
      "description": "Communication settings for the agent"
    }
  }
}
```

### AUTO_IMPROVE Block

```json
{
  "type": "object",
  "properties": {
    "target": {
      "type": "string",
      "description": "Target of the self-improvement process"
    },
    "frequency": {
      "type": "string",
      "description": "How often the self-improvement process should run"
    },
    "metrics": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Metrics to track for the self-improvement process"
    },
    "threshold": {
      "type": "number",
      "description": "Threshold for triggering the self-improvement process"
    },
    "actions": {
      "type": "array",
      "items": {
        "$ref": "#/definitions/action"
      },
      "description": "Actions to perform for self-improvement"
    }
  }
}
```

### DATA Block

```json
{
  "type": "object",
  "properties": {
    "sources": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "Name of the data source"
          },
          "type": {
            "type": "string",
            "enum": ["file", "api", "database"],
            "description": "Type of the data source"
          },
          "location": {
            "type": "string",
            "description": "Location of the data source"
          }
        },
        "required": ["name", "type", "location"]
      },
      "description": "Data sources"
    },
    "operations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": ["load", "transform", "combine", "store"],
            "description": "Type of the data operation"
          },
          "source": {
            "type": "string",
            "description": "Source of the data operation"
          },
          "destination": {
            "type": "string",
            "description": "Destination of the data operation"
          }
        },
        "required": ["type"]
      },
      "description": "Data operations"
    }
  }
}
```

### TOOLING Block

```json
{
  "type": "object",
  "properties": {
    "generate_grammar": {
      "type": "object",
      "properties": {
        "format": {
          "type": "string",
          "enum": ["BNF", "EBNF", "JSON"],
          "description": "Format of the grammar"
        },
        "output": {
          "type": "string",
          "description": "Output file for the grammar"
        }
      },
      "description": "Grammar generation settings"
    },
    "diagnostics": {
      "type": "object",
      "properties": {
        "level": {
          "type": "string",
          "enum": ["error", "warning", "info", "debug", "trace"],
          "description": "Diagnostics level"
        },
        "output": {
          "type": "string",
          "description": "Output file for diagnostics"
        }
      },
      "description": "Diagnostics settings"
    },
    "instrumentation": {
      "type": "object",
      "properties": {
        "enabled": {
          "type": "boolean",
          "description": "Whether instrumentation is enabled"
        },
        "output": {
          "type": "string",
          "description": "Output file for instrumentation"
        }
      },
      "description": "Instrumentation settings"
    }
  }
}
```

## Example JSON Representation

```json
{
  "blocks": [
    {
      "type": "CONFIG",
      "content": {
        "version": "1.0",
        "author": "John Doe",
        "description": "Example TARS program"
      }
    },
    {
      "type": "PROMPT",
      "content": {
        "text": "Generate a list of 5 ideas for improving code quality.",
        "model": "gpt-4",
        "temperature": 0.7
      }
    },
    {
      "type": "ACTION",
      "content": {
        "statements": [
          {
            "type": "assignment",
            "content": {
              "variable": "result",
              "value": {
                "type": "function-call",
                "content": {
                  "function": "processFile",
                  "arguments": ["example.cs"]
                }
              }
            }
          },
          {
            "type": "function-call",
            "content": {
              "function": "print",
              "arguments": [{"type": "variable", "name": "result"}]
            }
          }
        ]
      }
    }
  ]
}
```

## Using the Schema

The JSON Schema can be used to validate TARS DSL programs represented in JSON format. Many programming languages and tools provide support for JSON Schema validation.

### JavaScript Example

```javascript
const Ajv = require('ajv');
const ajv = new Ajv();
const validate = ajv.compile(schema);
const valid = validate(data);
if (!valid) {
  console.log(validate.errors);
}
```

### Python Example

```python
import jsonschema
from jsonschema import validate

try:
    validate(instance=data, schema=schema)
except jsonschema.exceptions.ValidationError as err:
    print(err)
```

### Online Validation

You can also use online tools to validate JSON against a schema:
- [JSON Schema Validator](https://www.jsonschemavalidator.net/)
- [JSON Schema Lint](https://jsonschemalint.com/)

## Notes

- The JSON Schema is a formal definition of the TARS DSL structure
- It can be used to validate TARS DSL programs represented in JSON format
- The schema is subject to change as the language evolves
- Extensions to the language should follow the patterns established in this schema
