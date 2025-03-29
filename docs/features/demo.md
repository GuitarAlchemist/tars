# TARS Demo Command

The TARS Demo Command provides a quick and interactive way to showcase TARS capabilities. It allows users to see TARS in action without having to set up their own files or scenarios.

## Overview

The demo command runs pre-configured demonstrations of various TARS capabilities, including:

1. **Self-Improvement**: Shows how TARS can analyze and improve code
2. **Code Generation**: Demonstrates TARS's ability to generate code from natural language descriptions
3. **Language Specifications**: Shows how TARS can generate formal language specifications for its DSL

## Usage

```bash
tarscli demo [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--type` | `all` | Type of demo to run (`self-improvement`, `code-generation`, `language-specs`, `all`) |
| `--model` | `llama3` | Model to use for the demo |

### Examples

Run all demos with the default model:
```bash
tarscli demo
```

Run only the self-improvement demo with a specific model:
```bash
tarscli demo --type self-improvement --model codellama
```

Run the code generation demo:
```bash
tarscli demo --type code-generation
```

## Demo Types

### Self-Improvement Demo

The self-improvement demo shows how TARS can analyze code, identify issues, and propose improvements. It:

1. Creates a demo file with intentional code issues
2. Analyzes the file to identify the issues
3. Proposes and applies improvements
4. Shows the improved code

Issues demonstrated include:
- Magic numbers
- Inefficient string concatenation
- Empty catch blocks
- Unused variables

### Code Generation Demo

The code generation demo shows how TARS can generate code from natural language descriptions. It:

1. Generates a simple C# class based on a description
2. Generates a more complex implementation (an in-memory cache)
3. Shows the generated code

### Language Specifications Demo

The language specifications demo shows how TARS can generate formal language specifications for its DSL. It:

1. Generates an EBNF specification
2. Generates a BNF specification
3. Generates a JSON schema
4. Generates markdown documentation

## Output

All demo outputs are saved to the `demo` directory in the TARS project root. This includes:

- Generated code files
- Language specifications
- Documentation files

## Use Cases

The demo command is useful for:

1. **New Users**: Quickly see what TARS can do without reading extensive documentation
2. **Presentations**: Demonstrate TARS capabilities to others
3. **Testing**: Verify that TARS is working correctly
4. **Learning**: Understand how TARS analyzes and improves code

## Customization

The demo command uses pre-configured examples, but you can customize the demos by:

1. Modifying the `DemoService.cs` file
2. Adding new demo types
3. Changing the example code or prompts

## Technical Details

The demo command is implemented in the `DemoService` class, which:

1. Creates a demo directory if it doesn't exist
2. Runs the selected demo type(s)
3. Uses the appropriate services (SelfImprovementService, OllamaService, etc.)
4. Saves the outputs to the demo directory

The service is registered in the dependency injection container and accessed through the CLI command.
