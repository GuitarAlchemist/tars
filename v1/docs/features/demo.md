# TARS Demo Command

The TARS Demo Command provides a quick and interactive way to showcase TARS capabilities. It allows users to see TARS in action without having to set up their own files or scenarios.

## Overview

The demo command runs pre-configured demonstrations of various TARS capabilities, including:

1. **Self-Improvement**: Shows how TARS can analyze and improve code
2. **Code Generation**: Demonstrates TARS's ability to generate code from natural language descriptions
3. **Language Specifications**: Shows how TARS can generate formal language specifications for its DSL
4. **ChatBot**: Demonstrates TARS's conversational capabilities
5. **Deep Thinking**: Shows how TARS can generate in-depth explorations on complex topics
6. **Learning Plan**: Demonstrates generation of personalized learning plans
7. **Course Generator**: Shows how TARS can create structured course content
8. **Tutorial Organizer**: Demonstrates management and categorization of tutorials
9. **Speech**: Shows TARS's text-to-speech capabilities
10. **MCP**: Demonstrates Model Context Protocol integration

## Usage

```bash
tarscli demo [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--type` | `all` | Type of demo to run (`self-improvement`, `code-generation`, `language-specs`, `chatbot`, `deep-thinking`, `learning-plan`, `course-generator`, `tutorial-organizer`, `speech`, `mcp`, `all`) |
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

Run the chatbot demo:
```bash
tarscli demo --type chatbot
```

Run the learning plan demo:
```bash
tarscli demo --type learning-plan
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

### ChatBot Demo

The chatbot demo shows how TARS can engage in natural language conversations. It:

1. Simulates a conversation with predefined questions
2. Shows TARS's responses to various queries
3. Saves the conversation transcript

### Deep Thinking Demo

The deep thinking demo shows how TARS can generate in-depth explorations on complex topics. It:

1. Generates a deep thinking exploration on a specified topic
2. Displays a preview of the generated content
3. Generates related topics for further exploration

### Learning Plan Demo

The learning plan demo shows how TARS can generate personalized learning plans. It:

1. Creates a learning plan with specified parameters (name, topic, skill level, goals, etc.)
2. Displays the introduction and modules of the learning plan
3. Saves the complete learning plan as a JSON file

### Course Generator Demo

The course generator demo shows how TARS can create structured course content. It:

1. Generates a course with specified parameters (title, description, topic, etc.)
2. Displays the overview and lessons of the course
3. Saves the complete course as a JSON file

### Tutorial Organizer Demo

The tutorial organizer demo shows how TARS can manage and categorize tutorials. It:

1. Creates multiple tutorials with different categories and difficulty levels
2. Lists all tutorials with their metadata
3. Saves the tutorial content as markdown files

### Speech Demo

The speech demo shows TARS's text-to-speech capabilities. It:

1. Converts a text message to speech
2. Lists available voices
3. Saves information about the demo to a text file

### MCP Demo

The MCP (Model Context Protocol) demo shows how TARS integrates with the Model Context Protocol. It:

1. Executes a command using MCP
2. Generates code using MCP
3. Displays the results of these operations

## Output

All demo outputs are saved to the `demo` directory in the TARS project root. This includes:

- Generated code files
- Language specifications
- Documentation files
- Learning plans and courses
- Tutorial content
- Chat transcripts
- Deep thinking explorations

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
