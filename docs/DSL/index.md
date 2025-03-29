# TARS DSL Reference

The TARS Domain Specific Language (DSL) is a specialized language designed for defining AI workflows, agent behaviors, and self-improvement processes. This reference provides comprehensive documentation for the TARS DSL.

## Introduction

TARS DSL uses a block-based syntax with curly braces, similar to languages like C# and JavaScript. Each block has a type, an optional name, and content that can include properties, statements, and nested blocks.

```
BLOCK_TYPE [name] {
    property1: value1;
    property2: value2;
    
    NESTED_BLOCK {
        nestedProperty: nestedValue;
    }
}
```

## Language Reference

- [**Syntax Guide**](syntax.md) - Basic syntax and structure of TARS DSL
- [**Block Types**](block-types.md) - Available block types and their purposes
- [**Properties**](properties.md) - Property types and syntax
- [**Expressions**](expressions.md) - Expression syntax and evaluation
- [**Statements**](statements.md) - Available statements and their syntax
- [**Functions**](functions.md) - Built-in functions and their usage

## Block Types

- [**CONFIG Block**](blocks/config.md) - Configuration settings for TARS programs
- [**PROMPT Block**](blocks/prompt.md) - Defining prompts for AI models
- [**ACTION Block**](blocks/action.md) - Defining actions to be performed
- [**TASK Block**](blocks/task.md) - Defining tasks with properties and actions
- [**AGENT Block**](blocks/agent.md) - Defining AI agents with capabilities and tasks
- [**AUTO_IMPROVE Block**](blocks/auto-improve.md) - Defining self-improvement processes
- [**DATA Block**](blocks/data.md) - Defining data sources and operations
- [**TOOLING Block**](blocks/tooling.md) - Defining tools and utilities

## Formal Specifications

- [**EBNF Specification**](ebnf.md) - Extended Backus-Naur Form specification
- [**BNF Specification**](bnf.md) - Backus-Naur Form specification
- [**JSON Schema**](json-schema.md) - JSON Schema for TARS DSL

## Examples

- [**Simple Examples**](examples/simple.md) - Basic examples of TARS DSL
- [**Code Analysis**](examples/code-analysis.md) - Examples of code analysis workflows
- [**Multi-Agent Workflows**](examples/multi-agent.md) - Examples of multi-agent workflows
- [**Self-Improvement**](examples/self-improvement.md) - Examples of self-improvement processes

## Tools

- [**CLI Commands**](tools/cli.md) - CLI commands for working with TARS DSL
- [**Language Server**](tools/language-server.md) - Language server for IDE integration
- [**Visualization**](tools/visualization.md) - Tools for visualizing TARS DSL

## Guides

- [**Getting Started**](guides/getting-started.md) - Getting started with TARS DSL
- [**Best Practices**](guides/best-practices.md) - Best practices for writing TARS DSL
- [**Debugging**](guides/debugging.md) - Debugging TARS DSL programs
- [**Advanced Techniques**](guides/advanced.md) - Advanced techniques for TARS DSL

## Resources

- [**Community Examples**](resources/community.md) - Examples from the TARS community
- [**External Tools**](resources/tools.md) - External tools for working with TARS DSL
- [**Related Languages**](resources/related.md) - Languages related to TARS DSL
