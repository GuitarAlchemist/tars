# TARS Code Generation Scripts

This directory contains scripts for TARS code generation capabilities.

## Overview

These scripts enable TARS to generate code based on various inputs, including explorations and specifications.

## Scripts

- **tars-code-from-exploration.ps1**: Script for generating code from exploration results
- **tars-code-generator.ps1**: General-purpose code generation script
- **tars-generate.ps1**: Script for generating code based on specifications

## Usage

To generate code from an exploration:

```powershell
.\Scripts\CodeGeneration\tars-code-from-exploration.ps1 -Exploration "path/to/exploration.md" -Output "path/to/output.cs"
```

To use the general-purpose code generator:

```powershell
.\Scripts\CodeGeneration\tars-code-generator.ps1 -Prompt "Generate a C# class for a customer entity" -Output "path/to/output.cs"
```

## Related Documentation

- [Code Generation Documentation](../docs/features/code-generation.md)
