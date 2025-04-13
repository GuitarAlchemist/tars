# TARS Prompt Engineering Scripts

This directory contains scripts for TARS prompt engineering.

## Overview

These scripts help create and refine prompts for TARS to generate better results from AI models.

## Scripts

- **tars-advanced-prompt.ps1**: Script for creating advanced prompts
- **tars-prompt-engineering.ps1**: Script for prompt engineering

## Usage

To create an advanced prompt:

```powershell
.\Scripts\PromptEngineering\tars-advanced-prompt.ps1 -Template "TemplateId" -Parameters @{Param1="Value1"; Param2="Value2"}
```

To run prompt engineering:

```powershell
.\Scripts\PromptEngineering\tars-prompt-engineering.ps1 -Prompt "Initial prompt" -Goal "Goal description"
```

## Related Documentation

- [Prompt Engineering Documentation](../docs/features/prompt-engineering.md)
