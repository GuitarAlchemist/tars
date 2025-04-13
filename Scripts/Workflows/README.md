# TARS Workflow Scripts

This directory contains scripts for TARS workflows.

## Overview

These scripts enable TARS to run complex workflows involving multiple steps and components.

## Scripts

- **tars-workflow.ps1**: Script for running TARS workflows
- **run-collaborative-improvement.ps1**: Script for running collaborative improvement workflows

## Usage

To run a TARS workflow:

```powershell
.\Scripts\Workflows\tars-workflow.ps1 -Workflow "WorkflowName" -Input "InputData"
```

To run a collaborative improvement workflow:

```powershell
.\Scripts\Workflows\run-collaborative-improvement.ps1 -Target "path/to/target.cs"
```

## Related Documentation

- [Workflow Documentation](../docs/features/workflows.md)
