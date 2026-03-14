# TARS Knowledge Management Scripts

This directory contains scripts for TARS knowledge management capabilities.

## Overview

These scripts enable TARS to extract, process, and apply knowledge from various sources.

## Scripts

- **test-knowledge-application.ps1**: Script for testing knowledge application
- **test-knowledge-cycle.ps1**: Script for testing the full knowledge cycle
- **test-knowledge-extraction.ps1**: Script for testing knowledge extraction
- **tars-learning.ps1**: Script for TARS learning processes

## Usage

To test knowledge extraction:

```powershell
.\Scripts\Knowledge\test-knowledge-extraction.ps1 -Source "path/to/source"
```

To test knowledge application:

```powershell
.\Scripts\Knowledge\test-knowledge-application.ps1 -Knowledge "KnowledgeId" -Target "TargetId"
```

## Related Documentation

- [Knowledge Extraction Documentation](../docs/features/knowledge-extraction.md)
