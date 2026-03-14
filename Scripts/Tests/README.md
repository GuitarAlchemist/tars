# TARS Test Scripts

This directory contains scripts for testing TARS capabilities.

## Overview

These scripts test various TARS features and capabilities to ensure they are working correctly.

## Scripts

- **test_fsharp_analysis.ps1**: Script for testing F# code analysis
- **test-mcp-connection.ps1**: Script for testing MCP connection

## Usage

To test F# code analysis:

```powershell
.\Scripts\Tests\test_fsharp_analysis.ps1 -Path "path/to/file.fs"
```

To test MCP connection:

```powershell
.\Scripts\Tests\test-mcp-connection.ps1 -Url "http://localhost:8999/"
```

## Related Documentation

- [Testing Documentation](../docs/testing.md)
