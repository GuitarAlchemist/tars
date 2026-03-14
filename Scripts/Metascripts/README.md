# TARS Metascript Scripts

This directory contains scripts for working with TARS metascripts.

## Overview

Metascripts are a powerful feature of TARS that allow for the creation of complex workflows and automation.

## Scripts

- **apply-metascript.ps1**: Script for applying a metascript to a specific context
- **register-metascript.ps1**: Script for registering a new metascript with TARS
- **tars-metascript.ps1**: Script for working with TARS metascripts

## Usage

To register a new metascript:

```powershell
.\Scripts\Metascripts\register-metascript.ps1 -Path "path/to/metascript.tars"
```

To apply a metascript:

```powershell
.\Scripts\Metascripts\apply-metascript.ps1 -Name "MetascriptName" -Context "ContextName"
```

## Related Documentation

- [Metascripts Documentation](../docs/features/Metascripts.md)
