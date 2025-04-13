# TARS Docker Auto-Coding Scripts

This directory contains scripts for TARS Docker Auto-Coding.

## Overview

TARS Docker Auto-Coding Scripts are PowerShell scripts that enable TARS to auto-code itself using Docker containers. These scripts provide a simple and easy-to-use interface for the Docker auto-coding feature.

## Scripts

- `Run-AutoCoding.ps1`: Finds TODOs in the codebase and improves them.
- `Run-AutoCoding-Simple.ps1`: A simple script that demonstrates auto-coding by improving a test file.
- `Run-AutoCoding-Real.ps1`: A more advanced script that finds TODOs in the codebase and improves them with real implementations.
- `Test-AutoCoding.ps1`: Tests the auto-coding capabilities.
- `Test-AutoCodingDocker.ps1`: Tests the auto-coding capabilities using Docker.

## Running the Scripts

To run a script, simply execute it from the command line:

```powershell
.\Scripts\AutoCoding\Docker\Run-AutoCoding-Simple.ps1
```

## Prerequisites

- Docker Desktop 4.40.0 or later
- TARS CLI built and available
- Docker network created for TARS

## Conclusion

TARS Docker Auto-Coding Scripts provide a simple and easy-to-use interface for the Docker auto-coding feature. By using these scripts, you can enable TARS to analyze, improve, and test its own code without human intervention.
