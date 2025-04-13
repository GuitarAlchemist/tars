# TARS Scripts

This directory contains scripts for demonstrating and using TARS capabilities.

> **Note**: The scripts in this directory are organized into categories to make them easier to find and use.

## Working Demo Scripts

These scripts demonstrate TARS capabilities using only commands that are known to work correctly:

- **Working-Demo.bat** - A simplified Windows batch file that works reliably
- **Working-Demo.ps1** - A simplified PowerShell script that works reliably

## Script Categories

### [Auto-Coding Scripts](AutoCoding/README.md)

Scripts for TARS Auto-Coding capabilities:

- **Docker Auto-Coding**: Scripts for Docker auto-coding
- **Swarm Auto-Coding**: Scripts for Swarm auto-coding
- **Auto-Coding Demos**: Demo scripts for auto-coding

### [Autonomous Improvement Scripts](AutonomousImprovement/README.md)

Scripts for TARS autonomous improvement capabilities.

### [Retroaction Scripts](Retroaction/README.md)

Scripts for TARS retroaction capabilities.

### [Metascript Scripts](Metascripts/README.md)

Scripts for working with TARS metascripts.

### [Knowledge Management Scripts](Knowledge/README.md)

Scripts for TARS knowledge management capabilities.

### [Code Generation Scripts](CodeGeneration/README.md)

Scripts for TARS code generation capabilities.

### [Demo Scripts](Demos/README.md)

Scripts for demonstrating TARS capabilities.

### [Utility Scripts](Utilities/README.md)

Utility scripts for TARS.

### [Test Scripts](Tests/README.md)

Scripts for testing TARS capabilities.

### [Workflow Scripts](Workflows/README.md)

Scripts for TARS workflows.

### [Prompt Engineering Scripts](PromptEngineering/README.md)

Scripts for TARS prompt engineering.

### Demo-TARS.ps1

A PowerShell Core script that showcases TARS capabilities by running various TARS CLI commands in a structured and visually appealing way.

### Demo-AutoCoding.ps1 and Demo-AutoCoding.bat

These scripts showcase TARS Auto-Coding capabilities. They demonstrate how TARS can auto-code itself using Docker containers and swarm architecture.

**Requirements:**
- PowerShell Core 7.0 or higher (for .ps1 version)
- Windows operating system (for .bat version)
- Docker Desktop 4.40.0 or later
- TARS CLI built and available at `../TarsCli/bin/Debug/net9.0/tarscli.exe`

**Usage:**
```powershell
./Demo-AutoCoding.ps1
```

Or for the batch file version:
```
Demo-AutoCoding.bat
```

**Features:**
- Colorful, formatted output
- Interactive prompts for customizing the demo
- Structured sections demonstrating different auto-coding capabilities
- Pauses between sections for better readability

### Demo-A2A.ps1 and Demo-A2A.bat

These scripts demonstrate the A2A (Agent-to-Agent) protocol capabilities of TARS. They showcase how TARS can communicate with other A2A-compatible agents and expose its capabilities through a standardized interface.

**Requirements:**
- PowerShell Core 7.0 or higher
- TARS CLI built and available at `../TarsCli/bin/Debug/net9.0/tarscli.exe`

**Usage:**
```powershell
./Demo-TARS.ps1
```

**Features:**
- Colorful, formatted output
- Interactive prompts for customizing the demo
- Structured sections demonstrating different TARS capabilities
- Pauses between sections for better readability

### Demo-TARS.bat

A Windows batch file version of the demo script for users who don't have PowerShell Core installed.

**Requirements:**
- Windows operating system
- TARS CLI built and available at `../TarsCli/bin/Debug/net9.0/tarscli.exe`

**Usage:**
```
Demo-TARS.bat
```

**Features:**
- Simple, text-based output
- Prompts for customizing the demo topic
- Structured sections demonstrating different TARS capabilities
- Pauses between sections for better readability

## Demo Sections

### Demo-TARS.ps1 and Demo-TARS.bat Sections

1. **Basic Information** - Version and help information
2. **Deep Thinking** - Generate deep thinking explorations and related topics
3. **Chat Bot** - Interact with the TARS chat bot
4. **Speech System** - Text-to-speech capabilities
5. **Console Capture** - Capture and analyze console output
6. **Model Context Protocol (MCP)** - MCP status and capabilities
7. **Self-Improvement** - Self-improvement status and capabilities
8. **Documentation** - Browse TARS documentation
9. **Language Specification** - TARS DSL specification
10. **Demo Mode** - Run the built-in TARS demo

### Demo-AutoCoding.ps1 Sections

1. **Docker Auto-Coding** - Simple auto-coding using Docker
2. **Swarm Auto-Coding** - Advanced auto-coding using Docker swarm
3. **Auto-Coding with TARS CLI** - Using the TARS CLI for auto-coding

### Demo-A2A.ps1 and Demo-A2A.bat Sections

1. **A2A Server** - Start the A2A server
2. **Agent Card** - Get the TARS agent card
3. **Code Generation Skill** - Send a code generation task
4. **Code Analysis Skill** - Send a code analysis task
5. **Knowledge Extraction Skill** - Send a knowledge extraction task
6. **Self Improvement Skill** - Send a self improvement task
7. **MCP Bridge** - Use A2A through MCP
8. **Stopping the Server** - Stop the A2A server

## Customization

Both scripts allow you to customize the demo topic. When prompted, enter a topic of your choice, or press Enter to use the default topic (Artificial Intelligence).

## Troubleshooting

If you encounter errors running the scripts:

1. Make sure you've built the TARS CLI project first:
   ```
   dotnet build TarsCli/TarsCli.csproj
   ```

2. Check that the TARS CLI executable exists at:
   ```
   TarsCli/bin/Debug/net9.0/tarscli.exe
   ```

3. For PowerShell script execution issues, you may need to set the execution policy:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

4. **Important**: The scripts must be run from the repository root directory, not from the Scripts directory. This is because the TARS CLI expects to find the `appsettings.json` file in the current directory.

   ```
   # Run from repository root
   cd C:\Users\spare\source\repos\tars
   .\Scripts\Demo-TARS.bat

   # Or for PowerShell
   cd C:\Users\spare\source\repos\tars
   .\Scripts\Demo-TARS.ps1
   ```

   The scripts will automatically change to the repository root directory if needed, but it's best to start from there.

5. **Use Working Demo Scripts**: If you encounter issues with the main demo scripts, try using the Working-Demo scripts instead. These scripts use only commands that are known to work correctly and provide a more reliable demonstration experience.
