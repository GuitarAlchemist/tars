# TARS Demo Scripts

This directory contains scripts for demonstrating TARS capabilities.

## Working Demo Scripts

These scripts demonstrate TARS capabilities using only commands that are known to work correctly:

- **Working-Demo.bat** - A simplified Windows batch file that works reliably
- **Working-Demo.ps1** - A simplified PowerShell script that works reliably

## Available Scripts

### Demo-TARS.ps1

A PowerShell Core script that showcases TARS capabilities by running various TARS CLI commands in a structured and visually appealing way.

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

Both scripts demonstrate the following TARS capabilities:

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
