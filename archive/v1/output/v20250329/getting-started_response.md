Here is the improved documentation:

```markdown
# Getting Started with TARS

This guide helps you install and start using TARS' core features quickly.

## Prerequisites

Before installing TARS, ensure you have:
- **.NET 9 SDK** or later
- **Git** for version control
- **Ollama** for local language model inference
- **PowerShell** (Windows) or **Bash** (Linux/macOS)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/GuitarAlchemist/tars.git
cd tars
```

### Step 2: Install Prerequisites

Run the prerequisites installation script:

#### Windows

```powershell
.\Scripts\Install-Prerequisites.ps1
```

#### Linux/macOS

```bash
./Scripts/install-prerequisites.sh
```

This script sets up Ollama, downloads required language models, and installs dependencies.

### Step 3: Build the Project

```bash
dotnet build
```

### Step 4: Run TARS

```bash
dotnet run --project TarsCli/TarsCli.csproj -- help
```

This displays available commands and options.

## Quick Start Guide

### Analyzing Code

To analyze a file for potential improvements:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-analyze --file path/to/file.cs --model llama3
```

This analyzes the file, displaying potential issues and improvement opportunities.

### Proposing Improvements

To generate improvement proposals for a file:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-propose --file path/to/file.cs --model llama3
```

This analyzes the file, generates improvement proposals, and asks if you want to apply the changes.

### Automatic Rewriting

To automatically analyze, propose, and apply improvements:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-rewrite --file path/to/file.cs --model llama3 --auto-apply
```

This analyzes the file, generates improvement proposals, and applies the changes.

### Using the Master Control Program (MCP)

To generate code using the MCP:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- mcp code path/to/file.cs "public class MyClass { }"
```

For multi-line code blocks, use triple-quoted syntax:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- mcp triple-code path/to/file.cs """
using System;

public class Program
{
    public static void Main()
    {
        Console.WriteLine("Hello, World!");
    }
}
"""
```

To execute terminal commands:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- mcp execute "echo Hello, World!"
```

### Working with Hugging Face Models

To find the best coding models on Hugging Face:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- huggingface search --model-type coding
```

To install a model:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- huggingface install --model microsoft/phi-2
```

### Troubleshooting

If you encounter issues, refer to the [FAQ](faq.md) or open an issue on the [GitHub repository](https://github.com/GuitarAlchemist/tars/issues).

Changes:

* Added a brief introduction to TARS and its features
* Improved formatting for easier reading
* Simplified language and reduced repetition
* Moved troubleshooting section to the end, as it's less essential information
* Corrected code blocks for consistency and readability