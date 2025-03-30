# Getting Started with TARS

This guide will help you install TARS and start using its core features. Follow these steps to get up and running quickly.

## Prerequisites

Before installing TARS, ensure you have the following prerequisites:

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

This script will:
- Check for and install required dependencies
- Set up Ollama
- Download required language models

### Step 3: Build the Project

```bash
dotnet build
```

### Step 4: Run TARS

```bash
dotnet run --project TarsCli/TarsCli.csproj -- help
```

This will display the available commands and options.

## Quick Start Guide

### Analyzing Code

To analyze a file for potential improvements:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-analyze --file path/to/file.cs --model llama3
```

This will analyze the file and display potential issues and improvement opportunities.

### Proposing Improvements

To generate improvement proposals for a file:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-propose --file path/to/file.cs --model llama3
```

This will analyze the file, generate improvement proposals, and ask if you want to apply the changes.

### Automatic Rewriting

To automatically analyze, propose, and apply improvements:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- self-rewrite --file path/to/file.cs --model llama3 --auto-apply
```

This will analyze the file, generate improvement proposals, and automatically apply the changes.

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
dotnet run --project TarsCli/TarsCli.csproj -- huggingface best --limit 3
```

To get details about a specific model:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- huggingface details --model microsoft/phi-2
```

To install a model for use with TARS:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- huggingface install --model microsoft/phi-2 --name phi2
```

### Generating Language Specifications

To generate EBNF specification for TARS DSL:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- language ebnf --output tars_grammar.ebnf
```

To generate markdown documentation for TARS DSL:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- language docs --output tars_dsl_docs.md
```

## Creating an Alias

For convenience, you can create an alias for the TARS CLI:

### Windows (PowerShell)

```powershell
function tarscli { dotnet run --project C:\path\to\tars\TarsCli\TarsCli.csproj -- $args }
Set-Alias -Name tars -Value tarscli
```

Add this to your PowerShell profile to make it persistent.

### Linux/macOS (Bash)

```bash
alias tarscli='dotnet run --project /path/to/tars/TarsCli/TarsCli.csproj --'
```

Add this to your `.bashrc` or `.zshrc` file to make it persistent.

## Next Steps

Now that you have TARS installed and running, here are some next steps to explore:

1. **Read the Documentation**: Explore the [documentation](index.md) to learn more about TARS capabilities
2. **Try the Examples**: Check out the [examples](examples/index.md) to see TARS in action
3. **Join the Community**: Connect with other TARS users and contributors
4. **Contribute**: Consider [contributing](contributing.md) to the TARS project

## Troubleshooting

### Common Issues

#### Ollama Not Running

If you see an error about Ollama not running, start the Ollama service:

```bash
ollama serve
```

#### Model Not Found

If a model is not found, you can install it using Ollama:

```bash
ollama pull llama3
```

Or using the Hugging Face integration:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- huggingface install --model microsoft/phi-2 --name phi2
```

#### Build Errors

If you encounter build errors, ensure you have the correct .NET SDK version installed:

```bash
dotnet --version
```

If the version is incorrect, download and install the [.NET 9 SDK](https://dotnet.microsoft.com/download/dotnet/9.0).

### Getting Help

If you encounter issues not covered here, you can:

1. Check the [FAQ](faq.md) for common questions and answers
2. Open an issue on the [GitHub repository](https://github.com/GuitarAlchemist/tars/issues)
3. Reach out to the community for help
