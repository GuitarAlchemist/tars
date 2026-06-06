# Frequently Asked Questions

This document answers common questions about TARS, its capabilities, and how to use it effectively.

## General Questions

### What is TARS?

TARS (Transformative Autonomous Reasoning System) is an advanced AI-powered system designed to enhance software development through autonomous reasoning, code analysis, and self-improvement. It combines state-of-the-art language models with specialized tools to provide a comprehensive development assistant.

### What programming languages does TARS support?

TARS primarily supports C# and F#, with the core engine components implemented in F# and the CLI and application interfaces implemented in C#. However, the self-improvement system can analyze and improve code in various languages, including:

- C#
- F#
- JavaScript/TypeScript
- Python
- Java
- And more

### Is TARS free to use?

Yes, TARS is an open-source project available under the MIT license. You can use, modify, and distribute it freely, subject to the terms of the license.

### What are the system requirements for TARS?

TARS requires:

- **.NET 9 SDK** or later
- **Git** for version control
- **Ollama** for local language model inference
- **PowerShell** (Windows) or **Bash** (Linux/macOS)
- **Sufficient RAM** for running language models (8GB minimum, 16GB+ recommended)
- **Sufficient disk space** for storing models (10GB+ recommended)

### How does TARS compare to GitHub Copilot or other AI coding assistants?

TARS differs from tools like GitHub Copilot in several ways:

1. **Local Operation**: TARS can run entirely locally, without requiring cloud services
2. **Self-Improvement**: TARS includes a self-improvement system that learns from its interactions
3. **Customizability**: TARS is highly customizable and extensible
4. **Multi-Agent Architecture**: TARS employs a multi-agent architecture for complex tasks
5. **Open Source**: TARS is fully open source, allowing for complete transparency and customization

## Installation and Setup

### How do I install TARS?

To install TARS:

1. Clone the repository: `git clone https://github.com/GuitarAlchemist/tars.git`
2. Navigate to the directory: `cd tars`
3. Run the prerequisites installation script: `.\Scripts\Install-Prerequisites.ps1` (Windows) or `./Scripts/install-prerequisites.sh` (Linux/macOS)
4. Build the project: `dotnet build`

See the [Getting Started](getting-started.md) guide for more detailed instructions.

### How do I update TARS to the latest version?

To update TARS:

1. Navigate to the TARS directory
2. Pull the latest changes: `git pull`
3. Build the project: `dotnet build`

### How do I configure TARS?

TARS can be configured through the `appsettings.json` file in the `TarsCli` directory. This file contains settings for various components, including:

- Ollama configuration
- Hugging Face API key
- Self-improvement settings
- MCP configuration

### How do I create an alias for TARS?

#### Windows (PowerShell)

```powershell
function tarscli { dotnet run --project C:\path\to\tars\TarsCli\TarsCli.csproj -- $args }
Set-Alias -Name tars -Value tarscli
```

Add this to your PowerShell profile to make it persistent.

#### Linux/macOS (Bash)

```bash
alias tarscli='dotnet run --project /path/to/tars/TarsCli/TarsCli.csproj --'
```

Add this to your `.bashrc` or `.zshrc` file to make it persistent.

## Language Models

### What language models does TARS support?

TARS supports a wide range of language models through Ollama, including:

- Llama 3
- CodeLlama
- Phi-2
- Mistral
- And many more

You can also install models from Hugging Face using the Hugging Face integration.

### How do I install a new language model?

To install a model using Ollama:

```bash
ollama pull llama3
```

To install a model from Hugging Face:

```bash
tarscli huggingface install --model microsoft/phi-2 --name phi2
```

### How do I choose the right model for my needs?

When choosing a model, consider:

1. **Size**: Larger models generally provide better results but require more resources
2. **Specialization**: Some models are specialized for specific tasks (e.g., code generation)
3. **Performance**: Different models have different performance characteristics
4. **Resource Requirements**: Consider your available RAM and disk space

For code-related tasks, models like CodeLlama, Phi-2, and StarCoder are good choices.

### Can I use OpenAI models with TARS?

TARS currently focuses on local models through Ollama, but support for OpenAI models is planned for a future release.

## Self-Improvement System

### How does the self-improvement system work?

The self-improvement system:

1. Analyzes code for potential issues using static analysis and language models
2. Generates improvement proposals with explanations
3. Applies improvements automatically or with user approval
4. Learns from feedback to improve future recommendations

See the [Self-Improvement System](features/self-improvement.md) documentation for more details.

### What types of issues can the self-improvement system detect?

The self-improvement system can detect various issues, including:

- Magic numbers (unnamed numeric literals)
- Empty catch blocks
- String concatenation in loops
- Unused variables
- Mutable variables in F# (when immutable alternatives exist)
- Imperative loops in F# (when functional alternatives exist)
- Long methods
- TODO comments

### How do I analyze a file for potential improvements?

To analyze a file:

```bash
tarscli self-analyze --file path/to/file.cs --model llama3
```

### How do I apply improvements automatically?

To apply improvements automatically:

```bash
tarscli self-propose --file path/to/file.cs --model llama3 --auto-accept
```

Or:

```bash
tarscli self-rewrite --file path/to/file.cs --model llama3 --auto-apply
```

### Is it safe to apply improvements automatically?

While the self-improvement system is designed to make safe improvements, it's generally recommended to review proposed changes before applying them, especially for critical code. Use the auto-accept option with caution.

## Master Control Program (MCP)

### What is the Master Control Program (MCP)?

The Master Control Program (MCP) is a component of TARS that enables autonomous operation and integration with external systems. It can execute commands, generate code, and integrate with tools like Augment Code without requiring manual confirmation.

See the [Master Control Program (MCP)](features/mcp.md) documentation for more details.

### How do I generate code using the MCP?

To generate code:

```bash
tarscli mcp code path/to/file.cs "public class MyClass { }"
```

For multi-line code:

```bash
tarscli mcp triple-code path/to/file.cs """
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

### How do I execute terminal commands using the MCP?

To execute terminal commands:

```bash
tarscli mcp execute "echo Hello, World!"
```

### Is it safe to use the MCP for terminal commands?

The MCP executes commands with the same permissions as the user running TARS. Be careful when:

1. Executing commands that modify the system
2. Running commands with elevated privileges
3. Executing commands from untrusted sources

## Hugging Face Integration

### What is the Hugging Face integration?

The Hugging Face integration allows TARS to browse, download, and install models from the Hugging Face Hub. It provides commands for searching, downloading, and installing models for use with TARS.

See the [Hugging Face Integration](features/huggingface.md) documentation for more details.

### How do I search for models on Hugging Face?

To search for models:

```bash
tarscli huggingface search --query "code generation" --task text-generation --limit 5
```

### How do I install a model from Hugging Face?

To install a model:

```bash
tarscli huggingface install --model microsoft/phi-2 --name phi2
```

### Do I need a Hugging Face API key?

An API key is not required for basic operations like searching and downloading public models. However, you'll need an API key to access private models or to use advanced features. You can set your API key in the `appsettings.json` file.

## Troubleshooting

### Ollama is not running

If you see an error about Ollama not running, start the Ollama service:

```bash
ollama serve
```

### Model not found

If a model is not found, you can install it using Ollama:

```bash
ollama pull llama3
```

Or using the Hugging Face integration:

```bash
tarscli huggingface install --model microsoft/phi-2 --name phi2
```

### Out of memory errors

If you encounter out of memory errors when running models:

1. Try using a smaller model
2. Close other memory-intensive applications
3. Increase your system's swap space
4. Consider adding more RAM to your system

### Build errors

If you encounter build errors:

1. Ensure you have the correct .NET SDK version installed: `dotnet --version`
2. Try cleaning the solution: `dotnet clean`
3. Restore packages: `dotnet restore`
4. Rebuild: `dotnet build`

## Contributing

### How can I contribute to TARS?

You can contribute to TARS in several ways:

1. **Code Contributions**: Implement new features or fix bugs
2. **Documentation**: Improve the documentation
3. **Testing**: Test TARS and report issues
4. **Ideas**: Share ideas for new features or improvements

See the [Contributing Guide](contributing.md) for more details.

### How do I report a bug?

To report a bug:

1. Check if the bug has already been reported in the [GitHub issues](https://github.com/GuitarAlchemist/tars/issues)
2. If not, create a new issue with a clear description of the bug, steps to reproduce, and expected vs. actual behavior
3. Include relevant information like your operating system, .NET version, and TARS version

### How do I request a new feature?

To request a new feature:

1. Check if the feature has already been requested in the [GitHub issues](https://github.com/GuitarAlchemist/tars/issues)
2. If not, create a new issue with a clear description of the feature and why it would be valuable
3. Consider implementing the feature yourself and submitting a pull request
