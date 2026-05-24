# TARS API Reference

This section provides reference documentation for TARS APIs and services.

## Core APIs

- **TarsEngine API** - Core engine APIs for reasoning and analysis
- **Self-Improvement API** - APIs for code analysis and improvement
- **MCP API** - APIs for the Master Control Program
- **Hugging Face Integration API** - APIs for interacting with Hugging Face

## Service Interfaces

### OllamaService

```csharp
public class OllamaService
{
    // Generate text using the Ollama API
    public async Task<string> GenerateCompletion(string prompt, string model);
    
    // Check if a model is available
    public async Task<bool> IsModelAvailable(string model);
    
    // Get a list of available models
    public async Task<List<string>> GetAvailableModels();
}
```

### SelfImprovementService

```csharp
public class SelfImprovementService
{
    // Analyze a file for potential improvements
    public async Task<bool> AnalyzeFile(string filePath, string model);
    
    // Propose improvements for a file
    public async Task<bool> ProposeImprovement(string filePath, string model, bool autoAccept = false);
    
    // Analyze, propose, and apply improvements to a file
    public async Task<bool> RewriteFile(string filePath, string model, bool autoApply = false);
}
```

### HuggingFaceService

```csharp
public class HuggingFaceService
{
    // Search for models on Hugging Face
    public async Task<List<HuggingFaceModel>> SearchModelsAsync(string query, string task = "text-generation", int limit = 10);
    
    // Get the best coding models from Hugging Face
    public async Task<List<HuggingFaceModel>> GetBestCodingModelsAsync(int limit = 10);
    
    // Get detailed information about a model
    public async Task<HuggingFaceModelDetails> GetModelDetailsAsync(string modelId);
    
    // Download a model from Hugging Face
    public async Task<bool> DownloadModelAsync(string modelId);
    
    // Install a model from Hugging Face to Ollama
    public async Task<bool> InstallModelAsync(string modelId, string ollamaModelName = "");
}
```

### LanguageSpecificationService

```csharp
public class LanguageSpecificationService
{
    // Generate EBNF specification for TARS DSL
    public async Task<string> GenerateEbnfAsync();
    
    // Generate BNF specification for TARS DSL
    public async Task<string> GenerateBnfAsync();
    
    // Generate a JSON schema for TARS DSL
    public async Task<string> GenerateJsonSchemaAsync();
    
    // Generate a markdown documentation for TARS DSL
    public async Task<string> GenerateMarkdownDocumentationAsync();
}
```

### DocumentationService

```csharp
public class DocumentationService
{
    // Get all documentation entries
    public List<DocEntry> GetAllDocEntries();
    
    // Get a documentation entry by path
    public DocEntry GetDocEntry(string path);
    
    // Search documentation entries
    public List<DocEntry> SearchDocEntries(string query);
    
    // Format markdown content for console display
    public string FormatMarkdownForConsole(string markdown);
}
```

## CLI Commands

### Self-Improvement Commands

```bash
# Analyze a file for potential improvements
tarscli self-analyze --file path/to/file.cs --model llama3

# Propose improvements for a file
tarscli self-propose --file path/to/file.cs --model llama3 --auto-accept

# Analyze, propose, and apply improvements to a file
tarscli self-rewrite --file path/to/file.cs --model llama3 --auto-apply
```

### Hugging Face Commands

```bash
# Search for models on Hugging Face
tarscli huggingface search --query "code generation" --task text-generation --limit 5

# Get the best coding models from Hugging Face
tarscli huggingface best --limit 3

# Get detailed information about a model
tarscli huggingface details --model microsoft/phi-2

# Download a model from Hugging Face
tarscli huggingface download --model microsoft/phi-2

# Install a model from Hugging Face to Ollama
tarscli huggingface install --model microsoft/phi-2 --name phi2
```

### Language Specification Commands

```bash
# Generate EBNF specification for TARS DSL
tarscli language ebnf --output tars_grammar.ebnf

# Generate BNF specification for TARS DSL
tarscli language bnf --output tars_grammar.bnf

# Generate JSON schema for TARS DSL
tarscli language json-schema --output tars_schema.json

# Generate markdown documentation for TARS DSL
tarscli language docs --output tars_dsl_docs.md
```

### Documentation Commands

```bash
# List all documentation
tarscli docs-explore --list

# Search for documentation
tarscli docs-explore --search "self-improvement"

# View a specific document
tarscli docs-explore --path index.md
```

## Error Handling

All APIs follow a consistent error handling pattern:

1. **Exceptions**: APIs throw exceptions for unexpected errors
2. **Return Values**: APIs return boolean values to indicate success or failure
3. **Logging**: APIs log errors and warnings to the configured logger

## Authentication

TARS APIs do not currently require authentication. However, some external APIs (like Hugging Face) may require API keys, which can be configured in the `appsettings.json` file.

## Rate Limiting

TARS does not currently implement rate limiting for its APIs. However, external APIs (like Hugging Face) may have rate limits, which are respected by the TARS services.

## Future API Plans

Future versions of TARS will include:

1. **REST API**: A REST API for remote access to TARS functionality
2. **WebSocket API**: A WebSocket API for real-time communication
3. **GraphQL API**: A GraphQL API for flexible querying
4. **Authentication**: Authentication and authorization for API access
5. **Rate Limiting**: Rate limiting to prevent abuse
