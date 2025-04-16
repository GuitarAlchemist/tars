using System.Text.Json;
using Microsoft.Extensions.Logging;
using TarsEngine.Services;

namespace TarsCli.Services.Agents;

/// <summary>
/// Agent for generating and updating documentation
/// </summary>
public class DocumentationGeneratorAgent
{
    private readonly ILogger<DocumentationGeneratorAgent> _logger;
    private readonly LlmService _llmService;
    private readonly FileService _fileService;

    /// <summary>
    /// Initializes a new instance of the DocumentationGeneratorAgent class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="llmService">LLM service</param>
    /// <param name="fileService">File service</param>
    public DocumentationGeneratorAgent(
        ILogger<DocumentationGeneratorAgent> logger,
        LlmService llmService,
        FileService fileService)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _llmService = llmService ?? throw new ArgumentNullException(nameof(llmService));
        _fileService = fileService ?? throw new ArgumentNullException(nameof(fileService));
    }

    /// <summary>
    /// Handles an MCP request
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    public async Task<JsonElement> HandleRequestAsync(JsonElement request)
    {
        try
        {
            // Extract operation from the request
            var operation = "generate";
            if (request.TryGetProperty("operation", out var operationElement))
            {
                operation = operationElement.GetString() ?? "generate";
            }

            // Handle the operation
            return operation switch
            {
                "generate" => await GenerateDocumentationAsync(request),
                "update" => await UpdateDocumentationAsync(request),
                "extract" => await ExtractDocumentationAsync(request),
                "summarize" => await SummarizeDocumentationAsync(request),
                _ => CreateErrorResponse($"Unknown operation: {operation}")
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error handling request");
            return CreateErrorResponse(ex.Message);
        }
    }

    /// <summary>
    /// Generates documentation for a file
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    private async Task<JsonElement> GenerateDocumentationAsync(JsonElement request)
    {
        // Extract file path and content from the request
        if (!request.TryGetProperty("file_path", out var filePathElement) ||
            !request.TryGetProperty("file_content", out var fileContentElement))
        {
            return CreateErrorResponse("Missing required parameters: file_path, file_content");
        }

        var filePath = filePathElement.GetString();
        var fileContent = fileContentElement.GetString();

        if (string.IsNullOrEmpty(filePath) || string.IsNullOrEmpty(fileContent))
        {
            return CreateErrorResponse("Invalid parameters: file_path, file_content");
        }

        // Extract model from the request (optional)
        var model = "llama3";
        if (request.TryGetProperty("model", out var modelElement))
        {
            model = modelElement.GetString() ?? "llama3";
        }

        // Determine the language from the file extension
        var fileExtension = Path.GetExtension(filePath).TrimStart('.');
        var language = fileExtension.ToLower() switch
        {
            "cs" => "csharp",
            "fs" => "fsharp",
            "md" => "markdown",
            _ => "unknown"
        };

        // Generate documentation
        var documentation = await GenerateDocumentationForFileAsync(filePath, fileContent, language, model);

        // Create the documentation file path
        var fileName = Path.GetFileNameWithoutExtension(filePath);
        var directory = Path.GetDirectoryName(filePath);
        var docFileName = $"{fileName}.md";
        var docFilePath = Path.Combine(directory, "docs", docFileName);

        // Create the response
        var responseObj = new
        {
            success = true,
            file_path = filePath,
            language,
            documentation_file_path = docFilePath,
            documentation_content = documentation
        };

        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
    }

    /// <summary>
    /// Updates documentation for a file
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    private async Task<JsonElement> UpdateDocumentationAsync(JsonElement request)
    {
        // Extract file path, content, and documentation path from the request
        if (!request.TryGetProperty("file_path", out var filePathElement) ||
            !request.TryGetProperty("file_content", out var fileContentElement) ||
            !request.TryGetProperty("documentation_path", out var docPathElement))
        {
            return CreateErrorResponse("Missing required parameters: file_path, file_content, documentation_path");
        }

        var filePath = filePathElement.GetString();
        var fileContent = fileContentElement.GetString();
        var docPath = docPathElement.GetString();

        if (string.IsNullOrEmpty(filePath) || string.IsNullOrEmpty(fileContent) || string.IsNullOrEmpty(docPath))
        {
            return CreateErrorResponse("Invalid parameters: file_path, file_content, documentation_path");
        }

        // Extract model from the request (optional)
        var model = "llama3";
        if (request.TryGetProperty("model", out var modelElement))
        {
            model = modelElement.GetString() ?? "llama3";
        }

        // Read the existing documentation
        var existingDocumentation = "";
        if (File.Exists(docPath))
        {
            existingDocumentation = await _fileService.ReadFileAsync(docPath);
        }

        // Determine the language from the file extension
        var fileExtension = Path.GetExtension(filePath).TrimStart('.');
        var language = fileExtension.ToLower() switch
        {
            "cs" => "csharp",
            "fs" => "fsharp",
            "md" => "markdown",
            _ => "unknown"
        };

        // Update the documentation
        var updatedDocumentation = await UpdateDocumentationForFileAsync(filePath, fileContent, existingDocumentation, language, model);

        // Create the response
        var responseObj = new
        {
            success = true,
            file_path = filePath,
            language,
            documentation_path = docPath,
            original_documentation = existingDocumentation,
            updated_documentation = updatedDocumentation,
            has_updates = existingDocumentation != updatedDocumentation
        };

        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
    }

    /// <summary>
    /// Extracts documentation from a file
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    private async Task<JsonElement> ExtractDocumentationAsync(JsonElement request)
    {
        // Extract file path and content from the request
        if (!request.TryGetProperty("file_path", out var filePathElement) ||
            !request.TryGetProperty("file_content", out var fileContentElement))
        {
            return CreateErrorResponse("Missing required parameters: file_path, file_content");
        }

        var filePath = filePathElement.GetString();
        var fileContent = fileContentElement.GetString();

        if (string.IsNullOrEmpty(filePath) || string.IsNullOrEmpty(fileContent))
        {
            return CreateErrorResponse("Invalid parameters: file_path, file_content");
        }

        // Determine the language from the file extension
        var fileExtension = Path.GetExtension(filePath).TrimStart('.');
        var language = fileExtension.ToLower() switch
        {
            "cs" => "csharp",
            "fs" => "fsharp",
            "md" => "markdown",
            _ => "unknown"
        };

        // Extract documentation
        var extractedDocs = await ExtractDocumentationFromFileAsync(filePath, fileContent, language);

        // Create the response
        var responseObj = new
        {
            success = true,
            file_path = filePath,
            language,
            extracted_documentation = extractedDocs,
            has_documentation = !string.IsNullOrEmpty(extractedDocs)
        };

        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
    }

    /// <summary>
    /// Summarizes documentation
    /// </summary>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    private async Task<JsonElement> SummarizeDocumentationAsync(JsonElement request)
    {
        // Extract documentation content from the request
        if (!request.TryGetProperty("documentation_content", out var docContentElement))
        {
            return CreateErrorResponse("Missing required parameter: documentation_content");
        }

        var docContent = docContentElement.GetString();

        if (string.IsNullOrEmpty(docContent))
        {
            return CreateErrorResponse("Invalid parameter: documentation_content");
        }

        // Extract model from the request (optional)
        var model = "llama3";
        if (request.TryGetProperty("model", out var modelElement))
        {
            model = modelElement.GetString() ?? "llama3";
        }

        // Summarize the documentation
        var summary = await SummarizeDocumentationContentAsync(docContent, model);

        // Create the response
        var responseObj = new
        {
            success = true,
            documentation_content = docContent,
            summary,
            summary_length = summary.Length
        };

        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
    }

    /// <summary>
    /// Generates documentation for a file
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="fileContent">The file content</param>
    /// <param name="language">The language</param>
    /// <param name="model">The model</param>
    /// <returns>The documentation</returns>
    private async Task<string> GenerateDocumentationForFileAsync(string filePath, string fileContent, string language, string model)
    {
        // Create a prompt for the LLM
        var prompt = $@"Generate comprehensive Markdown documentation for the following {language} file.

Include:
1. A title based on the file name
2. A brief description of the file's purpose
3. An overview of the main components/classes/functions
4. Detailed documentation for each component
5. Usage examples where appropriate
6. Any dependencies or requirements

File path: {filePath}

File content:
```{language}
{fileContent}
```

Generate the documentation in Markdown format.";

        // Generate the documentation
        return await _llmService.GetCompletionAsync(prompt, temperature: 0.2, maxTokens: 2000, model: model);
    }

    /// <summary>
    /// Updates documentation for a file
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="fileContent">The file content</param>
    /// <param name="existingDocumentation">The existing documentation</param>
    /// <param name="language">The language</param>
    /// <param name="model">The model</param>
    /// <returns>The updated documentation</returns>
    private async Task<string> UpdateDocumentationForFileAsync(string filePath, string fileContent, string existingDocumentation, string language, string model)
    {
        // Create a prompt for the LLM
        var prompt = $@"Update the existing Markdown documentation for the following {language} file.

File path: {filePath}

File content:
```{language}
{fileContent}
```

Existing documentation:
```markdown
{existingDocumentation}
```

Update the documentation to reflect the current state of the file. Keep the existing structure and formatting, but update the content to match the current file. Add documentation for any new components and remove documentation for components that no longer exist.

Generate the updated documentation in Markdown format.";

        // Generate the updated documentation
        return await _llmService.GetCompletionAsync(prompt, temperature: 0.2, maxTokens: 2000, model: model);
    }

    /// <summary>
    /// Extracts documentation from a file
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="fileContent">The file content</param>
    /// <param name="language">The language</param>
    /// <returns>The extracted documentation</returns>
    private async Task<string> ExtractDocumentationFromFileAsync(string filePath, string fileContent, string language)
    {
        // For C# and F#, extract XML documentation comments
        if (language == "csharp" || language == "fsharp")
        {
            var lines = fileContent.Split('\n');
            var docLines = new List<string>();
            var inDocComment = false;

            foreach (var line in lines)
            {
                var trimmedLine = line.Trim();

                if (trimmedLine.StartsWith("/// "))
                {
                    docLines.Add(trimmedLine.Substring(4));
                }
                else if (trimmedLine.StartsWith("/**") || trimmedLine.StartsWith("/*"))
                {
                    inDocComment = true;
                    docLines.Add(trimmedLine.Substring(3).Trim());
                }
                else if (inDocComment && trimmedLine.EndsWith("*/"))
                {
                    inDocComment = false;
                    docLines.Add(trimmedLine.Substring(0, trimmedLine.Length - 2).Trim());
                }
                else if (inDocComment)
                {
                    docLines.Add(trimmedLine.TrimStart('*').Trim());
                }
            }

            return string.Join("\n", docLines);
        }
        // For Markdown, return the content as is
        else if (language == "markdown")
        {
            return fileContent;
        }
        // For other languages, return an empty string
        else
        {
            return string.Empty;
        }
    }

    /// <summary>
    /// Summarizes documentation content
    /// </summary>
    /// <param name="docContent">The documentation content</param>
    /// <param name="model">The model</param>
    /// <returns>The summary</returns>
    private async Task<string> SummarizeDocumentationContentAsync(string docContent, string model)
    {
        // Create a prompt for the LLM
        var prompt = $@"Summarize the following documentation:

```markdown
{docContent}
```

Provide a concise summary that captures the key points of the documentation. The summary should be no more than 3 paragraphs.";

        // Generate the summary
        return await _llmService.GetCompletionAsync(prompt, temperature: 0.3, maxTokens: 500, model: model);
    }

    /// <summary>
    /// Creates an error response
    /// </summary>
    /// <param name="message">The error message</param>
    /// <returns>The error response</returns>
    private JsonElement CreateErrorResponse(string message)
    {
        var responseObj = new
        {
            success = false,
            error = message
        };

        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(responseObj));
    }
}
