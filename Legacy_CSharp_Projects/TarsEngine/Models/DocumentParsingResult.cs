namespace TarsEngine.Models;

/// <summary>
/// Represents the result of parsing a document
/// </summary>
public class DocumentParsingResult
{
    /// <summary>
    /// Gets or sets the unique identifier for the parsing result
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the path to the parsed document
    /// </summary>
    public string DocumentPath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the title of the document
    /// </summary>
    public string Title { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the document type
    /// </summary>
    public DocumentType DocumentType { get; set; }

    /// <summary>
    /// Gets or sets the parsed content sections
    /// </summary>
    public List<ContentSection> Sections { get; set; } = new();

    /// <summary>
    /// Gets or sets the metadata extracted from the document
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();

    /// <summary>
    /// Gets or sets the timestamp when the document was parsed
    /// </summary>
    public DateTime ParsedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets any errors that occurred during parsing
    /// </summary>
    public List<string> Errors { get; set; } = new();

    /// <summary>
    /// Gets or sets any warnings that occurred during parsing
    /// </summary>
    public List<string> Warnings { get; set; } = new();

    /// <summary>
    /// Gets whether the parsing was successful
    /// </summary>
    public bool IsSuccessful => Errors.Count == 0;
}

/// <summary>
/// Represents a section of content in a parsed document
/// </summary>
public class ContentSection
{
    /// <summary>
    /// Gets or sets the unique identifier for the content section
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the heading or title of the section
    /// </summary>
    public string Heading { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the content type of the section
    /// </summary>
    public ContentType ContentType { get; set; }

    /// <summary>
    /// Gets or sets the raw text content of the section
    /// </summary>
    public string RawContent { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the processed content of the section
    /// </summary>
    public string ProcessedContent { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the code blocks in the section (if any)
    /// </summary>
    public List<CodeBlock> CodeBlocks { get; set; } = new();

    /// <summary>
    /// Gets or sets the order of the section in the document
    /// </summary>
    public int Order { get; set; }

    /// <summary>
    /// Gets or sets the parent section ID (if this is a subsection)
    /// </summary>
    public string? ParentSectionId { get; set; }

    /// <summary>
    /// Gets or sets the metadata for the section
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}

/// <summary>
/// Represents a code block in a document
/// </summary>
public class CodeBlock
{
    /// <summary>
    /// Gets or sets the unique identifier for the code block
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the programming language of the code block
    /// </summary>
    public string Language { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the code content
    /// </summary>
    public string Code { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets any comments or annotations for the code block
    /// </summary>
    public string Comments { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the line number where the code block starts in the original document
    /// </summary>
    public int StartLine { get; set; }

    /// <summary>
    /// Gets or sets the line number where the code block ends in the original document
    /// </summary>
    public int EndLine { get; set; }

    /// <summary>
    /// Gets or sets whether the code block is executable
    /// </summary>
    public bool IsExecutable { get; set; }

    /// <summary>
    /// Gets or sets the metadata for the code block
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}

/// <summary>
/// Represents the type of a document
/// </summary>
public enum DocumentType
{
    /// <summary>
    /// Unknown document type
    /// </summary>
    Unknown,

    /// <summary>
    /// Markdown document
    /// </summary>
    Markdown,

    /// <summary>
    /// Chat transcript
    /// </summary>
    ChatTranscript,

    /// <summary>
    /// Code file
    /// </summary>
    CodeFile,

    /// <summary>
    /// Reflection document
    /// </summary>
    Reflection,

    /// <summary>
    /// Documentation
    /// </summary>
    Documentation
}

/// <summary>
/// Represents the type of content in a section
/// </summary>
public enum ContentType
{
    /// <summary>
    /// Unknown content type
    /// </summary>
    Unknown,

    /// <summary>
    /// Text content
    /// </summary>
    Text,

    /// <summary>
    /// Code content
    /// </summary>
    Code,

    /// <summary>
    /// Question content
    /// </summary>
    Question,

    /// <summary>
    /// Answer content
    /// </summary>
    Answer,

    /// <summary>
    /// Concept explanation
    /// </summary>
    Concept,

    /// <summary>
    /// Example content
    /// </summary>
    Example,

    /// <summary>
    /// Insight or reflection
    /// </summary>
    Insight
}
