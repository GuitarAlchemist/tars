using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Markdig;
using Markdig.Syntax;
using Markdig.Syntax.Inlines;

namespace TarsCli.Services
{
    /// <summary>
    /// Service for exploring and displaying TARS documentation
    /// </summary>
    public class DocumentationService
    {
        private readonly ILogger<DocumentationService> _logger;
        private readonly IConfiguration _configuration;
        private readonly string _docsPath;
        private readonly Dictionary<string, DocEntry> _docEntries = new Dictionary<string, DocEntry>();
        private readonly MarkdownPipeline _pipeline;

        public DocumentationService(
            ILogger<DocumentationService> logger,
            IConfiguration configuration)
        {
            _logger = logger;
            _configuration = configuration;
            
            // Get the docs path from configuration or use default
            var projectRoot = _configuration["Tars:ProjectRoot"] ?? Directory.GetCurrentDirectory();
            _docsPath = Path.Combine(projectRoot, "docs");
            
            // Create Markdig pipeline
            _pipeline = new MarkdownPipelineBuilder()
                .UseAdvancedExtensions()
                .Build();
            
            // Initialize documentation index
            InitializeDocIndex();
        }

        /// <summary>
        /// Initialize the documentation index
        /// </summary>
        private void InitializeDocIndex()
        {
            try
            {
                _logger.LogInformation("Initializing documentation index");
                
                if (!Directory.Exists(_docsPath))
                {
                    _logger.LogWarning($"Documentation directory not found: {_docsPath}");
                    return;
                }
                
                // Get all markdown files
                var markdownFiles = Directory.GetFiles(_docsPath, "*.md", SearchOption.AllDirectories);
                
                foreach (var file in markdownFiles)
                {
                    try
                    {
                        var relativePath = Path.GetRelativePath(_docsPath, file).Replace("\\", "/");
                        var content = File.ReadAllText(file);
                        
                        // Parse the markdown
                        var document = Markdown.Parse(content, _pipeline);
                        
                        // Get the title (first h1)
                        var title = GetTitle(document) ?? Path.GetFileNameWithoutExtension(file);
                        
                        // Get the summary (first paragraph after the title)
                        var summary = GetSummary(document);
                        
                        // Create a doc entry
                        var docEntry = new DocEntry
                        {
                            Path = relativePath,
                            Title = title,
                            Summary = summary,
                            Content = content
                        };
                        
                        _docEntries[relativePath] = docEntry;
                        _logger.LogDebug($"Added doc entry: {relativePath} - {title}");
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, $"Error processing markdown file: {file}");
                    }
                }
                
                _logger.LogInformation($"Documentation index initialized with {_docEntries.Count} entries");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error initializing documentation index");
            }
        }

        /// <summary>
        /// Get the title from a markdown document
        /// </summary>
        private string GetTitle(MarkdownDocument document)
        {
            var heading = document.Descendants<HeadingBlock>().FirstOrDefault(h => h.Level == 1);
            if (heading == null)
            {
                return null;
            }
            
            var inlines = heading.Inline?.Descendants<LiteralInline>().ToList();
            if (inlines == null || inlines.Count == 0)
            {
                return null;
            }
            
            return string.Join("", inlines.Select(i => i.Content.ToString()));
        }

        /// <summary>
        /// Get the summary from a markdown document
        /// </summary>
        private string GetSummary(MarkdownDocument document)
        {
            // Find the first paragraph after the first heading
            var heading = document.Descendants<HeadingBlock>().FirstOrDefault(h => h.Level == 1);
            if (heading == null)
            {
                return null;
            }
            
            var paragraph = document.Descendants<ParagraphBlock>()
                .FirstOrDefault(p => p.Line > heading.Line);
            
            if (paragraph == null)
            {
                return null;
            }
            
            var inlines = paragraph.Inline?.Descendants<LiteralInline>().ToList();
            if (inlines == null || inlines.Count == 0)
            {
                return null;
            }
            
            var summary = string.Join("", inlines.Select(i => i.Content.ToString()));
            
            // Truncate if too long
            if (summary.Length > 200)
            {
                summary = summary.Substring(0, 197) + "...";
            }
            
            return summary;
        }

        /// <summary>
        /// Get all documentation entries
        /// </summary>
        public List<DocEntry> GetAllDocEntries()
        {
            return _docEntries.Values.ToList();
        }

        /// <summary>
        /// Get a documentation entry by path
        /// </summary>
        public DocEntry GetDocEntry(string path)
        {
            if (_docEntries.TryGetValue(path, out var entry))
            {
                return entry;
            }
            
            return null;
        }

        /// <summary>
        /// Search documentation entries
        /// </summary>
        public List<DocEntry> SearchDocEntries(string query)
        {
            if (string.IsNullOrWhiteSpace(query))
            {
                return GetAllDocEntries();
            }
            
            var results = new List<DocEntry>();
            
            foreach (var entry in _docEntries.Values)
            {
                if (entry.Title.Contains(query, StringComparison.OrdinalIgnoreCase) ||
                    entry.Summary?.Contains(query, StringComparison.OrdinalIgnoreCase) == true ||
                    entry.Content.Contains(query, StringComparison.OrdinalIgnoreCase))
                {
                    results.Add(entry);
                }
            }
            
            return results;
        }

        /// <summary>
        /// Get the full path to a documentation file
        /// </summary>
        public string GetFullPath(string path)
        {
            return Path.Combine(_docsPath, path);
        }

        /// <summary>
        /// Format markdown content for console display
        /// </summary>
        public string FormatMarkdownForConsole(string markdown)
        {
            if (string.IsNullOrWhiteSpace(markdown))
            {
                return string.Empty;
            }
            
            var result = new StringBuilder();
            
            // Parse the markdown
            var document = Markdown.Parse(markdown, _pipeline);
            
            // Process each block
            foreach (var block in document)
            {
                if (block is HeadingBlock heading)
                {
                    var level = heading.Level;
                    var prefix = new string('#', level) + " ";
                    var text = GetInlineText(heading.Inline);
                    
                    result.AppendLine();
                    result.AppendLine(prefix + text);
                    result.AppendLine();
                }
                else if (block is ParagraphBlock paragraph)
                {
                    var text = GetInlineText(paragraph.Inline);
                    result.AppendLine(text);
                    result.AppendLine();
                }
                else if (block is ListBlock list)
                {
                    foreach (var item in list)
                    {
                        if (item is ListItemBlock listItem)
                        {
                            foreach (var itemBlock in listItem)
                            {
                                if (itemBlock is ParagraphBlock itemParagraph)
                                {
                                    var text = GetInlineText(itemParagraph.Inline);
                                    result.AppendLine("- " + text);
                                }
                            }
                        }
                    }
                    result.AppendLine();
                }
                else if (block is ThematicBreakBlock)
                {
                    result.AppendLine("-------------------");
                    result.AppendLine();
                }
                else if (block is CodeBlock codeBlock)
                {
                    result.AppendLine("```");
                    result.AppendLine(codeBlock.Lines.ToString());
                    result.AppendLine("```");
                    result.AppendLine();
                }
            }
            
            return result.ToString();
        }

        /// <summary>
        /// Get text from inline elements
        /// </summary>
        private string GetInlineText(ContainerInline inline)
        {
            if (inline == null)
            {
                return string.Empty;
            }
            
            var result = new StringBuilder();
            
            foreach (var item in inline)
            {
                if (item is LiteralInline literal)
                {
                    result.Append(literal.Content);
                }
                else if (item is EmphasisInline emphasis)
                {
                    var text = GetInlineText(emphasis);
                    if (emphasis.DelimiterCount == 2)
                    {
                        result.Append("**" + text + "**");
                    }
                    else
                    {
                        result.Append("*" + text + "*");
                    }
                }
                else if (item is LinkInline link)
                {
                    var text = GetInlineText(link);
                    result.Append(text);
                    
                    if (!string.IsNullOrWhiteSpace(link.Url))
                    {
                        result.Append(" [" + link.Url + "]");
                    }
                }
                else if (item is ContainerInline container)
                {
                    result.Append(GetInlineText(container));
                }
            }
            
            return result.ToString();
        }
    }

    /// <summary>
    /// Documentation entry
    /// </summary>
    public class DocEntry
    {
        /// <summary>
        /// Path to the documentation file
        /// </summary>
        public string Path { get; set; }
        
        /// <summary>
        /// Title of the documentation
        /// </summary>
        public string Title { get; set; }
        
        /// <summary>
        /// Summary of the documentation
        /// </summary>
        public string Summary { get; set; }
        
        /// <summary>
        /// Full content of the documentation
        /// </summary>
        public string Content { get; set; }
    }
}
