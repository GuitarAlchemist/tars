using System.Text;
using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.Extensions.Logging;

namespace DuplicationDemo;

/// <summary>
/// Analyzer for C# code duplication
/// </summary>
public class DuplicationAnalyzer
{
    private readonly ILogger<DuplicationAnalyzer> _logger;
    
    // Minimum sequence length to consider as duplication (in tokens)
    private const int MinimumDuplicateTokens = 100;
    
    // Minimum sequence length to consider as duplication (in lines)
    private const int MinimumDuplicateLines = 5;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="DuplicationAnalyzer"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    public DuplicationAnalyzer(ILogger<DuplicationAnalyzer> logger)
    {
        _logger = logger;
    }
    
    /// <summary>
    /// Analyzes duplication in a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <returns>Duplication analysis result</returns>
    public async Task<DuplicationResult> AnalyzeFileAsync(string filePath)
    {
        try
        {
            _logger.LogInformation("Analyzing duplication in file {FilePath}", filePath);
            
            // Read the file content
            var sourceCode = await File.ReadAllTextAsync(filePath);
            
            // Create the syntax tree
            var syntaxTree = CSharpSyntaxTree.ParseText(sourceCode);
            var root = await syntaxTree.GetRootAsync();
            
            // Find duplicated blocks
            var duplicatedBlocks = FindDuplicatedBlocks(root, filePath);
            
            // Calculate duplication metrics
            var totalLines = sourceCode.Split('\n').Length;
            var duplicatedLines = duplicatedBlocks.Sum(b => b.LineCount);
            var duplicationPercentage = totalLines > 0 ? (double)duplicatedLines / totalLines * 100 : 0;
            
            // Create the result
            var result = new DuplicationResult
            {
                FilePath = filePath,
                TotalLines = totalLines,
                DuplicatedLines = duplicatedLines,
                DuplicationPercentage = duplicationPercentage,
                DuplicatedBlocks = duplicatedBlocks
            };
            
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing duplication in file {FilePath}", filePath);
            throw;
        }
    }
    
    /// <summary>
    /// Analyzes duplication in a directory
    /// </summary>
    /// <param name="directoryPath">Path to the directory</param>
    /// <param name="searchPattern">Search pattern for files</param>
    /// <returns>Duplication analysis results</returns>
    public async Task<List<DuplicationResult>> AnalyzeDirectoryAsync(string directoryPath, string searchPattern = "*.cs")
    {
        try
        {
            _logger.LogInformation("Analyzing duplication in directory {DirectoryPath}", directoryPath);
            
            // Get all files matching the pattern
            var files = Directory.GetFiles(directoryPath, searchPattern, SearchOption.AllDirectories);
            
            // Analyze each file
            var results = new List<DuplicationResult>();
            foreach (var file in files)
            {
                try
                {
                    var result = await AnalyzeFileAsync(file);
                    results.Add(result);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error analyzing file {FilePath}", file);
                }
            }
            
            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing directory {DirectoryPath}", directoryPath);
            throw;
        }
    }
    
    /// <summary>
    /// Finds duplicated blocks in a syntax tree
    /// </summary>
    /// <param name="root">Syntax tree root</param>
    /// <param name="filePath">File path</param>
    /// <returns>List of duplicated blocks</returns>
    private List<DuplicatedBlock> FindDuplicatedBlocks(SyntaxNode root, string filePath)
    {
        var duplicatedBlocks = new List<DuplicatedBlock>();
        
        // Get all method declarations
        var methodDeclarations = root.DescendantNodes().OfType<MethodDeclarationSyntax>().ToList();
        
        // Compare each method with every other method
        for (int i = 0; i < methodDeclarations.Count; i++)
        {
            for (int j = i + 1; j < methodDeclarations.Count; j++)
            {
                var method1 = methodDeclarations[i];
                var method2 = methodDeclarations[j];
                
                // Find duplicated blocks between these methods
                var blocks = FindDuplicatedBlocksBetweenMethods(method1, method2, filePath);
                duplicatedBlocks.AddRange(blocks);
            }
        }
        
        // Find duplicated blocks within methods
        foreach (var method in methodDeclarations)
        {
            var blocks = FindDuplicatedBlocksWithinMethod(method, filePath);
            duplicatedBlocks.AddRange(blocks);
        }
        
        return duplicatedBlocks;
    }
    
    /// <summary>
    /// Finds duplicated blocks between two methods
    /// </summary>
    /// <param name="method1">First method</param>
    /// <param name="method2">Second method</param>
    /// <param name="filePath">File path</param>
    /// <returns>List of duplicated blocks</returns>
    private List<DuplicatedBlock> FindDuplicatedBlocksBetweenMethods(MethodDeclarationSyntax method1, MethodDeclarationSyntax method2, string filePath)
    {
        var duplicatedBlocks = new List<DuplicatedBlock>();
        
        // Get the statements from each method
        var statements1 = method1.Body?.Statements.ToList() ?? new List<StatementSyntax>();
        var statements2 = method2.Body?.Statements.ToList() ?? new List<StatementSyntax>();
        
        // Skip if either method has too few statements
        if (statements1.Count < 3 || statements2.Count < 3)
        {
            return duplicatedBlocks;
        }
        
        // Compare statement blocks
        for (int i = 0; i <= statements1.Count - 3; i++)
        {
            for (int j = 0; j <= statements2.Count - 3; j++)
            {
                // Check if the next 3 statements are similar
                if (AreSimilarStatements(statements1.Skip(i).Take(3).ToList(), statements2.Skip(j).Take(3).ToList()))
                {
                    // Find the maximum length of similar statements
                    int length = 3;
                    while (i + length < statements1.Count && j + length < statements2.Count &&
                           AreSimilarStatements(statements1[i + length], statements2[j + length]))
                    {
                        length++;
                    }
                    
                    // Create a duplicated block
                    var block = new DuplicatedBlock
                    {
                        SourceMethod = method1.Identifier.Text,
                        TargetMethod = method2.Identifier.Text,
                        SourceStartLine = statements1[i].GetLocation().GetLineSpan().StartLinePosition.Line + 1,
                        SourceEndLine = statements1[i + length - 1].GetLocation().GetLineSpan().EndLinePosition.Line + 1,
                        TargetStartLine = statements2[j].GetLocation().GetLineSpan().StartLinePosition.Line + 1,
                        TargetEndLine = statements2[j + length - 1].GetLocation().GetLineSpan().EndLinePosition.Line + 1,
                        LineCount = length,
                        SimilarityType = SimilarityType.Exact,
                        FilePath = filePath
                    };
                    
                    duplicatedBlocks.Add(block);
                    
                    // Skip the duplicated statements
                    i += length - 1;
                    break;
                }
            }
        }
        
        return duplicatedBlocks;
    }
    
    /// <summary>
    /// Finds duplicated blocks within a method
    /// </summary>
    /// <param name="method">Method</param>
    /// <param name="filePath">File path</param>
    /// <returns>List of duplicated blocks</returns>
    private List<DuplicatedBlock> FindDuplicatedBlocksWithinMethod(MethodDeclarationSyntax method, string filePath)
    {
        var duplicatedBlocks = new List<DuplicatedBlock>();
        
        // Get the statements from the method
        var statements = method.Body?.Statements.ToList() ?? new List<StatementSyntax>();
        
        // Skip if the method has too few statements
        if (statements.Count < 6)
        {
            return duplicatedBlocks;
        }
        
        // Compare statement blocks within the method
        for (int i = 0; i <= statements.Count - 3; i++)
        {
            for (int j = i + 3; j <= statements.Count - 3; j++)
            {
                // Check if the next 3 statements are similar
                if (AreSimilarStatements(statements.Skip(i).Take(3).ToList(), statements.Skip(j).Take(3).ToList()))
                {
                    // Find the maximum length of similar statements
                    int length = 3;
                    while (i + length < j && j + length < statements.Count &&
                           AreSimilarStatements(statements[i + length], statements[j + length]))
                    {
                        length++;
                    }
                    
                    // Create a duplicated block
                    var block = new DuplicatedBlock
                    {
                        SourceMethod = method.Identifier.Text,
                        TargetMethod = method.Identifier.Text,
                        SourceStartLine = statements[i].GetLocation().GetLineSpan().StartLinePosition.Line + 1,
                        SourceEndLine = statements[i + length - 1].GetLocation().GetLineSpan().EndLinePosition.Line + 1,
                        TargetStartLine = statements[j].GetLocation().GetLineSpan().StartLinePosition.Line + 1,
                        TargetEndLine = statements[j + length - 1].GetLocation().GetLineSpan().EndLinePosition.Line + 1,
                        LineCount = length,
                        SimilarityType = SimilarityType.Exact,
                        FilePath = filePath
                    };
                    
                    duplicatedBlocks.Add(block);
                    
                    // Skip the duplicated statements
                    i += length - 1;
                    break;
                }
            }
        }
        
        return duplicatedBlocks;
    }
    
    /// <summary>
    /// Checks if two lists of statements are similar
    /// </summary>
    /// <param name="statements1">First list of statements</param>
    /// <param name="statements2">Second list of statements</param>
    /// <returns>True if the statements are similar</returns>
    private bool AreSimilarStatements(List<StatementSyntax> statements1, List<StatementSyntax> statements2)
    {
        if (statements1.Count != statements2.Count)
        {
            return false;
        }
        
        for (int i = 0; i < statements1.Count; i++)
        {
            if (!AreSimilarStatements(statements1[i], statements2[i]))
            {
                return false;
            }
        }
        
        return true;
    }
    
    /// <summary>
    /// Checks if two statements are similar
    /// </summary>
    /// <param name="statement1">First statement</param>
    /// <param name="statement2">Second statement</param>
    /// <returns>True if the statements are similar</returns>
    private bool AreSimilarStatements(StatementSyntax statement1, StatementSyntax statement2)
    {
        // Check if the statements are of the same type
        if (statement1.GetType() != statement2.GetType())
        {
            return false;
        }
        
        // For simple statements, compare the normalized text
        var text1 = NormalizeStatement(statement1.ToString());
        var text2 = NormalizeStatement(statement2.ToString());
        
        return text1 == text2;
    }
    
    /// <summary>
    /// Normalizes a statement by removing whitespace and variable names
    /// </summary>
    /// <param name="statement">Statement</param>
    /// <returns>Normalized statement</returns>
    private string NormalizeStatement(string statement)
    {
        // Remove whitespace
        statement = Regex.Replace(statement, @"\s+", " ").Trim();
        
        // Replace variable names with placeholders
        statement = Regex.Replace(statement, @"\b[a-zA-Z_][a-zA-Z0-9_]*\b", "VAR");
        
        // Replace numeric literals with placeholders
        statement = Regex.Replace(statement, @"\b\d+\b", "NUM");
        
        // Replace string literals with placeholders
        statement = Regex.Replace(statement, @"""[^""]*""", "STR");
        
        return statement;
    }
}

/// <summary>
/// Represents a duplication analysis result
/// </summary>
public class DuplicationResult
{
    /// <summary>
    /// Gets or sets the file path
    /// </summary>
    public string FilePath { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the total number of lines
    /// </summary>
    public int TotalLines { get; set; }
    
    /// <summary>
    /// Gets or sets the number of duplicated lines
    /// </summary>
    public int DuplicatedLines { get; set; }
    
    /// <summary>
    /// Gets or sets the duplication percentage
    /// </summary>
    public double DuplicationPercentage { get; set; }
    
    /// <summary>
    /// Gets or sets the duplicated blocks
    /// </summary>
    public List<DuplicatedBlock> DuplicatedBlocks { get; set; } = new();
    
    /// <summary>
    /// Gets the duplication level
    /// </summary>
    public DuplicationLevel DuplicationLevel
    {
        get
        {
            if (DuplicationPercentage < 3)
            {
                return DuplicationLevel.Low;
            }
            else if (DuplicationPercentage < 10)
            {
                return DuplicationLevel.Moderate;
            }
            else if (DuplicationPercentage < 20)
            {
                return DuplicationLevel.High;
            }
            else
            {
                return DuplicationLevel.VeryHigh;
            }
        }
    }
    
    /// <summary>
    /// Gets a formatted report of the duplication analysis
    /// </summary>
    /// <returns>Formatted report</returns>
    public string GetReport()
    {
        var sb = new StringBuilder();
        
        sb.AppendLine($"File: {Path.GetFileName(FilePath)}");
        sb.AppendLine($"Total Lines: {TotalLines}");
        sb.AppendLine($"Duplicated Lines: {DuplicatedLines}");
        sb.AppendLine($"Duplication Percentage: {DuplicationPercentage:F2}%");
        sb.AppendLine($"Duplication Level: {DuplicationLevel}");
        sb.AppendLine();
        
        if (DuplicatedBlocks.Count > 0)
        {
            sb.AppendLine("Duplicated Blocks:");
            
            foreach (var block in DuplicatedBlocks)
            {
                sb.AppendLine($"  - {block.SourceMethod} (Lines {block.SourceStartLine}-{block.SourceEndLine}) and {block.TargetMethod} (Lines {block.TargetStartLine}-{block.TargetEndLine})");
                sb.AppendLine($"    Line Count: {block.LineCount}");
                sb.AppendLine($"    Similarity: {block.SimilarityType}");
                sb.AppendLine();
            }
        }
        else
        {
            sb.AppendLine("No duplicated blocks found.");
        }
        
        return sb.ToString();
    }
    
    /// <summary>
    /// Gets an HTML report of the duplication analysis
    /// </summary>
    /// <returns>HTML report</returns>
    public string GetHtmlReport()
    {
        var sb = new StringBuilder();
        
        sb.AppendLine("<!DOCTYPE html>");
        sb.AppendLine("<html>");
        sb.AppendLine("<head>");
        sb.AppendLine("  <title>Duplication Analysis Report</title>");
        sb.AppendLine("  <style>");
        sb.AppendLine("    body { font-family: Arial, sans-serif; margin: 20px; }");
        sb.AppendLine("    h1, h2 { color: #333; }");
        sb.AppendLine("    .low { color: green; }");
        sb.AppendLine("    .moderate { color: orange; }");
        sb.AppendLine("    .high { color: red; }");
        sb.AppendLine("    .veryhigh { color: darkred; }");
        sb.AppendLine("    .block { margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; }");
        sb.AppendLine("    .metrics { display: flex; flex-wrap: wrap; }");
        sb.AppendLine("    .metric { flex: 1; min-width: 200px; margin: 10px; padding: 10px; border: 1px solid #ddd; }");
        sb.AppendLine("  </style>");
        sb.AppendLine("</head>");
        sb.AppendLine("<body>");
        
        sb.AppendLine($"  <h1>Duplication Analysis Report for {Path.GetFileName(FilePath)}</h1>");
        
        sb.AppendLine("  <div class=\"metrics\">");
        sb.AppendLine("    <div class=\"metric\">");
        sb.AppendLine("      <h3>Total Lines</h3>");
        sb.AppendLine($"      <p>{TotalLines}</p>");
        sb.AppendLine("    </div>");
        
        sb.AppendLine("    <div class=\"metric\">");
        sb.AppendLine("      <h3>Duplicated Lines</h3>");
        sb.AppendLine($"      <p>{DuplicatedLines}</p>");
        sb.AppendLine("    </div>");
        
        sb.AppendLine("    <div class=\"metric\">");
        sb.AppendLine("      <h3>Duplication Percentage</h3>");
        sb.AppendLine($"      <p>{DuplicationPercentage:F2}%</p>");
        sb.AppendLine("    </div>");
        
        sb.AppendLine("    <div class=\"metric\">");
        sb.AppendLine("      <h3>Duplication Level</h3>");
        sb.AppendLine($"      <p class=\"{DuplicationLevel.ToString().ToLowerInvariant()}\">{DuplicationLevel}</p>");
        sb.AppendLine("    </div>");
        sb.AppendLine("  </div>");
        
        if (DuplicatedBlocks.Count > 0)
        {
            sb.AppendLine("  <h2>Duplicated Blocks</h2>");
            
            foreach (var block in DuplicatedBlocks)
            {
                sb.AppendLine("  <div class=\"block\">");
                sb.AppendLine($"    <h3>{block.SourceMethod} and {block.TargetMethod}</h3>");
                sb.AppendLine($"    <p>Source: Lines {block.SourceStartLine}-{block.SourceEndLine}</p>");
                sb.AppendLine($"    <p>Target: Lines {block.TargetStartLine}-{block.TargetEndLine}</p>");
                sb.AppendLine($"    <p>Line Count: {block.LineCount}</p>");
                sb.AppendLine($"    <p>Similarity: {block.SimilarityType}</p>");
                sb.AppendLine("  </div>");
            }
        }
        else
        {
            sb.AppendLine("  <h2>No duplicated blocks found.</h2>");
        }
        
        sb.AppendLine("</body>");
        sb.AppendLine("</html>");
        
        return sb.ToString();
    }
}

/// <summary>
/// Represents a duplicated block of code
/// </summary>
public class DuplicatedBlock
{
    /// <summary>
    /// Gets or sets the file path
    /// </summary>
    public string FilePath { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the source method
    /// </summary>
    public string SourceMethod { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the target method
    /// </summary>
    public string TargetMethod { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the source start line
    /// </summary>
    public int SourceStartLine { get; set; }
    
    /// <summary>
    /// Gets or sets the source end line
    /// </summary>
    public int SourceEndLine { get; set; }
    
    /// <summary>
    /// Gets or sets the target start line
    /// </summary>
    public int TargetStartLine { get; set; }
    
    /// <summary>
    /// Gets or sets the target end line
    /// </summary>
    public int TargetEndLine { get; set; }
    
    /// <summary>
    /// Gets or sets the line count
    /// </summary>
    public int LineCount { get; set; }
    
    /// <summary>
    /// Gets or sets the similarity type
    /// </summary>
    public SimilarityType SimilarityType { get; set; }
}

/// <summary>
/// Represents a duplication level
/// </summary>
public enum DuplicationLevel
{
    /// <summary>
    /// Low duplication (less than 3%)
    /// </summary>
    Low,
    
    /// <summary>
    /// Moderate duplication (3-10%)
    /// </summary>
    Moderate,
    
    /// <summary>
    /// High duplication (10-20%)
    /// </summary>
    High,
    
    /// <summary>
    /// Very high duplication (more than 20%)
    /// </summary>
    VeryHigh
}

/// <summary>
/// Represents a similarity type
/// </summary>
public enum SimilarityType
{
    /// <summary>
    /// Exact match
    /// </summary>
    Exact,
    
    /// <summary>
    /// Semantic similarity
    /// </summary>
    Semantic
}
