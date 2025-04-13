using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.Extensions.Logging;

namespace TarsEngine.Services;

/// <summary>
/// Analyzer for semantic similarity between code blocks
/// </summary>
public class SemanticSimilarityAnalyzer
{
    private readonly ILogger _logger;
    
    // Minimum similarity percentage to consider as semantic duplication
    private readonly double _minimumSimilarityPercentage;
    
    // Minimum sequence length to consider as duplication (in lines)
    private readonly int _minimumDuplicateLines;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="SemanticSimilarityAnalyzer"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    /// <param name="minimumSimilarityPercentage">Minimum similarity percentage</param>
    /// <param name="minimumDuplicateLines">Minimum number of lines to consider as duplication</param>
    public SemanticSimilarityAnalyzer(ILogger logger, double minimumSimilarityPercentage = 80, int minimumDuplicateLines = 5)
    {
        _logger = logger;
        _minimumSimilarityPercentage = minimumSimilarityPercentage;
        _minimumDuplicateLines = minimumDuplicateLines;
    }
    
    /// <summary>
    /// Calculates the semantic similarity between two code blocks
    /// </summary>
    /// <param name="code1">First code block</param>
    /// <param name="code2">Second code block</param>
    /// <returns>Similarity percentage (0-100)</returns>
    public double CalculateSimilarity(string code1, string code2)
    {
        try
        {
            // Normalize the code blocks
            var normalizedCode1 = NormalizeCode(code1);
            var normalizedCode2 = NormalizeCode(code2);
            
            // Calculate structural similarity
            var structuralSimilarity = CalculateStructuralSimilarity(normalizedCode1, normalizedCode2);
            
            // Calculate variable usage similarity
            var variableUsageSimilarity = CalculateVariableUsageSimilarity(code1, code2);
            
            // Calculate control flow similarity
            var controlFlowSimilarity = CalculateControlFlowSimilarity(code1, code2);
            
            // Calculate overall similarity (weighted average)
            var overallSimilarity = (structuralSimilarity * 0.5) + (variableUsageSimilarity * 0.3) + (controlFlowSimilarity * 0.2);
            
            return overallSimilarity;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating semantic similarity");
            return 0;
        }
    }
    
    /// <summary>
    /// Finds semantically similar code blocks in a file
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <param name="sourceCode">Source code</param>
    /// <returns>List of similar code blocks (start line, end line, similarity percentage)</returns>
    public List<(int StartLine, int EndLine, int SimilarStartLine, int SimilarEndLine, double SimilarityPercentage)> 
        FindSimilarCodeBlocks(string filePath, string sourceCode)
    {
        try
        {
            var similarBlocks = new List<(int, int, int, int, double)>();
            
            // Parse the source code
            var syntaxTree = CSharpSyntaxTree.ParseText(sourceCode);
            var root = syntaxTree.GetRoot();
            
            // Get all method declarations
            var methodDeclarations = root.DescendantNodes().OfType<MethodDeclarationSyntax>().ToList();
            
            // Compare each method with every other method
            for (int i = 0; i < methodDeclarations.Count; i++)
            {
                for (int j = i + 1; j < methodDeclarations.Count; j++)
                {
                    var method1 = methodDeclarations[i];
                    var method2 = methodDeclarations[j];
                    
                    // Skip if either method is too small
                    if (method1.Body == null || method2.Body == null)
                    {
                        continue;
                    }
                    
                    var method1Code = method1.Body.ToString();
                    var method2Code = method2.Body.ToString();
                    
                    var method1Lines = method1Code.Split('\n').Length;
                    var method2Lines = method2Code.Split('\n').Length;
                    
                    if (method1Lines < _minimumDuplicateLines || method2Lines < _minimumDuplicateLines)
                    {
                        continue;
                    }
                    
                    // Calculate similarity
                    var similarity = CalculateSimilarity(method1Code, method2Code);
                    
                    // If similarity is above threshold, add to the list
                    if (similarity >= _minimumSimilarityPercentage)
                    {
                        var method1StartLine = GetLineNumber(sourceCode, method1.Body.Span.Start);
                        var method1EndLine = GetLineNumber(sourceCode, method1.Body.Span.End);
                        var method2StartLine = GetLineNumber(sourceCode, method2.Body.Span.Start);
                        var method2EndLine = GetLineNumber(sourceCode, method2.Body.Span.End);
                        
                        similarBlocks.Add((method1StartLine, method1EndLine, method2StartLine, method2EndLine, similarity));
                    }
                }
            }
            
            return similarBlocks;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error finding similar code blocks in file {FilePath}", filePath);
            return new List<(int, int, int, int, double)>();
        }
    }
    
    /// <summary>
    /// Finds semantically similar code blocks across multiple files
    /// </summary>
    /// <param name="files">Dictionary of file paths and source code</param>
    /// <returns>List of similar code blocks (file path, start line, end line, similar file path, similar start line, similar end line, similarity percentage)</returns>
    public List<(string FilePath, int StartLine, int EndLine, string SimilarFilePath, int SimilarStartLine, int SimilarEndLine, double SimilarityPercentage)> 
        FindSimilarCodeBlocksAcrossFiles(Dictionary<string, string> files)
    {
        try
        {
            var similarBlocks = new List<(string, int, int, string, int, int, double)>();
            
            // Parse each file
            var fileSyntaxTrees = new Dictionary<string, (SyntaxTree Tree, SyntaxNode Root)>();
            foreach (var file in files)
            {
                var syntaxTree = CSharpSyntaxTree.ParseText(file.Value);
                var root = syntaxTree.GetRoot();
                fileSyntaxTrees[file.Key] = (syntaxTree, root);
            }
            
            // Compare methods across files
            var processedPairs = new HashSet<string>();
            foreach (var file1 in files)
            {
                foreach (var file2 in files)
                {
                    // Skip comparing a file with itself
                    if (file1.Key == file2.Key)
                    {
                        continue;
                    }
                    
                    // Create a unique key for this file pair to avoid processing the same pair twice
                    var pairKey = file1.Key.CompareTo(file2.Key) < 0
                        ? $"{file1.Key}|{file2.Key}"
                        : $"{file2.Key}|{file1.Key}";
                    
                    if (processedPairs.Contains(pairKey))
                    {
                        continue;
                    }
                    
                    processedPairs.Add(pairKey);
                    
                    // Get methods from both files
                    var methods1 = fileSyntaxTrees[file1.Key].Root.DescendantNodes().OfType<MethodDeclarationSyntax>().ToList();
                    var methods2 = fileSyntaxTrees[file2.Key].Root.DescendantNodes().OfType<MethodDeclarationSyntax>().ToList();
                    
                    // Compare each method from file1 with each method from file2
                    foreach (var method1 in methods1)
                    {
                        foreach (var method2 in methods2)
                        {
                            // Skip if either method is too small
                            if (method1.Body == null || method2.Body == null)
                            {
                                continue;
                            }
                            
                            var method1Code = method1.Body.ToString();
                            var method2Code = method2.Body.ToString();
                            
                            var method1Lines = method1Code.Split('\n').Length;
                            var method2Lines = method2Code.Split('\n').Length;
                            
                            if (method1Lines < _minimumDuplicateLines || method2Lines < _minimumDuplicateLines)
                            {
                                continue;
                            }
                            
                            // Calculate similarity
                            var similarity = CalculateSimilarity(method1Code, method2Code);
                            
                            // If similarity is above threshold, add to the list
                            if (similarity >= _minimumSimilarityPercentage)
                            {
                                var method1StartLine = GetLineNumber(file1.Value, method1.Body.Span.Start);
                                var method1EndLine = GetLineNumber(file1.Value, method1.Body.Span.End);
                                var method2StartLine = GetLineNumber(file2.Value, method2.Body.Span.Start);
                                var method2EndLine = GetLineNumber(file2.Value, method2.Body.Span.End);
                                
                                similarBlocks.Add((file1.Key, method1StartLine, method1EndLine, file2.Key, method2StartLine, method2EndLine, similarity));
                            }
                        }
                    }
                }
            }
            
            return similarBlocks;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error finding similar code blocks across files");
            return new List<(string, int, int, string, int, int, double)>();
        }
    }
    
    /// <summary>
    /// Normalizes code by removing comments, whitespace, and renaming variables
    /// </summary>
    /// <param name="code">Code to normalize</param>
    /// <returns>Normalized code</returns>
    private string NormalizeCode(string code)
    {
        // Remove comments
        code = Regex.Replace(code, @"//.*$", "", RegexOptions.Multiline);
        code = Regex.Replace(code, @"/\*.*?\*/", "", RegexOptions.Singleline);
        
        // Remove whitespace
        code = Regex.Replace(code, @"\s+", " ");
        
        // Remove string literals
        code = Regex.Replace(code, @""".*?""", "\"\"");
        
        // Remove numeric literals
        code = Regex.Replace(code, @"\b\d+\b", "0");
        
        return code;
    }
    
    /// <summary>
    /// Calculates the structural similarity between two code blocks
    /// </summary>
    /// <param name="code1">First code block</param>
    /// <param name="code2">Second code block</param>
    /// <returns>Similarity percentage (0-100)</returns>
    private double CalculateStructuralSimilarity(string code1, string code2)
    {
        // Use Levenshtein distance to calculate similarity
        var distance = LevenshteinDistance(code1, code2);
        var maxLength = Math.Max(code1.Length, code2.Length);
        
        if (maxLength == 0)
        {
            return 100;
        }
        
        return (1 - (double)distance / maxLength) * 100;
    }
    
    /// <summary>
    /// Calculates the variable usage similarity between two code blocks
    /// </summary>
    /// <param name="code1">First code block</param>
    /// <param name="code2">Second code block</param>
    /// <returns>Similarity percentage (0-100)</returns>
    private double CalculateVariableUsageSimilarity(string code1, string code2)
    {
        try
        {
            // Parse the code blocks
            var syntaxTree1 = CSharpSyntaxTree.ParseText(code1);
            var syntaxTree2 = CSharpSyntaxTree.ParseText(code2);
            
            var root1 = syntaxTree1.GetRoot();
            var root2 = syntaxTree2.GetRoot();
            
            // Get variable declarations
            var variables1 = root1.DescendantNodes().OfType<VariableDeclarationSyntax>().ToList();
            var variables2 = root2.DescendantNodes().OfType<VariableDeclarationSyntax>().ToList();
            
            // Get variable types
            var variableTypes1 = variables1.Select(v => v.Type.ToString()).ToList();
            var variableTypes2 = variables2.Select(v => v.Type.ToString()).ToList();
            
            // Calculate Jaccard similarity for variable types
            var similarity = CalculateJaccardSimilarity(variableTypes1, variableTypes2);
            
            return similarity * 100;
        }
        catch
        {
            // If parsing fails, fall back to a simpler approach
            return 0;
        }
    }
    
    /// <summary>
    /// Calculates the control flow similarity between two code blocks
    /// </summary>
    /// <param name="code1">First code block</param>
    /// <param name="code2">Second code block</param>
    /// <returns>Similarity percentage (0-100)</returns>
    private double CalculateControlFlowSimilarity(string code1, string code2)
    {
        try
        {
            // Parse the code blocks
            var syntaxTree1 = CSharpSyntaxTree.ParseText(code1);
            var syntaxTree2 = CSharpSyntaxTree.ParseText(code2);
            
            var root1 = syntaxTree1.GetRoot();
            var root2 = syntaxTree2.GetRoot();
            
            // Count control flow statements
            var ifCount1 = root1.DescendantNodes().OfType<IfStatementSyntax>().Count();
            var ifCount2 = root2.DescendantNodes().OfType<IfStatementSyntax>().Count();
            
            var forCount1 = root1.DescendantNodes().OfType<ForStatementSyntax>().Count();
            var forCount2 = root2.DescendantNodes().OfType<ForStatementSyntax>().Count();
            
            var foreachCount1 = root1.DescendantNodes().OfType<ForEachStatementSyntax>().Count();
            var foreachCount2 = root2.DescendantNodes().OfType<ForEachStatementSyntax>().Count();
            
            var whileCount1 = root1.DescendantNodes().OfType<WhileStatementSyntax>().Count();
            var whileCount2 = root2.DescendantNodes().OfType<WhileStatementSyntax>().Count();
            
            var switchCount1 = root1.DescendantNodes().OfType<SwitchStatementSyntax>().Count();
            var switchCount2 = root2.DescendantNodes().OfType<SwitchStatementSyntax>().Count();
            
            var tryCount1 = root1.DescendantNodes().OfType<TryStatementSyntax>().Count();
            var tryCount2 = root2.DescendantNodes().OfType<TryStatementSyntax>().Count();
            
            // Calculate similarity for each control flow type
            var ifSimilarity = CalculateSimilarityRatio(ifCount1, ifCount2);
            var forSimilarity = CalculateSimilarityRatio(forCount1, forCount2);
            var foreachSimilarity = CalculateSimilarityRatio(foreachCount1, foreachCount2);
            var whileSimilarity = CalculateSimilarityRatio(whileCount1, whileCount2);
            var switchSimilarity = CalculateSimilarityRatio(switchCount1, switchCount2);
            var trySimilarity = CalculateSimilarityRatio(tryCount1, tryCount2);
            
            // Calculate overall control flow similarity
            var overallSimilarity = (ifSimilarity + forSimilarity + foreachSimilarity + whileSimilarity + switchSimilarity + trySimilarity) / 6;
            
            return overallSimilarity * 100;
        }
        catch
        {
            // If parsing fails, fall back to a simpler approach
            return 0;
        }
    }
    
    /// <summary>
    /// Calculates the Levenshtein distance between two strings
    /// </summary>
    /// <param name="s">First string</param>
    /// <param name="t">Second string</param>
    /// <returns>Levenshtein distance</returns>
    private int LevenshteinDistance(string s, string t)
    {
        int m = s.Length;
        int n = t.Length;
        
        int[,] d = new int[m + 1, n + 1];
        
        if (m == 0)
        {
            return n;
        }
        
        if (n == 0)
        {
            return m;
        }
        
        for (int i = 0; i <= m; i++)
        {
            d[i, 0] = i;
        }
        
        for (int j = 0; j <= n; j++)
        {
            d[0, j] = j;
        }
        
        for (int j = 1; j <= n; j++)
        {
            for (int i = 1; i <= m; i++)
            {
                int cost = (s[i - 1] == t[j - 1]) ? 0 : 1;
                
                d[i, j] = Math.Min(
                    Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                    d[i - 1, j - 1] + cost);
            }
        }
        
        return d[m, n];
    }
    
    /// <summary>
    /// Calculates the Jaccard similarity between two sets
    /// </summary>
    /// <typeparam name="T">Type of elements in the sets</typeparam>
    /// <param name="set1">First set</param>
    /// <param name="set2">Second set</param>
    /// <returns>Jaccard similarity (0-1)</returns>
    private double CalculateJaccardSimilarity<T>(IEnumerable<T> set1, IEnumerable<T> set2)
    {
        var intersection = set1.Intersect(set2).Count();
        var union = set1.Union(set2).Count();
        
        if (union == 0)
        {
            return 1;
        }
        
        return (double)intersection / union;
    }
    
    /// <summary>
    /// Calculates the similarity ratio between two counts
    /// </summary>
    /// <param name="count1">First count</param>
    /// <param name="count2">Second count</param>
    /// <returns>Similarity ratio (0-1)</returns>
    private double CalculateSimilarityRatio(int count1, int count2)
    {
        if (count1 == 0 && count2 == 0)
        {
            return 1;
        }
        
        var max = Math.Max(count1, count2);
        var min = Math.Min(count1, count2);
        
        return (double)min / max;
    }
    
    /// <summary>
    /// Gets the line number for a position in the source code
    /// </summary>
    /// <param name="sourceCode">Source code</param>
    /// <param name="position">Position</param>
    /// <returns>Line number</returns>
    private int GetLineNumber(string sourceCode, int position)
    {
        // Count the number of newlines before the position
        var lineNumber = 1;
        for (int i = 0; i < position && i < sourceCode.Length; i++)
        {
            if (sourceCode[i] == '\n')
            {
                lineNumber++;
            }
        }
        
        return lineNumber;
    }
}
