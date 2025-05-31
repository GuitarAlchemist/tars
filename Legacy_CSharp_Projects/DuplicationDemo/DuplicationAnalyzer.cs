using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Web;
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

        // Compare statement blocks for exact duplication
        duplicatedBlocks.AddRange(FindExactDuplicatedBlocks(method1, method2, statements1, statements2, filePath));

        // Compare statement blocks for semantic duplication
        duplicatedBlocks.AddRange(FindSemanticDuplicatedBlocks(method1, method2, statements1, statements2, filePath));

        return duplicatedBlocks;
    }

    /// <summary>
    /// Finds exact duplicated blocks between two methods
    /// </summary>
    /// <param name="method1">First method</param>
    /// <param name="method2">Second method</param>
    /// <param name="statements1">Statements from first method</param>
    /// <param name="statements2">Statements from second method</param>
    /// <param name="filePath">File path</param>
    /// <returns>List of duplicated blocks</returns>
    private List<DuplicatedBlock> FindExactDuplicatedBlocks(
        MethodDeclarationSyntax method1,
        MethodDeclarationSyntax method2,
        List<StatementSyntax> statements1,
        List<StatementSyntax> statements2,
        string filePath)
    {
        var duplicatedBlocks = new List<DuplicatedBlock>();

        // Compare statement blocks
        for (int i = 0; i <= statements1.Count - 3; i++)
        {
            for (int j = 0; j <= statements2.Count - 3; j++)
            {
                // Check if the next 3 statements are exactly the same
                if (AreExactlyTheSameStatements(statements1.Skip(i).Take(3).ToList(), statements2.Skip(j).Take(3).ToList()))
                {
                    // Find the maximum length of similar statements
                    int length = 3;
                    while (i + length < statements1.Count && j + length < statements2.Count &&
                           AreExactlyTheSameStatements(statements1[i + length], statements2[j + length]))
                    {
                        length++;
                    }

                    // Get the duplicated code
                    var fullBlock = statements1.Skip(i).Take(length).ToList();
                    var sourceCode = string.Join("\n", fullBlock.Select(s => s.ToString().Trim()));

                    // Generate refactoring recommendation
                    var recommendation = GenerateRefactoringRecommendation(fullBlock, method1.Identifier.Text, method2.Identifier.Text);

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
                        SimilarityPercentage = 100.0,
                        RefactoringRecommendation = recommendation,
                        DuplicatedCode = sourceCode,
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
    /// Finds semantically similar blocks between two methods
    /// </summary>
    /// <param name="method1">First method</param>
    /// <param name="method2">Second method</param>
    /// <param name="statements1">Statements from first method</param>
    /// <param name="statements2">Statements from second method</param>
    /// <param name="filePath">File path</param>
    /// <returns>List of duplicated blocks</returns>
    private List<DuplicatedBlock> FindSemanticDuplicatedBlocks(
        MethodDeclarationSyntax method1,
        MethodDeclarationSyntax method2,
        List<StatementSyntax> statements1,
        List<StatementSyntax> statements2,
        string filePath)
    {
        var duplicatedBlocks = new List<DuplicatedBlock>();

        // Compare statement blocks
        for (int i = 0; i <= statements1.Count - 3; i++)
        {
            for (int j = 0; j <= statements2.Count - 3; j++)
            {
                // Check if the next 3 statements are semantically similar
                var statementsBlock1 = statements1.Skip(i).Take(3).ToList();
                var statementsBlock2 = statements2.Skip(j).Take(3).ToList();
                var similarityScore = CalculateBlockSimilarity(statementsBlock1, statementsBlock2);

                if (similarityScore >= 0.7) // 70% similarity threshold
                {
                    // Find the maximum length of similar statements
                    int length = 3;
                    while (i + length < statements1.Count && j + length < statements2.Count &&
                           CalculateStatementSimilarity(statements1[i + length], statements2[j + length]) >= 0.7)
                    {
                        length++;
                    }

                    // Calculate overall similarity for the entire block
                    var fullBlock1 = statements1.Skip(i).Take(length).ToList();
                    var fullBlock2 = statements2.Skip(j).Take(length).ToList();
                    var overallSimilarity = CalculateBlockSimilarity(fullBlock1, fullBlock2);

                    // Get the duplicated code
                    var sourceCode = string.Join("\n", fullBlock1.Select(s => s.ToString().Trim()));

                    // Generate refactoring recommendation
                    var recommendation = GenerateRefactoringRecommendation(fullBlock1, method1.Identifier.Text, method2.Identifier.Text);

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
                        SimilarityType = SimilarityType.Semantic,
                        SimilarityPercentage = overallSimilarity * 100,
                        RefactoringRecommendation = recommendation,
                        DuplicatedCode = sourceCode,
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
    /// Checks if two lists of statements are exactly the same
    /// </summary>
    /// <param name="statements1">First list of statements</param>
    /// <param name="statements2">Second list of statements</param>
    /// <returns>True if the statements are exactly the same</returns>
    private bool AreExactlyTheSameStatements(List<StatementSyntax> statements1, List<StatementSyntax> statements2)
    {
        if (statements1.Count != statements2.Count)
        {
            return false;
        }

        for (int i = 0; i < statements1.Count; i++)
        {
            var text1 = NormalizeStatement(statements1[i].ToString());
            var text2 = NormalizeStatement(statements2[i].ToString());

            if (text1 != text2)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Checks if a statement is exactly the same as another statement
    /// </summary>
    /// <param name="statement1">First statement</param>
    /// <param name="statement2">Second statement</param>
    /// <returns>True if the statements are exactly the same</returns>
    private bool AreExactlyTheSameStatements(StatementSyntax statement1, StatementSyntax statement2)
    {
        if (statement1.GetType() != statement2.GetType())
        {
            return false;
        }

        var text1 = NormalizeStatement(statement1.ToString());
        var text2 = NormalizeStatement(statement2.ToString());

        return text1 == text2;
    }

    /// <summary>
    /// Calculates the similarity between two blocks of statements
    /// </summary>
    /// <param name="statements1">First list of statements</param>
    /// <param name="statements2">Second list of statements</param>
    /// <returns>Similarity score between 0 and 1</returns>
    private double CalculateBlockSimilarity(List<StatementSyntax> statements1, List<StatementSyntax> statements2)
    {
        if (statements1.Count != statements2.Count || statements1.Count == 0)
        {
            return 0;
        }

        double totalSimilarity = 0;

        for (int i = 0; i < statements1.Count; i++)
        {
            totalSimilarity += CalculateStatementSimilarity(statements1[i], statements2[i]);
        }

        return totalSimilarity / statements1.Count;
    }

    /// <summary>
    /// Calculates the similarity between two statements
    /// </summary>
    /// <param name="statement1">First statement</param>
    /// <param name="statement2">Second statement</param>
    /// <returns>Similarity score between 0 and 1</returns>
    private double CalculateStatementSimilarity(StatementSyntax statement1, StatementSyntax statement2)
    {
        if (statement1.GetType() != statement2.GetType())
        {
            return 0;
        }

        var text1 = NormalizeStatement(statement1.ToString());
        var text2 = NormalizeStatement(statement2.ToString());

        // Check for exact match
        if (text1 == text2)
        {
            return 1.0;
        }

        // Calculate string similarity
        return CalculateSimilarity(text1, text2);
    }

    /// <summary>
    /// Generates a refactoring recommendation for duplicated code
    /// </summary>
    /// <param name="statements">Duplicated statements</param>
    /// <param name="method1">First method name</param>
    /// <param name="method2">Second method name</param>
    /// <returns>Refactoring recommendation</returns>
    private string GenerateRefactoringRecommendation(List<StatementSyntax> statements, string method1, string method2)
    {
        if (statements.Count < 3)
        {
            return "No refactoring needed for small duplications.";
        }

        // Check for validation code pattern
        bool hasNullCheck = statements.Any(s => s.ToString().Contains("null"));
        bool hasEmptyCheck = statements.Any(s =>
            s.ToString().Contains(".Count == 0") ||
            s.ToString().Contains(".Length == 0") ||
            s.ToString().Contains(".Any()"));

        if (hasNullCheck && hasEmptyCheck)
        {
            return $"Extract validation logic from {method1} and {method2} into a shared validation method.";
        }

        // Check for loop pattern
        bool hasLoop = statements.Any(s => s is ForEachStatementSyntax || s is ForStatementSyntax || s is WhileStatementSyntax);
        if (hasLoop)
        {
            return $"Extract loop logic from {method1} and {method2} into a shared helper method.";
        }

        // Check for complex calculation
        bool hasCalculation = statements.Count > 5 && statements.Any(s =>
            s.ToString().Contains("=") &&
            (s.ToString().Contains("+") || s.ToString().Contains("-") ||
             s.ToString().Contains("*") || s.ToString().Contains("/")));
        if (hasCalculation)
        {
            return $"Extract calculation logic from {method1} and {method2} into a shared calculation method.";
        }

        // Default recommendation
        return $"Extract duplicated code from {method1} and {method2} into a shared helper method.";
    }

    /// <summary>
    /// Checks if a statement is semantically similar to another statement
    /// </summary>
    /// <param name="statement1">First statement</param>
    /// <param name="statement2">Second statement</param>
    /// <returns>True if the statements are semantically similar</returns>
    private bool AreSemanticallySimilarStatements(StatementSyntax statement1, StatementSyntax statement2)
    {
        if (statement1.GetType() != statement2.GetType())
        {
            return false;
        }

        var text1 = NormalizeStatement(statement1.ToString());
        var text2 = NormalizeStatement(statement2.ToString());

        // Check for exact match first
        if (text1 == text2)
        {
            return true;
        }

        // Check for semantic similarity
        return AreSemanticallySimilar(statement1, statement2, text1, text2);
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

        // Check for exact match
        if (text1 == text2)
        {
            return true;
        }

        // Check for semantic similarity
        return AreSemanticallySimilar(statement1, statement2, text1, text2);
    }

    /// <summary>
    /// Checks if two statements are semantically similar
    /// </summary>
    /// <param name="statement1">First statement</param>
    /// <param name="statement2">Second statement</param>
    /// <param name="normalizedText1">Normalized text of first statement</param>
    /// <param name="normalizedText2">Normalized text of second statement</param>
    /// <returns>True if the statements are semantically similar</returns>
    private bool AreSemanticallySimilar(StatementSyntax statement1, StatementSyntax statement2, string normalizedText1, string normalizedText2)
    {
        // Check for similar structure
        var similarity = CalculateSimilarity(normalizedText1, normalizedText2);
        if (similarity >= 0.8) // 80% similarity threshold
        {
            return true;
        }

        // Check for specific statement types
        if (statement1 is IfStatementSyntax if1 && statement2 is IfStatementSyntax if2)
        {
            // Compare if statements
            return AreSimilarIfStatements(if1, if2);
        }
        else if (statement1 is ForEachStatementSyntax foreach1 && statement2 is ForEachStatementSyntax foreach2)
        {
            // Compare foreach statements
            return AreSimilarForEachStatements(foreach1, foreach2);
        }
        else if (statement1 is ForStatementSyntax for1 && statement2 is ForStatementSyntax for2)
        {
            // Compare for statements
            return AreSimilarForStatements(for1, for2);
        }
        else if (statement1 is ExpressionStatementSyntax expr1 && statement2 is ExpressionStatementSyntax expr2)
        {
            // Compare expression statements
            return AreSimilarExpressionStatements(expr1, expr2);
        }

        return false;
    }

    /// <summary>
    /// Calculates the similarity between two strings
    /// </summary>
    /// <param name="s1">First string</param>
    /// <param name="s2">Second string</param>
    /// <returns>Similarity score between 0 and 1</returns>
    private double CalculateSimilarity(string s1, string s2)
    {
        if (string.IsNullOrEmpty(s1) || string.IsNullOrEmpty(s2))
        {
            return 0;
        }

        // Calculate Levenshtein distance
        int[,] distance = new int[s1.Length + 1, s2.Length + 1];

        for (int i = 0; i <= s1.Length; i++)
        {
            distance[i, 0] = i;
        }

        for (int j = 0; j <= s2.Length; j++)
        {
            distance[0, j] = j;
        }

        for (int i = 1; i <= s1.Length; i++)
        {
            for (int j = 1; j <= s2.Length; j++)
            {
                int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;

                distance[i, j] = Math.Min(
                    Math.Min(distance[i - 1, j] + 1, distance[i, j - 1] + 1),
                    distance[i - 1, j - 1] + cost);
            }
        }

        // Calculate similarity as 1 - (distance / max length)
        return 1.0 - ((double)distance[s1.Length, s2.Length] / Math.Max(s1.Length, s2.Length));
    }

    /// <summary>
    /// Checks if two if statements are similar
    /// </summary>
    /// <param name="if1">First if statement</param>
    /// <param name="if2">Second if statement</param>
    /// <returns>True if the if statements are similar</returns>
    private bool AreSimilarIfStatements(IfStatementSyntax if1, IfStatementSyntax if2)
    {
        // Compare conditions
        var condition1 = NormalizeStatement(if1.Condition.ToString());
        var condition2 = NormalizeStatement(if2.Condition.ToString());

        // Check if the conditions are similar
        if (CalculateSimilarity(condition1, condition2) < 0.7)
        {
            return false;
        }

        // Compare statements in the if block
        if (if1.Statement is BlockSyntax block1 && if2.Statement is BlockSyntax block2)
        {
            // Check if the blocks have similar number of statements
            if (Math.Abs(block1.Statements.Count - block2.Statements.Count) > 1)
            {
                return false;
            }

            // Check if at least 70% of the statements are similar
            int similarStatements = 0;
            int minStatements = Math.Min(block1.Statements.Count, block2.Statements.Count);

            for (int i = 0; i < minStatements; i++)
            {
                if (AreSimilarStatements(block1.Statements[i], block2.Statements[i]))
                {
                    similarStatements++;
                }
            }

            return (double)similarStatements / minStatements >= 0.7;
        }

        return false;
    }

    /// <summary>
    /// Checks if two foreach statements are similar
    /// </summary>
    /// <param name="foreach1">First foreach statement</param>
    /// <param name="foreach2">Second foreach statement</param>
    /// <returns>True if the foreach statements are similar</returns>
    private bool AreSimilarForEachStatements(ForEachStatementSyntax foreach1, ForEachStatementSyntax foreach2)
    {
        // Compare the statement structure
        var expr1 = NormalizeStatement(foreach1.Expression.ToString());
        var expr2 = NormalizeStatement(foreach2.Expression.ToString());

        // Check if the expressions are similar
        if (CalculateSimilarity(expr1, expr2) < 0.7)
        {
            return false;
        }

        // Compare statements in the foreach block
        if (foreach1.Statement is BlockSyntax block1 && foreach2.Statement is BlockSyntax block2)
        {
            // Check if the blocks have similar number of statements
            if (Math.Abs(block1.Statements.Count - block2.Statements.Count) > 1)
            {
                return false;
            }

            // Check if at least 70% of the statements are similar
            int similarStatements = 0;
            int minStatements = Math.Min(block1.Statements.Count, block2.Statements.Count);

            for (int i = 0; i < minStatements; i++)
            {
                if (AreSimilarStatements(block1.Statements[i], block2.Statements[i]))
                {
                    similarStatements++;
                }
            }

            return (double)similarStatements / minStatements >= 0.7;
        }

        return false;
    }

    /// <summary>
    /// Checks if two for statements are similar
    /// </summary>
    /// <param name="for1">First for statement</param>
    /// <param name="for2">Second for statement</param>
    /// <returns>True if the for statements are similar</returns>
    private bool AreSimilarForStatements(ForStatementSyntax for1, ForStatementSyntax for2)
    {
        // Compare the statement structure
        var init1 = NormalizeStatement(for1.Declaration?.ToString() ?? "");
        var init2 = NormalizeStatement(for2.Declaration?.ToString() ?? "");

        var condition1 = NormalizeStatement(for1.Condition?.ToString() ?? "");
        var condition2 = NormalizeStatement(for2.Condition?.ToString() ?? "");

        // Check if the initializers and conditions are similar
        if (CalculateSimilarity(init1, init2) < 0.7 || CalculateSimilarity(condition1, condition2) < 0.7)
        {
            return false;
        }

        // Compare statements in the for block
        if (for1.Statement is BlockSyntax block1 && for2.Statement is BlockSyntax block2)
        {
            // Check if the blocks have similar number of statements
            if (Math.Abs(block1.Statements.Count - block2.Statements.Count) > 1)
            {
                return false;
            }

            // Check if at least 70% of the statements are similar
            int similarStatements = 0;
            int minStatements = Math.Min(block1.Statements.Count, block2.Statements.Count);

            for (int i = 0; i < minStatements; i++)
            {
                if (AreSimilarStatements(block1.Statements[i], block2.Statements[i]))
                {
                    similarStatements++;
                }
            }

            return (double)similarStatements / minStatements >= 0.7;
        }

        return false;
    }

    /// <summary>
    /// Checks if two expression statements are similar
    /// </summary>
    /// <param name="expr1">First expression statement</param>
    /// <param name="expr2">Second expression statement</param>
    /// <returns>True if the expression statements are similar</returns>
    private bool AreSimilarExpressionStatements(ExpressionStatementSyntax expr1, ExpressionStatementSyntax expr2)
    {
        // Compare the expressions
        var normalized1 = NormalizeStatement(expr1.Expression.ToString());
        var normalized2 = NormalizeStatement(expr2.Expression.ToString());

        return CalculateSimilarity(normalized1, normalized2) >= 0.8;
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

        // Replace method calls with placeholders
        statement = Regex.Replace(statement, @"VAR\s*\([^\)]*\)", "METHOD_CALL");

        // Replace property access with placeholders
        statement = Regex.Replace(statement, @"VAR\.VAR", "PROPERTY_ACCESS");

        // Replace if statements with placeholders
        statement = Regex.Replace(statement, @"if\s*\([^\)]*\)", "IF_CONDITION");

        // Replace for loops with placeholders
        statement = Regex.Replace(statement, @"for\s*\([^\)]*\)", "FOR_LOOP");

        // Replace foreach loops with placeholders
        statement = Regex.Replace(statement, @"foreach\s*\([^\)]*\)", "FOREACH_LOOP");

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
                sb.AppendLine($"    Similarity: {block.SimilarityType} ({block.SimilarityPercentage:F2}%)");
                sb.AppendLine($"    Recommendation: {block.RefactoringRecommendation}");

                // Show a snippet of the duplicated code
                if (!string.IsNullOrEmpty(block.DuplicatedCode))
                {
                    sb.AppendLine("    Code Snippet:");
                    var codeLines = block.DuplicatedCode.Split('\n');
                    var previewLines = codeLines.Length > 5 ? codeLines.Take(5).ToList() : codeLines.ToList();
                    foreach (var line in previewLines)
                    {
                        sb.AppendLine($"      {line}");
                    }
                    if (codeLines.Length > 5)
                    {
                        sb.AppendLine("      ...");
                    }
                }

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
        sb.AppendLine("    h1, h2, h3, h4 { color: #333; }");
        sb.AppendLine("    .low { color: green; }");
        sb.AppendLine("    .moderate { color: orange; }");
        sb.AppendLine("    .high { color: red; }");
        sb.AppendLine("    .veryhigh { color: darkred; }");
        sb.AppendLine("    .exact { color: #ff6b6b; font-weight: bold; }");
        sb.AppendLine("    .semantic { color: #feca57; font-weight: bold; }");
        sb.AppendLine("    .block { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }");
        sb.AppendLine("    .metrics { display: flex; flex-wrap: wrap; }");
        sb.AppendLine("    .metric { flex: 1; min-width: 200px; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }");
        sb.AppendLine("    .progress-container { width: 100%; background-color: #e0e0e0; border-radius: 5px; margin: 5px 0; }");
        sb.AppendLine("    .progress-bar { height: 20px; border-radius: 5px; text-align: center; color: white; font-weight: bold; }");
        sb.AppendLine("    .progress-low { background-color: #1dd1a1; }");
        sb.AppendLine("    .progress-moderate { background-color: #feca57; }");
        sb.AppendLine("    .progress-high { background-color: #ff6b6b; }");
        sb.AppendLine("    .recommendation { background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; }");
        sb.AppendLine("    .code { background-color: #f1f1f1; padding: 10px; border-radius: 5px; font-family: monospace; white-space: pre; overflow-x: auto; margin: 10px 0; }");
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

        // Add progress bar for duplication percentage
        string progressClass = "progress-low";
        if (DuplicationPercentage >= 20)
        {
            progressClass = "progress-high";
        }
        else if (DuplicationPercentage >= 10)
        {
            progressClass = "progress-moderate";
        }

        sb.AppendLine("      <div class=\"progress-container\">");
        sb.AppendLine($"        <div class=\"progress-bar {progressClass}\" style=\"width: {Math.Min(DuplicationPercentage, 100)}%\">{DuplicationPercentage:F2}%</div>");
        sb.AppendLine("      </div>");
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
                string levelClass = "low";
                if (block.LineCount >= 10)
                {
                    levelClass = "high";
                }
                else if (block.LineCount >= 5)
                {
                    levelClass = "moderate";
                }

                sb.AppendLine($"  <div class=\"block {levelClass}\">");
                sb.AppendLine($"    <h3>{block.SourceMethod} and {block.TargetMethod}</h3>");
                sb.AppendLine($"    <p>Source: Lines {block.SourceStartLine}-{block.SourceEndLine}</p>");
                sb.AppendLine($"    <p>Target: Lines {block.TargetStartLine}-{block.TargetEndLine}</p>");
                sb.AppendLine($"    <p>Line Count: {block.LineCount}</p>");

                // Add similarity percentage with progress bar
                sb.AppendLine($"    <p>Similarity: <span class=\"{block.SimilarityType.ToString().ToLower()}\">{block.SimilarityType}</span> ({block.SimilarityPercentage:F2}%)</p>");
                sb.AppendLine("    <div class=\"progress-container\">");
                string similarityClass = block.SimilarityType == SimilarityType.Exact ? "progress-high" : "progress-moderate";
                sb.AppendLine($"      <div class=\"progress-bar {similarityClass}\" style=\"width: {block.SimilarityPercentage}%\">{block.SimilarityPercentage:F2}%</div>");
                sb.AppendLine("    </div>");

                // Add refactoring recommendation
                if (!string.IsNullOrEmpty(block.RefactoringRecommendation))
                {
                    sb.AppendLine("    <div class=\"recommendation\">");
                    sb.AppendLine("      <h4>Refactoring Recommendation:</h4>");
                    sb.AppendLine($"      <p>{block.RefactoringRecommendation}</p>");
                    sb.AppendLine("    </div>");
                }

                // Add code snippet
                if (!string.IsNullOrEmpty(block.DuplicatedCode))
                {
                    sb.AppendLine("    <h4>Code Snippet:</h4>");
                    sb.AppendLine("    <div class=\"code\">");
                    var codeLines = block.DuplicatedCode.Split('\n');
                    var previewLines = codeLines.Length > 5 ? codeLines.Take(5).ToList() : codeLines.ToList();
                    foreach (var line in previewLines)
                    {
                        sb.AppendLine($"      {System.Web.HttpUtility.HtmlEncode(line)}");
                    }
                    if (codeLines.Length > 5)
                    {
                        sb.AppendLine("      ...");
                    }
                    sb.AppendLine("    </div>");
                }

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

    /// <summary>
    /// Gets or sets the similarity percentage (0-100)
    /// </summary>
    public double SimilarityPercentage { get; set; } = 100.0;

    /// <summary>
    /// Gets or sets the refactoring recommendation
    /// </summary>
    public string RefactoringRecommendation { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the duplicated code
    /// </summary>
    public string DuplicatedCode { get; set; } = string.Empty;
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
