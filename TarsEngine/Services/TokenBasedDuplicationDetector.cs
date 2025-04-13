using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;

namespace TarsEngine.Services;

/// <summary>
/// Detector for token-based code duplication
/// </summary>
public class TokenBasedDuplicationDetector
{
    private readonly ILogger _logger;

    // Minimum sequence length to consider as duplication (in tokens)
    private readonly int _minimumDuplicateTokens;

    // Minimum sequence length to consider as duplication (in lines)
    private readonly int _minimumDuplicateLines;

    /// <summary>
    /// Initializes a new instance of the <see cref="TokenBasedDuplicationDetector"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    /// <param name="minimumDuplicateTokens">Minimum number of tokens to consider as duplication</param>
    /// <param name="minimumDuplicateLines">Minimum number of lines to consider as duplication</param>
    public TokenBasedDuplicationDetector(ILogger logger, int minimumDuplicateTokens = 100, int minimumDuplicateLines = 5)
    {
        _logger = logger;
        _minimumDuplicateTokens = minimumDuplicateTokens;
        _minimumDuplicateLines = minimumDuplicateLines;
    }

    /// <summary>
    /// Detects duplicated code in a single file
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <param name="sourceCode">Source code</param>
    /// <returns>List of duplicated blocks</returns>
    public List<DuplicatedBlock> DetectDuplication(string filePath, string sourceCode)
    {
        try
        {
            // Tokenize the source code
            var tokens = TokenizeCode(sourceCode);

            // Find duplicated token sequences
            var duplicatedSequences = FindDuplicatedSequences(tokens, _minimumDuplicateTokens);

            // Convert token sequences to line ranges
            var duplicatedBlocks = ConvertToLineRanges(filePath, sourceCode, duplicatedSequences);

            return duplicatedBlocks;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting duplication in file {FilePath}", filePath);
            return new List<DuplicatedBlock>();
        }
    }

    /// <summary>
    /// Detects duplicated code across multiple files
    /// </summary>
    /// <param name="files">Dictionary of file paths and source code</param>
    /// <returns>List of duplicated blocks</returns>
    public List<DuplicatedBlock> DetectDuplicationAcrossFiles(Dictionary<string, string> files)
    {
        try
        {
            var allDuplicatedBlocks = new List<DuplicatedBlock>();

            // First, detect duplication within each file
            foreach (var file in files)
            {
                var duplicatedBlocks = DetectDuplication(file.Key, file.Value);
                allDuplicatedBlocks.AddRange(duplicatedBlocks);
            }

            // Then, detect duplication across files
            var fileTokens = new Dictionary<string, List<SyntaxToken>>();
            foreach (var file in files)
            {
                fileTokens[file.Key] = TokenizeCode(file.Value);
            }

            // Compare each file with every other file
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

                    // Find duplicated token sequences between these two files
                    var duplicatedSequences = FindDuplicatedSequencesBetweenFiles(
                        fileTokens[file1.Key], fileTokens[file2.Key], _minimumDuplicateTokens);

                    // Convert token sequences to line ranges
                    foreach (var sequence in duplicatedSequences)
                    {
                        var sourceStartLine = GetLineNumber(file1.Value, sequence.Item1);
                        var sourceEndLine = GetLineNumber(file1.Value, sequence.Item2);
                        var targetStartLine = GetLineNumber(file2.Value, sequence.Item3);
                        var targetEndLine = GetLineNumber(file2.Value, sequence.Item4);

                        // Skip if the duplicated block is too small
                        if (sourceEndLine - sourceStartLine + 1 < _minimumDuplicateLines ||
                            targetEndLine - targetStartLine + 1 < _minimumDuplicateLines)
                        {
                            continue;
                        }

                        // Get the duplicated code
                        var duplicatedCode = GetCodeBetweenLines(file1.Value, sourceStartLine, sourceEndLine);

                        // Calculate similarity percentage (100% for token-based duplication)
                        var similarityPercentage = 100.0;

                        // Create a duplicated block
                        var duplicatedBlock = new DuplicatedBlock
                        {
                            SourceFilePath = file1.Key,
                            SourceStartLine = sourceStartLine,
                            SourceEndLine = sourceEndLine,
                            TargetFilePath = file2.Key,
                            TargetStartLine = targetStartLine,
                            TargetEndLine = targetEndLine,
                            DuplicatedCode = duplicatedCode,
                            SimilarityPercentage = similarityPercentage
                        };

                        allDuplicatedBlocks.Add(duplicatedBlock);
                    }
                }
            }

            return allDuplicatedBlocks;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting duplication across files");
            return new List<DuplicatedBlock>();
        }
    }

    /// <summary>
    /// Tokenizes C# code
    /// </summary>
    /// <param name="sourceCode">Source code</param>
    /// <returns>List of tokens</returns>
    private List<SyntaxToken> TokenizeCode(string sourceCode)
    {
        // Parse the source code
        var syntaxTree = CSharpSyntaxTree.ParseText(sourceCode);
        var root = syntaxTree.GetRoot();

        // Get all tokens
        var tokens = root.DescendantTokens()
            .Where(t => !t.IsKind(SyntaxKind.EndOfFileToken))
            .Where(t => !t.IsKind(SyntaxKind.WhitespaceTrivia))
            .Where(t => !t.IsKind(SyntaxKind.EndOfLineTrivia))
            .Where(t => !t.IsKind(SyntaxKind.SingleLineCommentTrivia))
            .Where(t => !t.IsKind(SyntaxKind.MultiLineCommentTrivia))
            .Where(t => !t.IsKind(SyntaxKind.DocumentationCommentExteriorTrivia))
            .Where(t => !t.IsKind(SyntaxKind.XmlCommentStartToken))
            .Where(t => !t.IsKind(SyntaxKind.XmlCommentEndToken))
            .ToList();

        return tokens;
    }

    /// <summary>
    /// Finds duplicated token sequences in a single file
    /// </summary>
    /// <param name="tokens">List of tokens</param>
    /// <param name="minimumLength">Minimum sequence length</param>
    /// <returns>List of duplicated sequences (start position, end position)</returns>
    private List<(int, int)> FindDuplicatedSequences(List<SyntaxToken> tokens, int minimumLength)
    {
        var duplicatedSequences = new List<(int, int)>();

        // Use a suffix array-based approach to find duplicated sequences
        // This is a simplified implementation for demonstration purposes

        // For each possible starting position
        for (int i = 0; i < tokens.Count - minimumLength; i++)
        {
            // For each possible comparison starting position
            for (int j = i + 1; j < tokens.Count - minimumLength; j++)
            {
                // Count how many tokens match
                int k = 0;
                while (k < minimumLength && j + k < tokens.Count && TokensEqual(tokens[i + k], tokens[j + k]))
                {
                    k++;
                }

                // If we found a duplicated sequence of sufficient length
                if (k >= minimumLength)
                {
                    duplicatedSequences.Add((i, i + k - 1));

                    // Skip ahead to avoid overlapping sequences
                    j += k - 1;
                }
            }
        }

        return duplicatedSequences;
    }

    /// <summary>
    /// Finds duplicated token sequences between two files
    /// </summary>
    /// <param name="tokens1">Tokens from first file</param>
    /// <param name="tokens2">Tokens from second file</param>
    /// <param name="minimumLength">Minimum sequence length</param>
    /// <returns>List of duplicated sequences (start1, end1, start2, end2)</returns>
    private List<(int, int, int, int)> FindDuplicatedSequencesBetweenFiles(
        List<SyntaxToken> tokens1, List<SyntaxToken> tokens2, int minimumLength)
    {
        var duplicatedSequences = new List<(int, int, int, int)>();

        // For each possible starting position in the first file
        for (int i = 0; i < tokens1.Count - minimumLength; i++)
        {
            // For each possible starting position in the second file
            for (int j = 0; j < tokens2.Count - minimumLength; j++)
            {
                // Count how many tokens match
                int k = 0;
                while (k < minimumLength && i + k < tokens1.Count && j + k < tokens2.Count &&
                       TokensEqual(tokens1[i + k], tokens2[j + k]))
                {
                    k++;
                }

                // If we found a duplicated sequence of sufficient length
                if (k >= minimumLength)
                {
                    duplicatedSequences.Add((i, i + k - 1, j, j + k - 1));

                    // Skip ahead to avoid overlapping sequences
                    j += k - 1;
                }
            }
        }

        return duplicatedSequences;
    }

    /// <summary>
    /// Converts token sequences to line ranges
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <param name="sourceCode">Source code</param>
    /// <param name="sequences">Token sequences</param>
    /// <returns>List of duplicated blocks</returns>
    private List<DuplicatedBlock> ConvertToLineRanges(string filePath, string sourceCode, List<(int, int)> sequences)
    {
        var duplicatedBlocks = new List<DuplicatedBlock>();

        foreach (var sequence in sequences)
        {
            var startLine = GetLineNumber(sourceCode, sequence.Item1);
            var endLine = GetLineNumber(sourceCode, sequence.Item2);

            // Skip if the duplicated block is too small
            if (endLine - startLine + 1 < _minimumDuplicateLines)
            {
                continue;
            }

            // Get the duplicated code
            var duplicatedCode = GetCodeBetweenLines(sourceCode, startLine, endLine);

            // Create a duplicated block
            var duplicatedBlock = new DuplicatedBlock
            {
                SourceFilePath = filePath,
                SourceStartLine = startLine,
                SourceEndLine = endLine,
                TargetFilePath = filePath,
                TargetStartLine = startLine,
                TargetEndLine = endLine,
                DuplicatedCode = duplicatedCode,
                SimilarityPercentage = 100.0 // 100% for token-based duplication
            };

            duplicatedBlocks.Add(duplicatedBlock);
        }

        return duplicatedBlocks;
    }

    /// <summary>
    /// Gets the line number for a token position
    /// </summary>
    /// <param name="sourceCode">Source code</param>
    /// <param name="tokenPosition">Token position</param>
    /// <returns>Line number</returns>
    private int GetLineNumber(string sourceCode, int tokenPosition)
    {
        // This is a simplified implementation
        // In a real implementation, we would use the token's position in the source code

        // For now, just return a placeholder value
        return 1;
    }

    /// <summary>
    /// Gets the code between two line numbers
    /// </summary>
    /// <param name="sourceCode">Source code</param>
    /// <param name="startLine">Start line</param>
    /// <param name="endLine">End line</param>
    /// <returns>Code between the lines</returns>
    private string GetCodeBetweenLines(string sourceCode, int startLine, int endLine)
    {
        // This is a simplified implementation
        // In a real implementation, we would extract the code between the specified lines

        // For now, just return a placeholder value
        return "Duplicated code";
    }

    /// <summary>
    /// Checks if two tokens are equal
    /// </summary>
    /// <param name="token1">First token</param>
    /// <param name="token2">Second token</param>
    /// <returns>True if the tokens are equal</returns>
    private bool TokensEqual(SyntaxToken token1, SyntaxToken token2)
    {
        // For token-based duplication, we consider tokens equal if they have the same kind and text
        return token1.Kind() == token2.Kind() && token1.Text == token2.Text;
    }
}
