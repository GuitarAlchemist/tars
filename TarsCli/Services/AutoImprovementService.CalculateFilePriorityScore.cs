using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsCli.Models;

namespace TarsCli.Services
{
    // Partial class containing the file prioritization methods
    public partial class AutoImprovementService
    {
        /// <summary>
        /// Calculate a priority score for a file
        /// </summary>
        /// <param name="filePath">The path of the file</param>
        /// <returns>A priority score</returns>
        private FilePriorityScore CalculateFilePriorityScore(string filePath)
        {
            var score = new FilePriorityScore(filePath);
            var fileInfo = new FileInfo(filePath);

            // Base score based on file type
            double baseScore = 0;
            var extension = Path.GetExtension(filePath).ToLower();
            switch (extension)
            {
                case ".md":
                    baseScore = 1.0;
                    break;
                case ".cs":
                    baseScore = 2.0;
                    break;
                case ".fs":
                    baseScore = 2.5; // Prefer F# files as they're the core engine
                    break;
                default:
                    baseScore = 0.5;
                    break;
            }
            score.BaseScore = baseScore;
            score.AddFactor("FileType", baseScore);

            // Content score based on file size and content analysis
            double contentScore = 0;
            if (fileInfo.Exists && fileInfo.Length > 0)
            {
                // Size factor (normalized to 0-1 range)
                var sizeFactor = Math.Min(1.0, fileInfo.Length / 10000.0); // Cap at 10KB
                contentScore += sizeFactor;
                score.AddFactor("FileSize", sizeFactor);

                // Content analysis
                try
                {
                    var content = File.ReadAllText(filePath);
                    var contentAnalysisScore = AnalyzeFileContent(content, extension);
                    contentScore += contentAnalysisScore;
                    score.AddFactor("ContentAnalysis", contentAnalysisScore);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, $"Error analyzing content of file: {filePath}");
                }
            }
            score.ContentScore = contentScore;

            // Recency score based on last modified time
            double recencyScore = 0;
            if (fileInfo.Exists)
            {
                // More recent files get higher priority
                var ageInDays = (DateTime.Now - fileInfo.LastWriteTime).TotalDays;
                recencyScore = Math.Max(0, 1.0 - (ageInDays / 30.0)); // Normalize to 0-1 range over 30 days
                score.AddFactor("Recency", recencyScore);
            }
            score.RecencyScore = recencyScore;

            // Complexity score based on content complexity
            double complexityScore = 0;
            if (fileInfo.Exists && fileInfo.Length > 0)
            {
                try
                {
                    var content = File.ReadAllText(filePath);
                    var complexityAnalysisScore = AnalyzeFileComplexity(content, extension);
                    complexityScore = complexityAnalysisScore;
                    score.AddFactor("Complexity", complexityScore);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, $"Error analyzing complexity of file: {filePath}");
                }
            }
            score.ComplexityScore = complexityScore;

            // Improvement potential score based on previous analysis
            double improvementPotentialScore = 0;
            var previousImprovements = _state.ImprovementHistory
                .Where(i => i.FilePath == filePath)
                .ToList();

            if (previousImprovements.Any())
            {
                // Files that have been improved before might need more improvements
                var lastImprovement = previousImprovements.OrderByDescending(i => i.Timestamp).First();
                var daysSinceLastImprovement = (DateTime.Now - lastImprovement.Timestamp).TotalDays;

                // If it was improved recently, lower the priority
                if (daysSinceLastImprovement < 7)
                {
                    improvementPotentialScore = -1.0;
                }
                // If it was improved a while ago, increase the priority
                else if (daysSinceLastImprovement > 30)
                {
                    improvementPotentialScore = 0.5;
                }

                score.AddFactor("PreviousImprovements", improvementPotentialScore);
            }
            else
            {
                // Files that have never been improved get a boost
                improvementPotentialScore = 1.0;
                score.AddFactor("NeverImproved", improvementPotentialScore);
            }
            score.ImprovementPotentialScore = improvementPotentialScore;

            return score;
        }

        /// <summary>
        /// Analyze the content of a file
        /// </summary>
        /// <param name="content">The content of the file</param>
        /// <param name="extension">The extension of the file</param>
        /// <returns>A score based on the content analysis</returns>
        private double AnalyzeFileContent(string content, string extension)
        {
            double score = 0;

            // Check for TODOs, FIXMEs, and other improvement indicators
            var todoCount = Regex.Matches(content, @"TODO|FIXME|HACK|XXX|BUG", RegexOptions.IgnoreCase).Count;
            if (todoCount > 0)
            {
                score += Math.Min(1.0, todoCount / 5.0); // Cap at 5 TODOs
            }

            // Check for code smells in C# and F# files
            if (extension == ".cs" || extension == ".fs")
            {
                // Long methods
                var longMethodCount = Regex.Matches(content, @"(public|private|protected|internal)\s+\w+\s+\w+\s*\([^)]*\)\s*\{[^}]{1000,}\}", RegexOptions.Singleline).Count;
                if (longMethodCount > 0)
                {
                    score += Math.Min(1.0, longMethodCount / 3.0); // Cap at 3 long methods
                }

                // Commented-out code
                var commentedCodeCount = Regex.Matches(content, @"//\s*[a-zA-Z0-9_]+\s*\([^)]*\)").Count;
                if (commentedCodeCount > 0)
                {
                    score += Math.Min(0.5, commentedCodeCount / 5.0); // Cap at 5 commented-out code blocks
                }

                // Magic numbers
                var magicNumberCount = Regex.Matches(content, @"[^\w\.][0-9]{2,}[^\w\.]|[^\w\.][0][xX][0-9a-fA-F]{2,}[^\w\.]").Count;
                if (magicNumberCount > 0)
                {
                    score += Math.Min(0.5, magicNumberCount / 10.0); // Cap at 10 magic numbers
                }
            }

            // Check for documentation quality in markdown files
            if (extension == ".md")
            {
                // Check for headings
                var headingCount = Regex.Matches(content, @"^#+\s+.+$", RegexOptions.Multiline).Count;
                if (headingCount < 3)
                {
                    score += 0.5; // Few headings might indicate poor structure
                }

                // Check for code blocks
                var codeBlockCount = Regex.Matches(content, @"```[\s\S]*?```", RegexOptions.Multiline).Count;
                if (codeBlockCount > 0)
                {
                    score += Math.Min(0.5, codeBlockCount / 5.0); // Cap at 5 code blocks
                }

                // Check for TODOs in markdown
                var markdownTodoCount = Regex.Matches(content, @"\bTODO\b|\bFIXME\b", RegexOptions.IgnoreCase).Count;
                if (markdownTodoCount > 0)
                {
                    score += Math.Min(1.0, markdownTodoCount / 3.0); // Cap at 3 TODOs
                }
            }

            return score;
        }

        /// <summary>
        /// Analyze the complexity of a file
        /// </summary>
        /// <param name="content">The content of the file</param>
        /// <param name="extension">The extension of the file</param>
        /// <returns>A score based on the complexity analysis</returns>
        private double AnalyzeFileComplexity(string content, string extension)
        {
            double score = 0;

            // Simple complexity metrics
            var lineCount = content.Split('\n').Length;
            var normalizedLineCount = Math.Min(1.0, lineCount / 500.0); // Cap at 500 lines
            score += normalizedLineCount;

            // Check for nested control structures in code files
            if (extension == ".cs" || extension == ".fs")
            {
                // Count nested if statements, loops, etc.
                var nestedControlStructures = Regex.Matches(content, @"if\s*\([^)]*\)\s*\{[^{}]*if\s*\(", RegexOptions.Singleline).Count;
                nestedControlStructures += Regex.Matches(content, @"for\s*\([^)]*\)\s*\{[^{}]*for\s*\(", RegexOptions.Singleline).Count;
                nestedControlStructures += Regex.Matches(content, @"while\s*\([^)]*\)\s*\{[^{}]*while\s*\(", RegexOptions.Singleline).Count;

                if (nestedControlStructures > 0)
                {
                    score += Math.Min(1.0, nestedControlStructures / 5.0); // Cap at 5 nested structures
                }

                // Count method parameters
                var methodsWithManyParams = Regex.Matches(content, @"\w+\s*\([^)]{50,}\)", RegexOptions.Singleline).Count;
                if (methodsWithManyParams > 0)
                {
                    score += Math.Min(0.5, methodsWithManyParams / 3.0); // Cap at 3 methods with many params
                }
            }

            return score;
        }
    }
}
