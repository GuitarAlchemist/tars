# Script to update the EnhancedTreeOfThoughtCommand.cs file

$content = @"
using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.CodeAnalysis;
using TarsEngine.Services.TreeOfThought;

namespace TarsCli.Commands
{
    /// <summary>
    /// Command for running the enhanced Tree-of-Thought auto-improvement pipeline.
    /// </summary>
    public class EnhancedTreeOfThoughtCommand : Command
    {
        private readonly ILogger<EnhancedTreeOfThoughtCommand> _logger;
        private readonly EnhancedTreeOfThoughtService _enhancedTreeOfThoughtService;
        private readonly PatternDetector _patternDetector;
        private readonly CodeTransformer _codeTransformer;
        private readonly AnalysisReportGenerator _analysisReportGenerator;

        /// <summary>
        /// Initializes a new instance of the <see cref="EnhancedTreeOfThoughtCommand"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="enhancedTreeOfThoughtService">The enhanced Tree-of-Thought service.</param>
        /// <param name="patternDetector">The pattern detector.</param>
        /// <param name="codeTransformer">The code transformer.</param>
        /// <param name="analysisReportGenerator">The analysis report generator.</param>
        public EnhancedTreeOfThoughtCommand(
            ILogger<EnhancedTreeOfThoughtCommand> logger,
            EnhancedTreeOfThoughtService enhancedTreeOfThoughtService,
            PatternDetector patternDetector,
            CodeTransformer codeTransformer,
            AnalysisReportGenerator analysisReportGenerator)
            : base("enhanced-tot", "Run the enhanced Tree-of-Thought auto-improvement pipeline")
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _enhancedTreeOfThoughtService = enhancedTreeOfThoughtService ?? throw new ArgumentNullException(nameof(enhancedTreeOfThoughtService));
            _patternDetector = patternDetector ?? throw new ArgumentNullException(nameof(patternDetector));
            _codeTransformer = codeTransformer ?? throw new ArgumentNullException(nameof(codeTransformer));
            _analysisReportGenerator = analysisReportGenerator ?? throw new ArgumentNullException(nameof(analysisReportGenerator));

            // Add options
            var fileOption = new Option<string>(
                aliases: new[] { "--file", "-f" },
                description: "The file to improve");
            fileOption.IsRequired = true;

            var typeOption = new Option<string>(
                aliases: new[] { "--type", "-t" },
                description: "The type of improvement to make (performance, maintainability, error_handling, security)",
                getDefaultValue: () => "performance");

            var outputOption = new Option<string>(
                aliases: new[] { "--output", "-o" },
                description: "The output file for the report",
                getDefaultValue: () => "enhanced_tot_report.md");

            var formatOption = new Option<string>(
                aliases: new[] { "--format", "-fmt" },
                description: "The format of the report (markdown, html, json)",
                getDefaultValue: () => "markdown");

            // Add options to the command
            AddOption(fileOption);
            AddOption(typeOption);
            AddOption(outputOption);
            AddOption(formatOption);

            // Set the handler
            Handler = CommandHandler.Create<string, string, string, string>(HandleCommandAsync);
        }

        private async Task<int> HandleCommandAsync(string file, string type, string output, string format)
        {
            try
            {
                _logger.LogInformation("Running enhanced Tree-of-Thought auto-improvement pipeline");
                _logger.LogInformation("File: {File}", file);
                _logger.LogInformation("Improvement type: {Type}", type);
                _logger.LogInformation("Output: {Output}", output);
                _logger.LogInformation("Format: {Format}", format);

                // Check if the file exists
                if (!File.Exists(file))
                {
                    _logger.LogError("File does not exist: {File}", file);
                    return 1;
                }

                // Read the file content
                var code = await File.ReadAllTextAsync(file);

                // Step 1: Analyze the code
                _logger.LogInformation("Step 1: Analyzing the code");
                var analysisReport = _analysisReportGenerator.GenerateReport(code, file);
                _logger.LogInformation("Analysis completed: {PatternCount} patterns detected", analysisReport.DetectedPatterns.Count);

                // Step 2: Transform the code
                _logger.LogInformation("Step 2: Transforming the code");
                var transformationTypes = new List<string>();
                switch (type.ToLower())
                {
                    case "performance":
                        transformationTypes.Add("Performance");
                        break;
                    case "maintainability":
                        transformationTypes.Add("Maintainability");
                        break;
                    case "error_handling":
                        transformationTypes.Add("Error Handling");
                        break;
                    case "security":
                        transformationTypes.Add("Security");
                        break;
                    case "all":
                        transformationTypes.Add("Performance");
                        transformationTypes.Add("Maintainability");
                        transformationTypes.Add("Error Handling");
                        transformationTypes.Add("Security");
                        break;
                    default:
                        _logger.LogWarning("Unknown improvement type: {Type}, defaulting to Performance", type);
                        transformationTypes.Add("Performance");
                        break;
                }

                var transformationResult = _codeTransformer.TransformCode(code, transformationTypes);
                _logger.LogInformation("Transformation completed: {TransformationCount} transformations applied", transformationResult.AppliedTransformations.Count);

                // Step 3: Save the transformed code
                var fileInfo = new FileInfo(file);
                var transformedFilePath = Path.Combine(Path.GetDirectoryName(file), $"Improved_{fileInfo.Name}");
                await File.WriteAllTextAsync(transformedFilePath, transformationResult.TransformedCode);
                _logger.LogInformation("Transformed code saved to {TransformedFilePath}", transformedFilePath);

                // Step 4: Generate the report
                _logger.LogInformation("Step 4: Generating the report");
                string report;
                switch (format.ToLower())
                {
                    case "html":
                        report = _analysisReportGenerator.GenerateHtmlReport(analysisReport);
                        break;
                    case "json":
                        report = _analysisReportGenerator.GenerateJsonReport(analysisReport);
                        break;
                    case "markdown":
                    default:
                        report = _analysisReportGenerator.GenerateMarkdownReport(analysisReport);
                        break;
                }

                // Add transformation results to the report
                report += "\n\n## Transformation Results\n\n";
                report += $"**Original Code:** {file}\n\n";
                report += $"**Transformed Code:** {transformedFilePath}\n\n";
                report += "### Applied Transformations\n\n";
                foreach (var transformation in transformationResult.AppliedTransformations)
                {
                    report += $"- {transformation}\n";
                }

                if (transformationResult.Errors.Count > 0)
                {
                    report += "\n### Transformation Errors\n\n";
                    foreach (var error in transformationResult.Errors)
                    {
                        report += $"- {error}\n";
                    }
                }

                // Save the report
                await File.WriteAllTextAsync(output, report);
                _logger.LogInformation("Report saved to {Output}", output);

                _logger.LogInformation("Enhanced Tree-of-Thought auto-improvement pipeline completed successfully");
                return 0;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running enhanced Tree-of-Thought auto-improvement pipeline");
                return 1;
            }
        }
    }
}
"@

Set-Content -Path "TarsCli\Commands\EnhancedTreeOfThoughtCommand.cs" -Value $content
