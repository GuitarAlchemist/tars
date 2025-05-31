using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.TreeOfThought;

namespace TarsCli.Commands
{
    /// <summary>
    /// Command for working with metascripts using Tree-of-Thought reasoning.
    /// </summary>
    public class MetascriptTreeOfThoughtCommand : Command
    {
        private readonly ILogger<MetascriptTreeOfThoughtCommand> _logger;
        private readonly MetascriptTreeOfThoughtService _metascriptTreeOfThoughtService;

        /// <summary>
        /// Initializes a new instance of the <see cref="MetascriptTreeOfThoughtCommand"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="metascriptTreeOfThoughtService">The metascript Tree-of-Thought service.</param>
        public MetascriptTreeOfThoughtCommand(ILogger<MetascriptTreeOfThoughtCommand> logger, MetascriptTreeOfThoughtService metascriptTreeOfThoughtService)
            : base("metascript-tot", "Work with metascripts using Tree-of-Thought reasoning")
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _metascriptTreeOfThoughtService = metascriptTreeOfThoughtService ?? throw new ArgumentNullException(nameof(metascriptTreeOfThoughtService));

            // Add subcommands
            AddCommand(CreateGenerateCommand());
            AddCommand(CreateValidateCommand());
            AddCommand(CreateExecuteCommand());
            AddCommand(CreateAnalyzeCommand());
            AddCommand(CreatePipelineCommand());
        }

        private Command CreateGenerateCommand()
        {
            var command = new Command("generate", "Generate a metascript using Tree-of-Thought reasoning");

            var templateOption = new Option<string>(
                aliases: new[] { "--template", "-t" },
                description: "The template file to use");
            templateOption.IsRequired = true;

            var valuesOption = new Option<string>(
                aliases: new[] { "--values", "-v" },
                description: "The values file to use (JSON format)");
            valuesOption.IsRequired = true;

            var outputOption = new Option<string>(
                aliases: new[] { "--output", "-o" },
                description: "The output file for the generated metascript",
                getDefaultValue: () => "generated_metascript.tars");

            var thoughtTreeOption = new Option<string>(
                aliases: new[] { "--thought-tree", "-tt" },
                description: "The output file for the thought tree (JSON format)",
                getDefaultValue: () => "thought_tree.json");

            command.AddOption(templateOption);
            command.AddOption(valuesOption);
            command.AddOption(outputOption);
            command.AddOption(thoughtTreeOption);

            command.Handler = CommandHandler.Create<string, string, string, string>(HandleGenerateCommandAsync);

            return command;
        }

        private Command CreateValidateCommand()
        {
            var command = new Command("validate", "Validate a metascript using Tree-of-Thought reasoning");

            var metascriptOption = new Option<string>(
                aliases: new[] { "--metascript", "-m" },
                description: "The metascript file to validate");
            metascriptOption.IsRequired = true;

            var reportOption = new Option<string>(
                aliases: new[] { "--report", "-r" },
                description: "The output file for the validation report",
                getDefaultValue: () => "validation_report.md");

            var thoughtTreeOption = new Option<string>(
                aliases: new[] { "--thought-tree", "-tt" },
                description: "The output file for the thought tree (JSON format)",
                getDefaultValue: () => "thought_tree.json");

            command.AddOption(metascriptOption);
            command.AddOption(reportOption);
            command.AddOption(thoughtTreeOption);

            command.Handler = CommandHandler.Create<string, string, string>(HandleValidateCommandAsync);

            return command;
        }

        private Command CreateExecuteCommand()
        {
            var command = new Command("execute", "Execute a metascript using Tree-of-Thought reasoning");

            var metascriptOption = new Option<string>(
                aliases: new[] { "--metascript", "-m" },
                description: "The metascript file to execute");
            metascriptOption.IsRequired = true;

            var outputOption = new Option<string>(
                aliases: new[] { "--output", "-o" },
                description: "The output file for the execution output",
                getDefaultValue: () => "execution_output.txt");

            var reportOption = new Option<string>(
                aliases: new[] { "--report", "-r" },
                description: "The output file for the execution report",
                getDefaultValue: () => "execution_report.md");

            var thoughtTreeOption = new Option<string>(
                aliases: new[] { "--thought-tree", "-tt" },
                description: "The output file for the thought tree (JSON format)",
                getDefaultValue: () => "thought_tree.json");

            command.AddOption(metascriptOption);
            command.AddOption(outputOption);
            command.AddOption(reportOption);
            command.AddOption(thoughtTreeOption);

            command.Handler = CommandHandler.Create<string, string, string, string>(HandleExecuteCommandAsync);

            return command;
        }

        private Command CreateAnalyzeCommand()
        {
            var command = new Command("analyze", "Analyze the results of a metascript execution using Tree-of-Thought reasoning");

            var outputOption = new Option<string>(
                aliases: new[] { "--output", "-o" },
                description: "The execution output file to analyze");
            outputOption.IsRequired = true;

            var executionTimeOption = new Option<int>(
                aliases: new[] { "--execution-time", "-et" },
                description: "The execution time in milliseconds",
                getDefaultValue: () => 1000);

            var memoryUsageOption = new Option<int>(
                aliases: new[] { "--memory-usage", "-mu" },
                description: "The peak memory usage in megabytes",
                getDefaultValue: () => 100);

            var errorCountOption = new Option<int>(
                aliases: new[] { "--error-count", "-ec" },
                description: "The number of errors encountered",
                getDefaultValue: () => 0);

            var reportOption = new Option<string>(
                aliases: new[] { "--report", "-r" },
                description: "The output file for the analysis report",
                getDefaultValue: () => "analysis_report.md");

            var thoughtTreeOption = new Option<string>(
                aliases: new[] { "--thought-tree", "-tt" },
                description: "The output file for the thought tree (JSON format)",
                getDefaultValue: () => "thought_tree.json");

            command.AddOption(outputOption);
            command.AddOption(executionTimeOption);
            command.AddOption(memoryUsageOption);
            command.AddOption(errorCountOption);
            command.AddOption(reportOption);
            command.AddOption(thoughtTreeOption);

            command.Handler = CommandHandler.Create<string, int, int, int, string, string>(HandleAnalyzeCommandAsync);

            return command;
        }

        private Command CreatePipelineCommand()
        {
            var command = new Command("pipeline", "Run the complete metascript pipeline using Tree-of-Thought reasoning");

            var templateOption = new Option<string>(
                aliases: new[] { "--template", "-t" },
                description: "The template file to use");
            templateOption.IsRequired = true;

            var valuesOption = new Option<string>(
                aliases: new[] { "--values", "-v" },
                description: "The values file to use (JSON format)");
            valuesOption.IsRequired = true;

            var outputDirOption = new Option<string>(
                aliases: new[] { "--output-dir", "-od" },
                description: "The output directory for all files",
                getDefaultValue: () => "pipeline_output");

            command.AddOption(templateOption);
            command.AddOption(valuesOption);
            command.AddOption(outputDirOption);

            command.Handler = CommandHandler.Create<string, string, string>(HandlePipelineCommandAsync);

            return command;
        }

        private async Task<int> HandleGenerateCommandAsync(string template, string values, string output, string thoughtTree)
        {
            try
            {
                _logger.LogInformation("Generating metascript using Tree-of-Thought reasoning");
                _logger.LogInformation("Template: {Template}", template);
                _logger.LogInformation("Values: {Values}", values);
                _logger.LogInformation("Output: {Output}", output);
                _logger.LogInformation("Thought tree: {ThoughtTree}", thoughtTree);

                // Read the template file
                if (!File.Exists(template))
                {
                    _logger.LogError("Template file not found: {Template}", template);
                    return 1;
                }

                var templateContent = await File.ReadAllTextAsync(template);

                // Read the values file
                if (!File.Exists(values))
                {
                    _logger.LogError("Values file not found: {Values}", values);
                    return 1;
                }

                var valuesContent = await File.ReadAllTextAsync(values);
                var templateValues = ParseValuesFile(valuesContent);

                // Generate the metascript
                var (metascript, thoughtTreeJson) = await _metascriptTreeOfThoughtService.GenerateMetascriptAsync(templateContent, templateValues);

                // Save the metascript
                await File.WriteAllTextAsync(output, metascript);
                _logger.LogInformation("Metascript saved to {Output}", output);

                // Save the thought tree
                await File.WriteAllTextAsync(thoughtTree, thoughtTreeJson);
                _logger.LogInformation("Thought tree saved to {ThoughtTree}", thoughtTree);

                _logger.LogInformation("Metascript generation completed successfully");
                return 0;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating metascript");
                return 1;
            }
        }

        private async Task<int> HandleValidateCommandAsync(string metascript, string report, string thoughtTree)
        {
            try
            {
                _logger.LogInformation("Validating metascript using Tree-of-Thought reasoning");
                _logger.LogInformation("Metascript: {Metascript}", metascript);
                _logger.LogInformation("Report: {Report}", report);
                _logger.LogInformation("Thought tree: {ThoughtTree}", thoughtTree);

                // Read the metascript file
                if (!File.Exists(metascript))
                {
                    _logger.LogError("Metascript file not found: {Metascript}", metascript);
                    return 1;
                }

                var metascriptContent = await File.ReadAllTextAsync(metascript);

                // Validate the metascript
                var (isValid, errors, warnings, thoughtTreeJson) = await _metascriptTreeOfThoughtService.ValidateMetascriptAsync(metascriptContent);

                // Generate the report
                var reportContent = GenerateValidationReport(metascript, isValid, errors, warnings);

                // Save the report
                await File.WriteAllTextAsync(report, reportContent);
                _logger.LogInformation("Validation report saved to {Report}", report);

                // Save the thought tree
                await File.WriteAllTextAsync(thoughtTree, thoughtTreeJson);
                _logger.LogInformation("Thought tree saved to {ThoughtTree}", thoughtTree);

                _logger.LogInformation("Metascript validation completed successfully");
                return isValid ? 0 : 1;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error validating metascript");
                return 1;
            }
        }

        private async Task<int> HandleExecuteCommandAsync(string metascript, string output, string report, string thoughtTree)
        {
            try
            {
                _logger.LogInformation("Executing metascript using Tree-of-Thought reasoning");
                _logger.LogInformation("Metascript: {Metascript}", metascript);
                _logger.LogInformation("Output: {Output}", output);
                _logger.LogInformation("Report: {Report}", report);
                _logger.LogInformation("Thought tree: {ThoughtTree}", thoughtTree);

                // Read the metascript file
                if (!File.Exists(metascript))
                {
                    _logger.LogError("Metascript file not found: {Metascript}", metascript);
                    return 1;
                }

                var metascriptContent = await File.ReadAllTextAsync(metascript);

                // Execute the metascript
                var (outputContent, success, reportContent, thoughtTreeJson) = await _metascriptTreeOfThoughtService.ExecuteMetascriptAsync(metascriptContent);

                // Save the output
                await File.WriteAllTextAsync(output, outputContent);
                _logger.LogInformation("Execution output saved to {Output}", output);

                // Save the report
                await File.WriteAllTextAsync(report, reportContent);
                _logger.LogInformation("Execution report saved to {Report}", report);

                // Save the thought tree
                await File.WriteAllTextAsync(thoughtTree, thoughtTreeJson);
                _logger.LogInformation("Thought tree saved to {ThoughtTree}", thoughtTree);

                _logger.LogInformation("Metascript execution completed successfully");
                return success ? 0 : 1;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error executing metascript");
                return 1;
            }
        }

        private async Task<int> HandleAnalyzeCommandAsync(string output, int executionTime, int memoryUsage, int errorCount, string report, string thoughtTree)
        {
            try
            {
                _logger.LogInformation("Analyzing metascript results using Tree-of-Thought reasoning");
                _logger.LogInformation("Output: {Output}", output);
                _logger.LogInformation("Execution time: {ExecutionTime} ms", executionTime);
                _logger.LogInformation("Memory usage: {MemoryUsage} MB", memoryUsage);
                _logger.LogInformation("Error count: {ErrorCount}", errorCount);
                _logger.LogInformation("Report: {Report}", report);
                _logger.LogInformation("Thought tree: {ThoughtTree}", thoughtTree);

                // Read the output file
                if (!File.Exists(output))
                {
                    _logger.LogError("Output file not found: {Output}", output);
                    return 1;
                }

                var outputContent = await File.ReadAllTextAsync(output);

                // Analyze the results
                var (success, errors, warnings, impact, recommendations, thoughtTreeJson) = 
                    await _metascriptTreeOfThoughtService.AnalyzeResultsAsync(outputContent, executionTime, memoryUsage, errorCount);

                // Generate the report
                var reportContent = GenerateAnalysisReport(output, success, errors, warnings, impact, recommendations);

                // Save the report
                await File.WriteAllTextAsync(report, reportContent);
                _logger.LogInformation("Analysis report saved to {Report}", report);

                // Save the thought tree
                await File.WriteAllTextAsync(thoughtTree, thoughtTreeJson);
                _logger.LogInformation("Thought tree saved to {ThoughtTree}", thoughtTree);

                _logger.LogInformation("Metascript results analysis completed successfully");
                return success ? 0 : 1;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error analyzing metascript results");
                return 1;
            }
        }

        private async Task<int> HandlePipelineCommandAsync(string template, string values, string outputDir)
        {
            try
            {
                _logger.LogInformation("Running metascript pipeline using Tree-of-Thought reasoning");
                _logger.LogInformation("Template: {Template}", template);
                _logger.LogInformation("Values: {Values}", values);
                _logger.LogInformation("Output directory: {OutputDir}", outputDir);

                // Create the output directory
                Directory.CreateDirectory(outputDir);

                // Read the template file
                if (!File.Exists(template))
                {
                    _logger.LogError("Template file not found: {Template}", template);
                    return 1;
                }

                var templateContent = await File.ReadAllTextAsync(template);

                // Read the values file
                if (!File.Exists(values))
                {
                    _logger.LogError("Values file not found: {Values}", values);
                    return 1;
                }

                var valuesContent = await File.ReadAllTextAsync(values);
                var templateValues = ParseValuesFile(valuesContent);

                // Step 1: Generate the metascript
                _logger.LogInformation("Step 1: Generating metascript");
                var metascriptPath = Path.Combine(outputDir, "generated_metascript.tars");
                var generateThoughtTreePath = Path.Combine(outputDir, "generate_thought_tree.json");
                
                var (metascript, generateThoughtTreeJson) = await _metascriptTreeOfThoughtService.GenerateMetascriptAsync(templateContent, templateValues);
                
                await File.WriteAllTextAsync(metascriptPath, metascript);
                await File.WriteAllTextAsync(generateThoughtTreePath, generateThoughtTreeJson);
                
                _logger.LogInformation("Metascript generated and saved to {MetascriptPath}", metascriptPath);
                _logger.LogInformation("Generate thought tree saved to {ThoughtTreePath}", generateThoughtTreePath);

                // Step 2: Validate the metascript
                _logger.LogInformation("Step 2: Validating metascript");
                var validationReportPath = Path.Combine(outputDir, "validation_report.md");
                var validateThoughtTreePath = Path.Combine(outputDir, "validate_thought_tree.json");
                
                var (isValid, errors, warnings, validateThoughtTreeJson) = await _metascriptTreeOfThoughtService.ValidateMetascriptAsync(metascript);
                
                var validationReport = GenerateValidationReport(metascriptPath, isValid, errors, warnings);
                await File.WriteAllTextAsync(validationReportPath, validationReport);
                await File.WriteAllTextAsync(validateThoughtTreePath, validateThoughtTreeJson);
                
                _logger.LogInformation("Validation report saved to {ReportPath}", validationReportPath);
                _logger.LogInformation("Validate thought tree saved to {ThoughtTreePath}", validateThoughtTreePath);
                
                if (!isValid)
                {
                    _logger.LogError("Metascript validation failed with {ErrorCount} errors", errors.Count);
                    return 1;
                }

                // Step 3: Execute the metascript
                _logger.LogInformation("Step 3: Executing metascript");
                var outputPath = Path.Combine(outputDir, "execution_output.txt");
                var executionReportPath = Path.Combine(outputDir, "execution_report.md");
                var executeThoughtTreePath = Path.Combine(outputDir, "execute_thought_tree.json");
                
                var (outputContent, success, executionReport, executeThoughtTreeJson) = await _metascriptTreeOfThoughtService.ExecuteMetascriptAsync(metascript);
                
                await File.WriteAllTextAsync(outputPath, outputContent);
                await File.WriteAllTextAsync(executionReportPath, executionReport);
                await File.WriteAllTextAsync(executeThoughtTreePath, executeThoughtTreeJson);
                
                _logger.LogInformation("Execution output saved to {OutputPath}", outputPath);
                _logger.LogInformation("Execution report saved to {ReportPath}", executionReportPath);
                _logger.LogInformation("Execute thought tree saved to {ThoughtTreePath}", executeThoughtTreePath);
                
                if (!success)
                {
                    _logger.LogError("Metascript execution failed");
                    return 1;
                }

                // Step 4: Analyze the results
                _logger.LogInformation("Step 4: Analyzing results");
                var analysisReportPath = Path.Combine(outputDir, "analysis_report.md");
                var analyzeThoughtTreePath = Path.Combine(outputDir, "analyze_thought_tree.json");
                
                // Use some dummy metrics for demonstration
                var executionTime = 1000; // 1 second
                var memoryUsage = 100; // 100 MB
                var errorCount = 0;
                
                var (analysisSuccess, analysisErrors, analysisWarnings, impact, recommendations, analyzeThoughtTreeJson) = 
                    await _metascriptTreeOfThoughtService.AnalyzeResultsAsync(outputContent, executionTime, memoryUsage, errorCount);
                
                var analysisReport = GenerateAnalysisReport(outputPath, analysisSuccess, analysisErrors, analysisWarnings, impact, recommendations);
                await File.WriteAllTextAsync(analysisReportPath, analysisReport);
                await File.WriteAllTextAsync(analyzeThoughtTreePath, analyzeThoughtTreeJson);
                
                _logger.LogInformation("Analysis report saved to {ReportPath}", analysisReportPath);
                _logger.LogInformation("Analyze thought tree saved to {ThoughtTreePath}", analyzeThoughtTreePath);

                // Step 5: Integrate the results
                _logger.LogInformation("Step 5: Integrating results");
                var integrationReportPath = Path.Combine(outputDir, "integration_report.md");
                var integrateThoughtTreePath = Path.Combine(outputDir, "integrate_thought_tree.json");
                
                var (integrated, message, integrateThoughtTreeJson) = 
                    await _metascriptTreeOfThoughtService.IntegrateResultsAsync(outputContent, analysisSuccess, recommendations);
                
                var integrationReport = $"# Metascript Integration Report\n\n## Integration Status\n\n{(integrated ? "✅ Integrated" : "❌ Not Integrated")}\n\n## Message\n\n{message}\n\n## Recommendations\n\n{string.Join("\n", recommendations.ConvertAll(r => $"- {r}"))}";
                await File.WriteAllTextAsync(integrationReportPath, integrationReport);
                await File.WriteAllTextAsync(integrateThoughtTreePath, integrateThoughtTreeJson);
                
                _logger.LogInformation("Integration report saved to {ReportPath}", integrationReportPath);
                _logger.LogInformation("Integrate thought tree saved to {ThoughtTreePath}", integrateThoughtTreePath);
                
                if (!integrated)
                {
                    _logger.LogError("Metascript integration failed: {Message}", message);
                    return 1;
                }

                // Create a summary report
                _logger.LogInformation("Creating summary report");
                var summaryReportPath = Path.Combine(outputDir, "summary_report.md");
                
                var summaryReport = $@"# Metascript Pipeline Summary Report

## Overview

- **Template**: {template}
- **Values**: {values}
- **Output Directory**: {outputDir}

## Pipeline Steps

### 1. Generation

- **Status**: ✅ Completed
- **Output**: [Generated Metascript]({metascriptPath})
- **Thought Tree**: [Generate Thought Tree]({generateThoughtTreePath})

### 2. Validation

- **Status**: {(isValid ? "✅ Valid" : "❌ Invalid")}
- **Errors**: {errors.Count}
- **Warnings**: {warnings.Count}
- **Report**: [Validation Report]({validationReportPath})
- **Thought Tree**: [Validate Thought Tree]({validateThoughtTreePath})

### 3. Execution

- **Status**: {(success ? "✅ Successful" : "❌ Failed")}
- **Output**: [Execution Output]({outputPath})
- **Report**: [Execution Report]({executionReportPath})
- **Thought Tree**: [Execute Thought Tree]({executeThoughtTreePath})

### 4. Analysis

- **Status**: {(analysisSuccess ? "✅ Successful" : "❌ Failed")}
- **Errors**: {analysisErrors.Count}
- **Warnings**: {analysisWarnings.Count}
- **Impact**: {impact}
- **Report**: [Analysis Report]({analysisReportPath})
- **Thought Tree**: [Analyze Thought Tree]({analyzeThoughtTreePath})

### 5. Integration

- **Status**: {(integrated ? "✅ Integrated" : "❌ Not Integrated")}
- **Message**: {message}
- **Report**: [Integration Report]({integrationReportPath})
- **Thought Tree**: [Integrate Thought Tree]({integrateThoughtTreePath})

## Recommendations

{string.Join("\n", recommendations.ConvertAll(r => $"- {r}"))}

## Conclusion

The metascript pipeline {(integrated ? "completed successfully" : "failed")} with {errors.Count + analysisErrors.Count} errors and {warnings.Count + analysisWarnings.Count} warnings.
";
                
                await File.WriteAllTextAsync(summaryReportPath, summaryReport);
                _logger.LogInformation("Summary report saved to {ReportPath}", summaryReportPath);

                _logger.LogInformation("Metascript pipeline completed successfully");
                return 0;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running metascript pipeline");
                return 1;
            }
        }

        private Dictionary<string, string> ParseValuesFile(string content)
        {
            // In a real implementation, this would parse JSON
            // For simplicity, we'll use a simple format: key=value
            var values = new Dictionary<string, string>();
            
            var lines = content.Split('\n');
            foreach (var line in lines)
            {
                var trimmedLine = line.Trim();
                if (string.IsNullOrWhiteSpace(trimmedLine) || trimmedLine.StartsWith("#"))
                {
                    continue;
                }
                
                var parts = trimmedLine.Split('=', 2);
                if (parts.Length == 2)
                {
                    values[parts[0].Trim()] = parts[1].Trim();
                }
            }
            
            return values;
        }

        private string GenerateValidationReport(string metascriptPath, bool isValid, List<string> errors, List<string> warnings)
        {
            var report = new System.Text.StringBuilder();
            
            report.AppendLine("# Metascript Validation Report");
            report.AppendLine();
            
            report.AppendLine("## Overview");
            report.AppendLine();
            report.AppendLine($"- **Metascript**: {metascriptPath}");
            report.AppendLine($"- **Valid**: {(isValid ? "✅ Yes" : "❌ No")}");
            report.AppendLine($"- **Errors**: {errors.Count}");
            report.AppendLine($"- **Warnings**: {warnings.Count}");
            report.AppendLine();
            
            if (errors.Count > 0)
            {
                report.AppendLine("## Errors");
                report.AppendLine();
                foreach (var error in errors)
                {
                    report.AppendLine($"- {error}");
                }
                report.AppendLine();
            }
            
            if (warnings.Count > 0)
            {
                report.AppendLine("## Warnings");
                report.AppendLine();
                foreach (var warning in warnings)
                {
                    report.AppendLine($"- {warning}");
                }
                report.AppendLine();
            }
            
            report.AppendLine("## Conclusion");
            report.AppendLine();
            if (isValid)
            {
                report.AppendLine("The metascript is valid and can be executed.");
            }
            else
            {
                report.AppendLine("The metascript is invalid and needs to be fixed before execution.");
            }
            
            return report.ToString();
        }

        private string GenerateAnalysisReport(string outputPath, bool success, List<string> errors, List<string> warnings, string impact, List<string> recommendations)
        {
            var report = new System.Text.StringBuilder();
            
            report.AppendLine("# Metascript Results Analysis Report");
            report.AppendLine();
            
            report.AppendLine("## Overview");
            report.AppendLine();
            report.AppendLine($"- **Output**: {outputPath}");
            report.AppendLine($"- **Success**: {(success ? "✅ Yes" : "❌ No")}");
            report.AppendLine($"- **Errors**: {errors.Count}");
            report.AppendLine($"- **Warnings**: {warnings.Count}");
            report.AppendLine($"- **Impact**: {impact}");
            report.AppendLine();
            
            if (errors.Count > 0)
            {
                report.AppendLine("## Errors");
                report.AppendLine();
                foreach (var error in errors)
                {
                    report.AppendLine($"- {error}");
                }
                report.AppendLine();
            }
            
            if (warnings.Count > 0)
            {
                report.AppendLine("## Warnings");
                report.AppendLine();
                foreach (var warning in warnings)
                {
                    report.AppendLine($"- {warning}");
                }
                report.AppendLine();
            }
            
            if (recommendations.Count > 0)
            {
                report.AppendLine("## Recommendations");
                report.AppendLine();
                foreach (var recommendation in recommendations)
                {
                    report.AppendLine($"- {recommendation}");
                }
                report.AppendLine();
            }
            
            report.AppendLine("## Conclusion");
            report.AppendLine();
            report.AppendLine(impact);
            
            return report.ToString();
        }
    }
}
