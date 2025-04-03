using System;
using System.CommandLine;
using System.IO;
using System.Threading.Tasks;
using TarsCli.Services;

namespace TarsCli.Commands
{
    /// <summary>
    /// Command for using the F# code analysis and metascript engine
    /// </summary>
    public class FSharpAnalysisCommand(FSharpIntegrationService fsharpService)
    {
        /// <summary>
        /// Registers the F# analysis commands
        /// </summary>
        public Command RegisterCommands()
        {
            var fsharpCommand = new Command("fsharp", "F# code analysis and metascript commands");

            // Add analyze command
            var analyzeCommand = new Command("analyze", "Analyze code using F# analysis engine");
            var analyzePath = new Argument<string>("path", "Path to the file or project to analyze");
            var analyzeMaxFiles = new Option<int>(new[] { "--max-files", "-m" }, () => 50, "Maximum number of files to analyze");

            analyzeCommand.AddArgument(analyzePath);
            analyzeCommand.AddOption(analyzeMaxFiles);
            analyzeCommand.SetHandler(AnalyzeAsync, analyzePath, analyzeMaxFiles);
            fsharpCommand.AddCommand(analyzeCommand);

            // Add metascript command
            var metascriptCommand = new Command("metascript", "Apply metascript rules to code");
            var metascriptPath = new Argument<string>("path", "Path to the file or project to transform");
            var metascriptRules = new Option<string>(new[] { "--rules", "-r" }, "Path to the metascript rules file");
            var metascriptOutput = new Option<string>(new[] { "--output", "-o" }, "Output directory for transformed files");
            var metascriptMaxFiles = new Option<int>(new[] { "--max-files", "-m" }, () => 50, "Maximum number of files to transform");

            metascriptCommand.AddArgument(metascriptPath);
            metascriptCommand.AddOption(metascriptRules);
            metascriptCommand.AddOption(metascriptOutput);
            metascriptCommand.AddOption(metascriptMaxFiles);
            metascriptCommand.SetHandler(ApplyMetascriptAsync, metascriptPath, metascriptRules, metascriptOutput, metascriptMaxFiles);
            fsharpCommand.AddCommand(metascriptCommand);

            return fsharpCommand;
        }

        /// <summary>
        /// Analyzes code using the F# analysis engine
        /// </summary>
        private async Task<int> AnalyzeAsync(string path, int maxFiles)
        {
            try
            {
                Console.WriteLine($"Analyzing {path} using F# analysis engine...");

                if (File.Exists(path))
                {
                    // Analyze a single file
                    var result = await fsharpService.AnalyzeFileAsync(path);
                    DisplayAnalysisResult(result);
                }
                else if (Directory.Exists(path))
                {
                    // Analyze a project
                    var results = await fsharpService.AnalyzeProjectAsync(path, maxFiles);
                    Console.WriteLine($"Analyzed {results.Count} files");

                    foreach (var result in results)
                    {
                        DisplayAnalysisResult(result);
                    }
                }
                else
                {
                    Console.WriteLine($"Path not found: {path}");
                    return 1;
                }

                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error analyzing code: {ex.Message}");
                return 1;
            }
        }

        /// <summary>
        /// Applies metascript rules to code
        /// </summary>
        private async Task<int> ApplyMetascriptAsync(string path, string rulesPath, string outputPath, int maxFiles)
        {
            try
            {
                Console.WriteLine($"Applying metascript rules from {rulesPath} to {path}...");

                if (!File.Exists(rulesPath))
                {
                    Console.WriteLine($"Rules file not found: {rulesPath}");
                    return 1;
                }

                // Create output directory if it doesn't exist
                if (!string.IsNullOrEmpty(outputPath) && !Directory.Exists(outputPath))
                {
                    Directory.CreateDirectory(outputPath);
                }

                if (File.Exists(path))
                {
                    // Apply rules to a single file
                    var transformedCode = await fsharpService.ApplyMetascriptRulesToFileAsync(path, rulesPath);
                    
                    if (!string.IsNullOrEmpty(transformedCode))
                    {
                        // Save the transformed code
                        string outputFilePath = string.IsNullOrEmpty(outputPath)
                            ? Path.Combine(Path.GetDirectoryName(path), $"{Path.GetFileNameWithoutExtension(path)}_transformed{Path.GetExtension(path)}")
                            : Path.Combine(outputPath, Path.GetFileName(path));
                        
                        File.WriteAllText(outputFilePath, transformedCode);
                        Console.WriteLine($"Transformed code saved to: {outputFilePath}");
                    }
                }
                else if (Directory.Exists(path))
                {
                    // Apply rules to a project
                    var results = await fsharpService.ApplyMetascriptRulesToProjectAsync(path, rulesPath, maxFiles);
                    Console.WriteLine($"Transformed {results.Count} files");

                    // Save the transformed files
                    foreach (var result in results)
                    {
                        string relativePath = result.Key.Substring(path.Length).TrimStart('\\', '/');
                        string outputFilePath = string.IsNullOrEmpty(outputPath)
                            ? Path.Combine(Path.GetDirectoryName(result.Key), $"{Path.GetFileNameWithoutExtension(result.Key)}_transformed{Path.GetExtension(result.Key)}")
                            : Path.Combine(outputPath, relativePath);
                        
                        // Create directory if it doesn't exist
                        Directory.CreateDirectory(Path.GetDirectoryName(outputFilePath));
                        
                        File.WriteAllText(outputFilePath, result.Value);
                        Console.WriteLine($"Transformed: {result.Key} -> {outputFilePath}");
                    }
                }
                else
                {
                    Console.WriteLine($"Path not found: {path}");
                    return 1;
                }

                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error applying metascript rules: {ex.Message}");
                return 1;
            }
        }

        /// <summary>
        /// Displays the analysis result
        /// </summary>
        private void DisplayAnalysisResult(AnalysisResult result)
        {
            Console.WriteLine($"\nFile: {result.FilePath}");
            
            if (result.Issues.Count == 0)
            {
                Console.WriteLine("  No issues found");
                return;
            }
            
            Console.WriteLine($"  Found {result.Issues.Count} issues:");
            
            foreach (var issue in result.Issues)
            {
                Console.WriteLine($"  - [{issue.Type}] {issue.Description}");
                Console.WriteLine($"    Location: {issue.Location}");
                
                if (!string.IsNullOrEmpty(issue.Suggestion))
                {
                    Console.WriteLine($"    Suggestion: {issue.Suggestion}");
                }
            }
            
            if (result.SuggestedFixes.Count > 0)
            {
                Console.WriteLine($"  {result.SuggestedFixes.Count} suggested fixes available");
            }
        }
    }
}
