using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TarsEngine.Intelligence.Measurement;
using TarsEngine.Models.Metrics;
using TarsCli.Services;

namespace TarsCli.Commands
{
    /// <summary>
    /// Command for running benchmarks
    /// </summary>
    public class BenchmarkCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        /// <summary>
        /// Initializes a new instance of the <see cref="BenchmarkCommand"/> class
        /// </summary>
        /// <param name="serviceProvider">The service provider</param>
        public BenchmarkCommand(IServiceProvider serviceProvider) : base("benchmark", "Run benchmarks and establish baseline measurements")
        {
            _serviceProvider = serviceProvider;

            // Add subcommands
            AddCommand(new RunCodebaseBenchmarkCommand(_serviceProvider));
            AddCommand(new ShowBaselineThresholdsCommand(_serviceProvider));
            AddCommand(new SetBaselineThresholdCommand(_serviceProvider));
            AddCommand(new CompareMetricsCommand(_serviceProvider));
        }

        /// <summary>
        /// Command for running codebase benchmarks
        /// </summary>
        private class RunCodebaseBenchmarkCommand : Command
        {
            private readonly IServiceProvider _serviceProvider;

            /// <summary>
            /// Initializes a new instance of the <see cref="RunCodebaseBenchmarkCommand"/> class
            /// </summary>
            /// <param name="serviceProvider">The service provider</param>
            public RunCodebaseBenchmarkCommand(IServiceProvider serviceProvider) : base("run", "Run benchmarks on the codebase")
            {
                _serviceProvider = serviceProvider;

                // Add options
                var pathOption = new Option<string>(
                    "--path",
                    description: "Path to the codebase",
                    getDefaultValue: () => Directory.GetCurrentDirectory());

                var outputOption = new Option<string>(
                    "--output",
                    description: "Path to save the benchmark results",
                    getDefaultValue: () => "benchmark_results.json");

                var verboseOption = new Option<bool>(
                    "--verbose",
                    description: "Show detailed output",
                    getDefaultValue: () => false);

                AddOption(pathOption);
                AddOption(outputOption);
                AddOption(verboseOption);

                this.SetHandler((InvocationContext context) =>
                {
                    var path = context.ParseResult.GetValueForOption(pathOption);
                    var output = context.ParseResult.GetValueForOption(outputOption);
                    var verbose = context.ParseResult.GetValueForOption(verboseOption);

                    var logger = _serviceProvider.GetRequiredService<ILogger<BenchmarkCommand>>();
                    var benchmarkSystem = _serviceProvider.GetRequiredService<BenchmarkSystem>();
                    var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                    try
                    {
                        consoleService.WriteHeader("Running Codebase Benchmarks");
                        consoleService.WriteLine($"Codebase path: {path}");
                        consoleService.WriteLine($"Output path: {output}");
                        consoleService.WriteLine();

                        // Initialize the benchmark system
                        var initTask = benchmarkSystem.InitializeAsync();
                        initTask.Wait();

                        // Run the benchmarks
                        consoleService.WriteLine("Running benchmarks...");
                        var benchmarkTask = benchmarkSystem.RunCodebaseBenchmarksAsync(path);
                        benchmarkTask.Wait();
                        var results = benchmarkTask.Result;

                        // Display the results
                        consoleService.WriteLine();
                        consoleService.WriteSuccess($"Benchmark completed in {results.Duration.TotalSeconds:F2} seconds");
                        consoleService.WriteLine($"Total metrics collected: {results.TotalMetricsCollected}");
                        consoleService.WriteLine();

                        // Display complexity statistics
                        consoleService.WriteHeader("Complexity Statistics");
                        foreach (var (target, stats) in results.ComplexityStatistics)
                        {
                            consoleService.WriteLine($"Target: {target}");
                            consoleService.WriteLine($"  Count: {stats.Count}");
                            consoleService.WriteLine($"  Min: {stats.MinValue:F2}");
                            consoleService.WriteLine($"  Max: {stats.MaxValue:F2}");
                            consoleService.WriteLine($"  Average: {stats.AverageValue:F2}");
                            consoleService.WriteLine($"  Median: {stats.MedianValue:F2}");
                            consoleService.WriteLine($"  Standard Deviation: {stats.StandardDeviation:F2}");

                            if (verbose)
                            {
                                consoleService.WriteLine("  Percentiles:");
                                foreach (var (percentile, value) in stats.Percentiles)
                                {
                                    consoleService.WriteLine($"    {percentile}th: {value:F2}");
                                }
                            }

                            consoleService.WriteLine();
                        }

                        // Display maintainability statistics
                        consoleService.WriteHeader("Maintainability Statistics");
                        foreach (var (target, stats) in results.MaintainabilityStatistics)
                        {
                            consoleService.WriteLine($"Target: {target}");
                            consoleService.WriteLine($"  Count: {stats.Count}");
                            consoleService.WriteLine($"  Min: {stats.MinValue:F2}");
                            consoleService.WriteLine($"  Max: {stats.MaxValue:F2}");
                            consoleService.WriteLine($"  Average: {stats.AverageValue:F2}");
                            consoleService.WriteLine($"  Median: {stats.MedianValue:F2}");
                            consoleService.WriteLine($"  Standard Deviation: {stats.StandardDeviation:F2}");

                            if (verbose)
                            {
                                consoleService.WriteLine("  Percentiles:");
                                foreach (var (percentile, value) in stats.Percentiles)
                                {
                                    consoleService.WriteLine($"    {percentile}th: {value:F2}");
                                }
                            }

                            consoleService.WriteLine();
                        }

                        // Display Halstead statistics
                        consoleService.WriteHeader("Halstead Statistics");
                        foreach (var (target, stats) in results.HalsteadStatistics)
                        {
                            consoleService.WriteLine($"Target: {target}");
                            consoleService.WriteLine($"  Count: {stats.Count}");
                            consoleService.WriteLine($"  Min: {stats.MinValue:F2}");
                            consoleService.WriteLine($"  Max: {stats.MaxValue:F2}");
                            consoleService.WriteLine($"  Average: {stats.AverageValue:F2}");
                            consoleService.WriteLine($"  Median: {stats.MedianValue:F2}");
                            consoleService.WriteLine($"  Standard Deviation: {stats.StandardDeviation:F2}");

                            if (verbose)
                            {
                                consoleService.WriteLine("  Percentiles:");
                                foreach (var (percentile, value) in stats.Percentiles)
                                {
                                    consoleService.WriteLine($"    {percentile}th: {value:F2}");
                                }
                            }

                            consoleService.WriteLine();
                        }

                        // Save the results to a file
                        var json = System.Text.Json.JsonSerializer.Serialize(results, new System.Text.Json.JsonSerializerOptions
                        {
                            WriteIndented = true
                        });

                        File.WriteAllText(output, json);
                        consoleService.WriteSuccess($"Benchmark results saved to {output}");
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, "Error running benchmarks");
                        consoleService.WriteError($"Error running benchmarks: {ex.Message}");
                    }
                });
            }
        }

        /// <summary>
        /// Command for showing baseline thresholds
        /// </summary>
        private class ShowBaselineThresholdsCommand : Command
        {
            private readonly IServiceProvider _serviceProvider;

            /// <summary>
            /// Initializes a new instance of the <see cref="ShowBaselineThresholdsCommand"/> class
            /// </summary>
            /// <param name="serviceProvider">The service provider</param>
            public ShowBaselineThresholdsCommand(IServiceProvider serviceProvider) : base("show-thresholds", "Show baseline thresholds")
            {
                _serviceProvider = serviceProvider;

                // Add options
                var metricTypeOption = new Option<string>(
                    "--metric-type",
                    description: "Metric type (Complexity, Maintainability, Halstead)",
                    getDefaultValue: () => "");

                AddOption(metricTypeOption);

                this.SetHandler((InvocationContext context) =>
                {
                    var metricType = context.ParseResult.GetValueForOption(metricTypeOption);

                    var logger = _serviceProvider.GetRequiredService<ILogger<BenchmarkCommand>>();
                    var benchmarkSystem = _serviceProvider.GetRequiredService<BenchmarkSystem>();
                    var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                    try
                    {
                        // Initialize the benchmark system
                        var initTask = benchmarkSystem.InitializeAsync();
                        initTask.Wait();

                        if (string.IsNullOrEmpty(metricType))
                        {
                            // Show all thresholds
                            consoleService.WriteHeader("Complexity Thresholds");
                            var complexityThresholds = benchmarkSystem.GetBaselineThresholds("Complexity");
                            foreach (var (target, threshold) in complexityThresholds)
                            {
                                consoleService.WriteLine($"{target}: {threshold:F2}");
                            }

                            consoleService.WriteLine();
                            consoleService.WriteHeader("Maintainability Thresholds");
                            var maintainabilityThresholds = benchmarkSystem.GetBaselineThresholds("Maintainability");
                            foreach (var (target, threshold) in maintainabilityThresholds)
                            {
                                consoleService.WriteLine($"{target}: {threshold:F2}");
                            }

                            consoleService.WriteLine();
                            consoleService.WriteHeader("Halstead Thresholds");
                            var halsteadThresholds = benchmarkSystem.GetBaselineThresholds("Halstead");
                            foreach (var (target, threshold) in halsteadThresholds)
                            {
                                consoleService.WriteLine($"{target}: {threshold:F2}");
                            }
                        }
                        else
                        {
                            // Show thresholds for a specific metric type
                            consoleService.WriteHeader($"{metricType} Thresholds");
                            var thresholds = benchmarkSystem.GetBaselineThresholds(metricType);
                            foreach (var (target, threshold) in thresholds)
                            {
                                consoleService.WriteLine($"{target}: {threshold:F2}");
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, "Error showing baseline thresholds");
                        consoleService.WriteError($"Error showing baseline thresholds: {ex.Message}");
                    }
                });
            }
        }

        /// <summary>
        /// Command for setting a baseline threshold
        /// </summary>
        private class SetBaselineThresholdCommand : Command
        {
            private readonly IServiceProvider _serviceProvider;

            /// <summary>
            /// Initializes a new instance of the <see cref="SetBaselineThresholdCommand"/> class
            /// </summary>
            /// <param name="serviceProvider">The service provider</param>
            public SetBaselineThresholdCommand(IServiceProvider serviceProvider) : base("set-threshold", "Set a baseline threshold")
            {
                _serviceProvider = serviceProvider;

                // Add options
                var metricTypeOption = new Option<string>(
                    "--metric-type",
                    description: "Metric type (Complexity, Maintainability, Halstead)")
                {
                    IsRequired = true
                };

                var targetOption = new Option<string>(
                    "--target",
                    description: "Target (Method, Class, File)")
                {
                    IsRequired = true
                };

                var thresholdOption = new Option<double>(
                    "--threshold",
                    description: "Threshold value")
                {
                    IsRequired = true
                };

                AddOption(metricTypeOption);
                AddOption(targetOption);
                AddOption(thresholdOption);

                this.SetHandler((InvocationContext context) =>
                {
                    var metricType = context.ParseResult.GetValueForOption(metricTypeOption);
                    var target = context.ParseResult.GetValueForOption(targetOption);
                    var threshold = context.ParseResult.GetValueForOption(thresholdOption);

                    var logger = _serviceProvider.GetRequiredService<ILogger<BenchmarkCommand>>();
                    var benchmarkSystem = _serviceProvider.GetRequiredService<BenchmarkSystem>();
                    var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                    try
                    {
                        // Initialize the benchmark system
                        var initTask = benchmarkSystem.InitializeAsync();
                        initTask.Wait();

                        // Set the threshold
                        var setTask = benchmarkSystem.SetBaselineThresholdAsync(metricType, target, threshold);
                        setTask.Wait();

                        consoleService.WriteSuccess($"Set {metricType} threshold for {target} to {threshold:F2}");
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, "Error setting baseline threshold");
                        consoleService.WriteError($"Error setting baseline threshold: {ex.Message}");
                    }
                });
            }
        }
    }

    /// <summary>
    /// Command for comparing metrics between TARS-generated code and human-written code
    /// </summary>
    internal class CompareMetricsCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        /// <summary>
        /// Initializes a new instance of the <see cref="CompareMetricsCommand"/> class
        /// </summary>
        /// <param name="serviceProvider">The service provider</param>
        public CompareMetricsCommand(IServiceProvider serviceProvider) : base("compare", "Compare metrics between TARS-generated code and human-written code")
        {
            _serviceProvider = serviceProvider;

            // Add options
            var tarsCodePathOption = new Option<string>(
                "--tars-code",
                description: "Path to the TARS-generated code",
                getDefaultValue: () => "")
            {
                IsRequired = true
            };

            var humanCodePathOption = new Option<string>(
                "--human-code",
                description: "Path to the human-written code",
                getDefaultValue: () => "")
            {
                IsRequired = true
            };

            var outputOption = new Option<string>(
                "--output",
                description: "Path to save the comparison results",
                getDefaultValue: () => "comparison_results.json");

            var verboseOption = new Option<bool>(
                "--verbose",
                description: "Show detailed output",
                getDefaultValue: () => false);

            AddOption(tarsCodePathOption);
            AddOption(humanCodePathOption);
            AddOption(outputOption);
            AddOption(verboseOption);

            this.SetHandler((InvocationContext context) =>
            {
                var tarsCodePath = context.ParseResult.GetValueForOption(tarsCodePathOption);
                var humanCodePath = context.ParseResult.GetValueForOption(humanCodePathOption);
                var output = context.ParseResult.GetValueForOption(outputOption);
                var verbose = context.ParseResult.GetValueForOption(verboseOption);

                var logger = _serviceProvider.GetRequiredService<ILogger<BenchmarkCommand>>();
                var benchmarkSystem = _serviceProvider.GetRequiredService<BenchmarkSystem>();
                var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();

                try
                {
                    consoleService.WriteHeader("Comparing Metrics");
                    consoleService.WriteLine($"TARS code path: {tarsCodePath}");
                    consoleService.WriteLine($"Human code path: {humanCodePath}");
                    consoleService.WriteLine($"Output path: {output}");
                    consoleService.WriteLine();

                    // Initialize the benchmark system
                    var initTask = benchmarkSystem.InitializeAsync();
                    initTask.Wait();

                    // Run the comparison
                    consoleService.WriteLine("Running comparison...");
                    var comparisonTask = benchmarkSystem.CompareMetricsAsync(tarsCodePath, humanCodePath);
                    comparisonTask.Wait();
                    var results = comparisonTask.Result;

                    // Display the results
                    consoleService.WriteLine();
                    consoleService.WriteSuccess($"Comparison completed in {results.Duration.TotalSeconds:F2} seconds");
                    consoleService.WriteLine();

                    // Display summary statistics
                    DisplaySummaryStatistics(consoleService, results);

                    // Display complexity comparison
                    consoleService.WriteHeader("Complexity Comparison");
                    DisplayMetricComparison(consoleService, results.ComplexityComparison, "complexity", verbose);

                    // Display maintainability comparison
                    consoleService.WriteHeader("Maintainability Comparison");
                    DisplayMetricComparison(consoleService, results.MaintainabilityComparison, "maintainability", verbose);

                    // Display Halstead comparison
                    consoleService.WriteHeader("Halstead Comparison");
                    DisplayMetricComparison(consoleService, results.HalsteadComparison, "halstead", verbose);

                    // Display recommendations
                    DisplayRecommendations(consoleService, results);

                    // Save the results to a file
                    var json = System.Text.Json.JsonSerializer.Serialize(results, new System.Text.Json.JsonSerializerOptions
                    {
                        WriteIndented = true
                    });

                    File.WriteAllText(output, json);
                    consoleService.WriteSuccess($"Comparison results saved to {output}");
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error comparing metrics");
                    consoleService.WriteError($"Error comparing metrics: {ex.Message}");
                }
            });
        }

        /// <summary>
        /// Displays summary statistics for the comparison results
        /// </summary>
        /// <param name="consoleService">The console service</param>
        /// <param name="results">The comparison results</param>
        private void DisplaySummaryStatistics(ConsoleService consoleService, CodeComparisonResults results)
        {
            consoleService.WriteHeader("Summary Statistics");

            // Calculate overall statistics for each metric type
            var complexityStats = CalculateOverallStatistics(results.ComplexityComparison);
            var maintainabilityStats = CalculateOverallStatistics(results.MaintainabilityComparison);
            var halsteadStats = CalculateOverallStatistics(results.HalsteadComparison);

            // Display overall statistics
            consoleService.WriteLine("Complexity Metrics:");
            DisplayOverallStatistics(consoleService, complexityStats, "complexity");
            consoleService.WriteLine();

            consoleService.WriteLine("Maintainability Metrics:");
            DisplayOverallStatistics(consoleService, maintainabilityStats, "maintainability");
            consoleService.WriteLine();

            consoleService.WriteLine("Halstead Metrics:");
            DisplayOverallStatistics(consoleService, halsteadStats, "halstead");
            consoleService.WriteLine();

            // Display overall quality score
            var overallScore = CalculateOverallQualityScore(complexityStats, maintainabilityStats, halsteadStats);
            consoleService.WriteLine($"Overall Quality Score: {overallScore:F2}/10.0");

            if (overallScore >= 8.0)
            {
                consoleService.WriteSuccess("TARS code quality is excellent compared to human-written code");
            }
            else if (overallScore >= 6.0)
            {
                consoleService.WriteInfo("TARS code quality is good compared to human-written code");
            }
            else if (overallScore >= 4.0)
            {
                consoleService.WriteWarning("TARS code quality is average compared to human-written code");
            }
            else
            {
                consoleService.WriteError("TARS code quality needs improvement compared to human-written code");
            }

            consoleService.WriteLine();
        }

        /// <summary>
        /// Calculates overall statistics for a metric type
        /// </summary>
        /// <param name="comparisons">The metric comparisons</param>
        /// <returns>The overall statistics</returns>
        private (double AvgDiff, double MedianDiff, int BetterCount, int WorseCount, int TotalCount) CalculateOverallStatistics(
            Dictionary<string, MetricComparisonResult> comparisons)
        {
            if (comparisons.Count == 0)
                return (0, 0, 0, 0, 0);

            var avgDiffs = new List<double>();
            var medianDiffs = new List<double>();
            var betterCount = 0;
            var worseCount = 0;

            foreach (var comparison in comparisons.Values)
            {
                avgDiffs.Add(comparison.RelativeAverageDifference);
                medianDiffs.Add(comparison.RelativeMedianDifference);

                // For complexity, lower is better
                // For maintainability, higher is better
                var isComplexityMetric = comparison.TarsStats.GetType() == typeof(ComplexityMetric);
                var isBetter = isComplexityMetric ?
                    comparison.RelativeAverageDifference < 0 :
                    comparison.RelativeAverageDifference > 0;

                if (isBetter)
                    betterCount++;
                else
                    worseCount++;
            }

            return (
                avgDiffs.Count > 0 ? avgDiffs.Average() : 0,
                medianDiffs.Count > 0 ? medianDiffs.Average() : 0,
                betterCount,
                worseCount,
                comparisons.Count
            );
        }

        /// <summary>
        /// Displays overall statistics for a metric type
        /// </summary>
        /// <param name="consoleService">The console service</param>
        /// <param name="stats">The overall statistics</param>
        /// <param name="metricType">The metric type</param>
        private void DisplayOverallStatistics(
            ConsoleService consoleService,
            (double AvgDiff, double MedianDiff, int BetterCount, int WorseCount, int TotalCount) stats,
            string metricType)
        {
            var isComplexityMetric = metricType.Equals("complexity", StringComparison.OrdinalIgnoreCase) ||
                                    metricType.Equals("halstead", StringComparison.OrdinalIgnoreCase);
            var betterText = isComplexityMetric ? "lower" : "higher";
            var worseText = isComplexityMetric ? "higher" : "lower";

            // Display average difference
            var avgDiffFormatted = $"{stats.AvgDiff:P2}";
            var avgDiffText = $"  Average Difference: {avgDiffFormatted}";

            if ((isComplexityMetric && stats.AvgDiff < 0) || (!isComplexityMetric && stats.AvgDiff > 0))
                consoleService.WriteSuccess(avgDiffText);
            else if (Math.Abs(stats.AvgDiff) < 0.05) // Within 5%
                consoleService.WriteInfo(avgDiffText);
            else
                consoleService.WriteError(avgDiffText);

            // Display median difference
            var medianDiffFormatted = $"{stats.MedianDiff:P2}";
            var medianDiffText = $"  Median Difference: {medianDiffFormatted}";

            if ((isComplexityMetric && stats.MedianDiff < 0) || (!isComplexityMetric && stats.MedianDiff > 0))
                consoleService.WriteSuccess(medianDiffText);
            else if (Math.Abs(stats.MedianDiff) < 0.05) // Within 5%
                consoleService.WriteInfo(medianDiffText);
            else
                consoleService.WriteError(medianDiffText);

            // Display better/worse counts
            var betterPercent = stats.TotalCount > 0 ? (double)stats.BetterCount / stats.TotalCount : 0;
            var worsePercent = stats.TotalCount > 0 ? (double)stats.WorseCount / stats.TotalCount : 0;

            var betterCountText = $"  TARS is {betterText} in {stats.BetterCount}/{stats.TotalCount} cases ({betterPercent:P0})";
            var worseCountText = $"  TARS is {worseText} in {stats.WorseCount}/{stats.TotalCount} cases ({worsePercent:P0})";

            if (betterPercent >= 0.7)
                consoleService.WriteSuccess(betterCountText);
            else if (betterPercent >= 0.5)
                consoleService.WriteInfo(betterCountText);
            else
                consoleService.WriteWarning(betterCountText);

            if (worsePercent <= 0.3)
                consoleService.WriteSuccess(worseCountText);
            else if (worsePercent <= 0.5)
                consoleService.WriteInfo(worseCountText);
            else
                consoleService.WriteWarning(worseCountText);
        }

        /// <summary>
        /// Calculates an overall quality score based on the metrics
        /// </summary>
        /// <param name="complexityStats">The complexity statistics</param>
        /// <param name="maintainabilityStats">The maintainability statistics</param>
        /// <param name="halsteadStats">The Halstead statistics</param>
        /// <returns>The overall quality score (0-10)</returns>
        private double CalculateOverallQualityScore(
            (double AvgDiff, double MedianDiff, int BetterCount, int WorseCount, int TotalCount) complexityStats,
            (double AvgDiff, double MedianDiff, int BetterCount, int WorseCount, int TotalCount) maintainabilityStats,
            (double AvgDiff, double MedianDiff, int BetterCount, int WorseCount, int TotalCount) halsteadStats)
        {
            // Calculate scores for each metric type (0-10)
            var complexityScore = CalculateMetricTypeScore(complexityStats, true);
            var maintainabilityScore = CalculateMetricTypeScore(maintainabilityStats, false);
            var halsteadScore = CalculateMetricTypeScore(halsteadStats, true);

            // Weight the scores (complexity and maintainability are more important)
            var weightedScore = (complexityScore * 0.4) + (maintainabilityScore * 0.4) + (halsteadScore * 0.2);

            return Math.Min(10.0, Math.Max(0.0, weightedScore));
        }

        /// <summary>
        /// Calculates a score for a metric type
        /// </summary>
        /// <param name="stats">The metric statistics</param>
        /// <param name="lowerIsBetter">Whether lower values are better</param>
        /// <returns>The score (0-10)</returns>
        private double CalculateMetricTypeScore(
            (double AvgDiff, double MedianDiff, int BetterCount, int WorseCount, int TotalCount) stats,
            bool lowerIsBetter)
        {
            if (stats.TotalCount == 0)
                return 5.0; // Neutral score if no data

            // Calculate score based on average difference
            var avgDiffScore = lowerIsBetter ?
                10.0 - (stats.AvgDiff * 20.0) : // Lower is better, so negative diff is good
                10.0 + (stats.AvgDiff * 20.0);  // Higher is better, so positive diff is good

            // Calculate score based on better/worse ratio
            var betterRatio = (double)stats.BetterCount / stats.TotalCount;
            var betterWorseScore = betterRatio * 10.0;

            // Combine scores (weighted average)
            var combinedScore = (avgDiffScore * 0.7) + (betterWorseScore * 0.3);

            return Math.Min(10.0, Math.Max(0.0, combinedScore));
        }

        /// <summary>
        /// Displays a metric comparison
        /// </summary>
        /// <param name="consoleService">The console service</param>
        /// <param name="comparisons">The metric comparisons</param>
        /// <param name="metricType">The metric type</param>
        /// <param name="verbose">Whether to show detailed output</param>
        private void DisplayMetricComparison(
            ConsoleService consoleService,
            Dictionary<string, MetricComparisonResult> comparisons,
            string metricType,
            bool verbose)
        {
            if (comparisons.Count == 0)
            {
                consoleService.WriteWarning("No comparison data available");
                consoleService.WriteLine();
                return;
            }

            var isComplexityMetric = metricType.Equals("complexity", StringComparison.OrdinalIgnoreCase) ||
                                    metricType.Equals("halstead", StringComparison.OrdinalIgnoreCase);

            foreach (var (target, comparison) in comparisons)
            {
                consoleService.WriteLine($"Target: {target}");

                // Display average values with color coding
                var tarsAvg = $"  TARS Average: {comparison.TarsStats.AverageValue:F2}";
                var humanAvg = $"  Human Average: {comparison.HumanStats.AverageValue:F2}";

                consoleService.WriteLine(tarsAvg);
                consoleService.WriteLine(humanAvg);

                // Display difference with color coding
                var diffText = $"  Difference: {comparison.AverageDifference:F2} ({comparison.RelativeAverageDifference:P2})";

                if ((isComplexityMetric && comparison.AverageDifference < 0) ||
                    (!isComplexityMetric && comparison.AverageDifference > 0))
                {
                    consoleService.WriteSuccess(diffText);
                }
                else if (Math.Abs(comparison.RelativeAverageDifference) < 0.05) // Within 5%
                {
                    consoleService.WriteInfo(diffText);
                }
                else
                {
                    consoleService.WriteError(diffText);
                }

                // Display median values
                consoleService.WriteLine($"  TARS Median: {comparison.TarsStats.MedianValue:F2}");
                consoleService.WriteLine($"  Human Median: {comparison.HumanStats.MedianValue:F2}");

                // Display median difference with color coding
                var medianDiffText = $"  Difference: {comparison.MedianDifference:F2} ({comparison.RelativeMedianDifference:P2})";

                if ((isComplexityMetric && comparison.MedianDifference < 0) ||
                    (!isComplexityMetric && comparison.MedianDifference > 0))
                {
                    consoleService.WriteSuccess(medianDiffText);
                }
                else if (Math.Abs(comparison.RelativeMedianDifference) < 0.05) // Within 5%
                {
                    consoleService.WriteInfo(medianDiffText);
                }
                else
                {
                    consoleService.WriteError(medianDiffText);
                }

                // Display percentile differences if verbose
                if (verbose)
                {
                    consoleService.WriteLine("  Percentile Differences:");

                    foreach (var (percentile, difference) in comparison.PercentileDifferences)
                    {
                        var percentileDiffText = $"    {percentile}th: {difference:F2}";

                        if ((isComplexityMetric && difference < 0) || (!isComplexityMetric && difference > 0))
                            consoleService.WriteSuccess(percentileDiffText);
                        else if (Math.Abs(difference) < comparison.HumanStats.Percentiles[percentile] * 0.05) // Within 5%
                            consoleService.WriteInfo(percentileDiffText);
                        else
                            consoleService.WriteError(percentileDiffText);
                    }
                }

                consoleService.WriteLine();
            }
        }

        /// <summary>
        /// Displays recommendations based on the comparison results
        /// </summary>
        /// <param name="consoleService">The console service</param>
        /// <param name="results">The comparison results</param>
        private void DisplayRecommendations(ConsoleService consoleService, CodeComparisonResults results)
        {
            consoleService.WriteHeader("Recommendations");

            // Analyze complexity issues
            var complexityIssues = IdentifyComplexityIssues(results.ComplexityComparison);
            if (complexityIssues.Count > 0)
            {
                consoleService.WriteLine("Complexity Recommendations:");
                foreach (var issue in complexityIssues)
                {
                    consoleService.WriteWarning($"  - {issue}");
                }
                consoleService.WriteLine();
            }

            // Analyze maintainability issues
            var maintainabilityIssues = IdentifyMaintainabilityIssues(results.MaintainabilityComparison);
            if (maintainabilityIssues.Count > 0)
            {
                consoleService.WriteLine("Maintainability Recommendations:");
                foreach (var issue in maintainabilityIssues)
                {
                    consoleService.WriteWarning($"  - {issue}");
                }
                consoleService.WriteLine();
            }

            // Analyze Halstead issues
            var halsteadIssues = IdentifyHalsteadIssues(results.HalsteadComparison);
            if (halsteadIssues.Count > 0)
            {
                consoleService.WriteLine("Halstead Recommendations:");
                foreach (var issue in halsteadIssues)
                {
                    consoleService.WriteWarning($"  - {issue}");
                }
                consoleService.WriteLine();
            }

            // If no issues found, display a success message
            if (complexityIssues.Count == 0 && maintainabilityIssues.Count == 0 && halsteadIssues.Count == 0)
            {
                consoleService.WriteSuccess("No significant issues found. TARS code quality is comparable to or better than human-written code.");
                consoleService.WriteLine();
            }
        }

        /// <summary>
        /// Identifies complexity issues in the comparison results
        /// </summary>
        /// <param name="comparisons">The complexity comparisons</param>
        /// <returns>A list of issues</returns>
        private List<string> IdentifyComplexityIssues(Dictionary<string, MetricComparisonResult> comparisons)
        {
            var issues = new List<string>();

            foreach (var (target, comparison) in comparisons)
            {
                // For complexity, lower is better
                if (comparison.RelativeAverageDifference > 0.2) // 20% higher complexity
                {
                    issues.Add($"Reduce complexity in {target} components (currently {comparison.RelativeAverageDifference:P0} higher than human code)");
                }
                else if (comparison.RelativeAverageDifference > 0.1) // 10% higher complexity
                {
                    issues.Add($"Consider simplifying {target} components (currently {comparison.RelativeAverageDifference:P0} higher than human code)");
                }
            }

            return issues;
        }

        /// <summary>
        /// Identifies maintainability issues in the comparison results
        /// </summary>
        /// <param name="comparisons">The maintainability comparisons</param>
        /// <returns>A list of issues</returns>
        private List<string> IdentifyMaintainabilityIssues(Dictionary<string, MetricComparisonResult> comparisons)
        {
            var issues = new List<string>();

            foreach (var (target, comparison) in comparisons)
            {
                // For maintainability, higher is better
                if (comparison.RelativeAverageDifference < -0.2) // 20% lower maintainability
                {
                    issues.Add($"Improve maintainability in {target} components (currently {Math.Abs(comparison.RelativeAverageDifference):P0} lower than human code)");
                }
                else if (comparison.RelativeAverageDifference < -0.1) // 10% lower maintainability
                {
                    issues.Add($"Consider enhancing maintainability in {target} components (currently {Math.Abs(comparison.RelativeAverageDifference):P0} lower than human code)");
                }
            }

            return issues;
        }

        /// <summary>
        /// Identifies Halstead issues in the comparison results
        /// </summary>
        /// <param name="comparisons">The Halstead comparisons</param>
        /// <returns>A list of issues</returns>
        private List<string> IdentifyHalsteadIssues(Dictionary<string, MetricComparisonResult> comparisons)
        {
            var issues = new List<string>();

            foreach (var (target, comparison) in comparisons)
            {
                // For Halstead metrics, lower is generally better
                if (comparison.RelativeAverageDifference > 0.2) // 20% higher Halstead metrics
                {
                    issues.Add($"Reduce complexity in {target} components (Halstead metrics are {comparison.RelativeAverageDifference:P0} higher than human code)");
                }
                else if (comparison.RelativeAverageDifference > 0.1) // 10% higher Halstead metrics
                {
                    issues.Add($"Consider simplifying {target} components (Halstead metrics are {comparison.RelativeAverageDifference:P0} higher than human code)");
                }
            }

            return issues;
        }
    }
}
