using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services.Testing;

/// <summary>
/// Repository for test patterns
/// </summary>
public class TestPatternRepository : ITestPatternRepository
{
    private readonly ILogger<TestPatternRepository> _logger;
    private readonly string _patternFilePath;
    private readonly Dictionary<string, TestPattern> _patterns = new();
    private readonly object _lock = new();
    private bool _isLoaded = false;

    /// <summary>
    /// Initializes a new instance of the TestPatternRepository class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    public TestPatternRepository(ILogger<TestPatternRepository> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        
        // Create the patterns directory if it doesn't exist
        var patternsDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Patterns");
        if (!Directory.Exists(patternsDirectory))
        {
            Directory.CreateDirectory(patternsDirectory);
        }

        _patternFilePath = Path.Combine(patternsDirectory, "test_patterns.json");
        LoadPatterns();
    }

    /// <inheritdoc/>
    public TestPattern GetPattern(string type, string methodName)
    {
        EnsurePatternsLoaded();

        var key = GetKey(type, methodName);
        lock (_lock)
        {
            if (_patterns.TryGetValue(key, out var pattern))
            {
                return pattern;
            }

            // Try with just the type
            key = GetKey(type, "*");
            if (_patterns.TryGetValue(key, out pattern))
            {
                return pattern;
            }

            return null;
        }
    }

    /// <inheritdoc/>
    public async Task SavePatternAsync(TestPattern pattern)
    {
        if (pattern == null)
        {
            throw new ArgumentNullException(nameof(pattern));
        }

        EnsurePatternsLoaded();

        var key = GetKey(pattern.Type, pattern.MethodName);
        lock (_lock)
        {
            _patterns[key] = pattern;
        }

        await SavePatternsAsync();
    }

    /// <inheritdoc/>
    public async Task LearnFromSuccessfulTestAsync(TestResult test)
    {
        if (test == null)
        {
            throw new ArgumentNullException(nameof(test));
        }

        try
        {
            // Extract the test pattern
            var (type, methodName, testDataTemplate, assertionTemplate) = ExtractTestPattern(test);
            if (string.IsNullOrEmpty(type) || string.IsNullOrEmpty(methodName))
            {
                return;
            }

            // Get existing pattern or create a new one
            var key = GetKey(type, methodName);
            TestPattern pattern;
            lock (_lock)
            {
                if (_patterns.TryGetValue(key, out var existingPattern))
                {
                    pattern = existingPattern;
                    pattern.SuccessCount++;
                }
                else
                {
                    pattern = new TestPattern
                    {
                        Type = type,
                        MethodName = methodName,
                        TestDataTemplate = testDataTemplate,
                        AssertionTemplate = assertionTemplate,
                        SuccessCount = 1,
                        FailureCount = 0
                    };
                    _patterns[key] = pattern;
                }
            }

            await SavePatternsAsync();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error learning from successful test");
        }
    }

    /// <summary>
    /// Extracts a test pattern from a test result
    /// </summary>
    /// <param name="test">Test result</param>
    /// <returns>Tuple containing type, method name, test data template, and assertion template</returns>
    private (string Type, string MethodName, string TestDataTemplate, string AssertionTemplate) ExtractTestPattern(TestResult test)
    {
        try
        {
            // Extract method name from test name
            var methodNameMatch = Regex.Match(test.TestName, @"^([a-zA-Z0-9_]+)_Should");
            if (!methodNameMatch.Success)
            {
                return (null, null, null, null);
            }

            var methodName = methodNameMatch.Groups[1].Value;

            // TODO: Extract type, test data template, and assertion template from test code
            // This would require access to the test code, which we don't have in the TestResult
            // For now, we'll return placeholder values

            return ("int", methodName, "var {name} = 42;", "Assert.AreEqual(42, result);");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting test pattern");
            return (null, null, null, null);
        }
    }

    /// <summary>
    /// Gets a key for a type and method name
    /// </summary>
    /// <param name="type">Type</param>
    /// <param name="methodName">Method name</param>
    /// <returns>Key</returns>
    private static string GetKey(string type, string methodName)
    {
        return $"{type}:{methodName}";
    }

    /// <summary>
    /// Ensures patterns are loaded
    /// </summary>
    private void EnsurePatternsLoaded()
    {
        if (!_isLoaded)
        {
            LoadPatterns();
        }
    }

    /// <summary>
    /// Loads patterns from the file
    /// </summary>
    private void LoadPatterns()
    {
        lock (_lock)
        {
            try
            {
                if (File.Exists(_patternFilePath))
                {
                    var json = File.ReadAllText(_patternFilePath);
                    var patterns = JsonSerializer.Deserialize<List<TestPattern>>(json);
                    if (patterns != null)
                    {
                        _patterns.Clear();
                        foreach (var pattern in patterns)
                        {
                            var key = GetKey(pattern.Type, pattern.MethodName);
                            _patterns[key] = pattern;
                        }
                    }
                }
                _isLoaded = true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error loading test patterns");
            }
        }
    }

    /// <summary>
    /// Saves patterns to the file
    /// </summary>
    private async Task SavePatternsAsync()
    {
        try
        {
            List<TestPattern> patterns;
            lock (_lock)
            {
                patterns = _patterns.Values.ToList();
            }

            var json = JsonSerializer.Serialize(patterns, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(_patternFilePath, json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving test patterns");
        }
    }
}
