using System.Text.Json;

namespace TarsCli.Services;

/// <summary>
/// Service for learning from code transformations
/// </summary>
public class TransformationLearningService
{
    private readonly ILogger<TransformationLearningService> _logger;
    private readonly string _learningDataPath;
    private TransformationLearningData _learningData;

    public TransformationLearningService(ILogger<TransformationLearningService> logger)
    {
        _logger = logger;
            
        // Store learning data in the user's app data directory
        var appDataPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "TarsCli",
            "Retroaction");
            
        // Ensure directory exists
        if (!Directory.Exists(appDataPath))
        {
            Directory.CreateDirectory(appDataPath);
        }
            
        _learningDataPath = Path.Combine(appDataPath, "learning_data.json");
            
        // Load existing learning data or create new
        _learningData = LoadLearningData();
    }

    /// <summary>
    /// Records a successful transformation
    /// </summary>
    /// <param name="ruleName">Name of the rule that was applied</param>
    /// <param name="originalCode">The original code that was transformed</param>
    /// <param name="transformedCode">The transformed code</param>
    /// <param name="filePath">Path to the file that was transformed</param>
    /// <param name="wasAccepted">Whether the transformation was accepted by the user</param>
    public async Task RecordTransformationAsync(
        string ruleName,
        string originalCode,
        string transformedCode,
        string filePath,
        bool wasAccepted)
    {
        try
        {
            // Create a new transformation record
            var record = new TransformationRecord
            {
                RuleName = ruleName,
                OriginalCode = originalCode,
                TransformedCode = transformedCode,
                FilePath = filePath,
                Timestamp = DateTime.UtcNow,
                WasAccepted = wasAccepted
            };
                
            // Add to learning data
            _learningData.Transformations.Add(record);
                
            // Update rule statistics
            if (!_learningData.RuleStatistics.TryGetValue(ruleName, out var stats))
            {
                stats = new RuleStatistics
                {
                    RuleName = ruleName,
                    TotalApplications = 0,
                    SuccessfulApplications = 0
                };
                _learningData.RuleStatistics[ruleName] = stats;
            }
                
            stats.TotalApplications++;
            if (wasAccepted)
            {
                stats.SuccessfulApplications++;
            }
                
            // Save learning data
            await SaveLearningDataAsync();
                
            _logger.LogInformation($"Recorded transformation for rule '{ruleName}' (accepted: {wasAccepted})");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error recording transformation");
        }
    }

    /// <summary>
    /// Gets statistics for all rules
    /// </summary>
    public Dictionary<string, RuleStatistics> GetRuleStatistics()
    {
        return _learningData.RuleStatistics;
    }

    /// <summary>
    /// Gets the success rate for a rule
    /// </summary>
    /// <param name="ruleName">Name of the rule</param>
    public double GetRuleSuccessRate(string ruleName)
    {
        if (_learningData.RuleStatistics.TryGetValue(ruleName, out var stats) && stats.TotalApplications > 0)
        {
            return (double)stats.SuccessfulApplications / stats.TotalApplications;
        }
            
        return 0;
    }

    /// <summary>
    /// Gets similar code patterns that have been successfully transformed
    /// </summary>
    /// <param name="code">The code to find similar patterns for</param>
    /// <param name="similarityThreshold">Threshold for similarity (0-1)</param>
    public List<TransformationRecord> GetSimilarPatterns(string code, double similarityThreshold = 0.7)
    {
        return _learningData.Transformations
            .Where(t => t.WasAccepted && CalculateSimilarity(t.OriginalCode, code) >= similarityThreshold)
            .OrderByDescending(t => CalculateSimilarity(t.OriginalCode, code))
            .ToList();
    }

    /// <summary>
    /// Suggests new rules based on successful transformations
    /// </summary>
    public List<SuggestedRule> SuggestNewRules()
    {
        var suggestedRules = new List<SuggestedRule>();
            
        // Group successful transformations by pattern similarity
        var successfulTransformations = _learningData.Transformations
            .Where(t => t.WasAccepted)
            .ToList();
            
        // This is a simplified approach - a real implementation would use more sophisticated clustering
        var clusters = new List<List<TransformationRecord>>();
            
        foreach (var transformation in successfulTransformations)
        {
            var addedToCluster = false;
                
            foreach (var cluster in clusters)
            {
                if (cluster.Any(t => CalculateSimilarity(t.OriginalCode, transformation.OriginalCode) >= 0.8))
                {
                    cluster.Add(transformation);
                    addedToCluster = true;
                    break;
                }
            }
                
            if (!addedToCluster)
            {
                clusters.Add([transformation]);
            }
        }
            
        // For each cluster with more than 3 items, suggest a new rule
        foreach (var cluster in clusters.Where(c => c.Count >= 3))
        {
            // Find the most common rule name in the cluster
            var mostCommonRuleName = cluster
                .GroupBy(t => t.RuleName)
                .OrderByDescending(g => g.Count())
                .First()
                .Key;
                
            // Create a suggested rule based on the cluster
            var suggestedRule = new SuggestedRule
            {
                Name = $"Suggested_{mostCommonRuleName}_{Guid.NewGuid().ToString().Substring(0, 8)}",
                Pattern = GeneralizePattern(cluster.Select(t => t.OriginalCode).ToList()),
                Replacement = GeneralizePattern(cluster.Select(t => t.TransformedCode).ToList()),
                Confidence = Math.Min(1.0, cluster.Count / 10.0),
                SupportingExamples = cluster.Take(5).ToList()
            };
                
            suggestedRules.Add(suggestedRule);
        }
            
        return suggestedRules;
    }

    /// <summary>
    /// Loads learning data from disk
    /// </summary>
    private TransformationLearningData LoadLearningData()
    {
        try
        {
            if (File.Exists(_learningDataPath))
            {
                var json = File.ReadAllText(_learningDataPath);
                return JsonSerializer.Deserialize<TransformationLearningData>(json) ?? CreateNewLearningData();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading learning data");
        }
            
        return CreateNewLearningData();
    }

    /// <summary>
    /// Saves learning data to disk
    /// </summary>
    private async Task SaveLearningDataAsync()
    {
        try
        {
            var json = JsonSerializer.Serialize(_learningData, new JsonSerializerOptions
            {
                WriteIndented = true
            });
                
            await File.WriteAllTextAsync(_learningDataPath, json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving learning data");
        }
    }

    /// <summary>
    /// Creates new learning data
    /// </summary>
    private TransformationLearningData CreateNewLearningData()
    {
        return new TransformationLearningData
        {
            Transformations = [],
            RuleStatistics = new Dictionary<string, RuleStatistics>()
        };
    }

    /// <summary>
    /// Calculates similarity between two code snippets
    /// </summary>
    /// <param name="code1">First code snippet</param>
    /// <param name="code2">Second code snippet</param>
    private double CalculateSimilarity(string code1, string code2)
    {
        // This is a simplified implementation using Levenshtein distance
        // A real implementation would use more sophisticated code similarity metrics
            
        // Normalize the code by removing whitespace and converting to lowercase
        var normalized1 = NormalizeCode(code1);
        var normalized2 = NormalizeCode(code2);
            
        // Calculate Levenshtein distance
        var distance = LevenshteinDistance(normalized1, normalized2);
            
        // Convert to similarity (0-1)
        var maxLength = Math.Max(normalized1.Length, normalized2.Length);
        if (maxLength == 0) return 1.0; // Both strings are empty
            
        return 1.0 - (double)distance / maxLength;
    }

    /// <summary>
    /// Normalizes code for comparison
    /// </summary>
    private string NormalizeCode(string code)
    {
        // Remove whitespace and convert to lowercase
        return new string(code.Where(c => !char.IsWhiteSpace(c)).ToArray()).ToLowerInvariant();
    }

    /// <summary>
    /// Calculates Levenshtein distance between two strings
    /// </summary>
    private int LevenshteinDistance(string s, string t)
    {
        var n = s.Length;
        var m = t.Length;
        var d = new int[n + 1, m + 1];
            
        if (n == 0) return m;
        if (m == 0) return n;
            
        for (var i = 0; i <= n; i++)
            d[i, 0] = i;
            
        for (var j = 0; j <= m; j++)
            d[0, j] = j;
            
        for (var j = 1; j <= m; j++)
        {
            for (var i = 1; i <= n; i++)
            {
                var cost = (s[i - 1] == t[j - 1]) ? 0 : 1;
                d[i, j] = Math.Min(
                    Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                    d[i - 1, j - 1] + cost);
            }
        }
            
        return d[n, m];
    }

    /// <summary>
    /// Generalizes a set of code patterns into a single pattern
    /// </summary>
    /// <param name="patterns">The patterns to generalize</param>
    private string GeneralizePattern(List<string> patterns)
    {
        // This is a simplified implementation
        // A real implementation would use more sophisticated pattern generalization techniques
            
        if (patterns.Count == 0) return string.Empty;
        if (patterns.Count == 1) return patterns[0];
            
        // Start with the first pattern
        var generalizedPattern = patterns[0];
            
        // For each subsequent pattern, find common parts
        for (var i = 1; i < patterns.Count; i++)
        {
            generalizedPattern = FindCommonPattern(generalizedPattern, patterns[i]);
        }
            
        return generalizedPattern;
    }

    /// <summary>
    /// Finds the common pattern between two code snippets
    /// </summary>
    private string FindCommonPattern(string pattern1, string pattern2)
    {
        // Split into tokens
        var tokens1 = pattern1.Split([' ', '\t', '\r', '\n'], StringSplitOptions.RemoveEmptyEntries);
        var tokens2 = pattern2.Split([' ', '\t', '\r', '\n'], StringSplitOptions.RemoveEmptyEntries);
            
        // Find longest common subsequence
        var lcs = LongestCommonSubsequence(tokens1, tokens2);
            
        // Convert back to string
        return string.Join(" ", lcs);
    }

    /// <summary>
    /// Finds the longest common subsequence of two arrays
    /// </summary>
    private string[] LongestCommonSubsequence(string[] a, string[] b)
    {
        var lengths = new int[a.Length + 1, b.Length + 1];
            
        // Compute the length of the LCS
        for (var i = 0; i <= a.Length; i++)
        {
            for (var j = 0; j <= b.Length; j++)
            {
                if (i == 0 || j == 0)
                    lengths[i, j] = 0;
                else if (a[i - 1] == b[j - 1])
                    lengths[i, j] = lengths[i - 1, j - 1] + 1;
                else
                    lengths[i, j] = Math.Max(lengths[i - 1, j], lengths[i, j - 1]);
            }
        }
            
        // Reconstruct the LCS
        var result = new List<string>();
        int x = a.Length, y = b.Length;
            
        while (x > 0 && y > 0)
        {
            if (a[x - 1] == b[y - 1])
            {
                result.Add(a[x - 1]);
                x--;
                y--;
            }
            else if (lengths[x - 1, y] > lengths[x, y - 1])
            {
                x--;
            }
            else
            {
                y--;
            }
        }
            
        result.Reverse();
        return result.ToArray();
    }
}

/// <summary>
/// Data structure for storing transformation learning data
/// </summary>
public class TransformationLearningData
{
    public List<TransformationRecord> Transformations { get; set; }
    public Dictionary<string, RuleStatistics> RuleStatistics { get; set; }
}

/// <summary>
/// Record of a code transformation
/// </summary>
public class TransformationRecord
{
    public string RuleName { get; set; }
    public string OriginalCode { get; set; }
    public string TransformedCode { get; set; }
    public string FilePath { get; set; }
    public DateTime Timestamp { get; set; }
    public bool WasAccepted { get; set; }
}

/// <summary>
/// Statistics for a transformation rule
/// </summary>
public class RuleStatistics
{
    public string RuleName { get; set; }
    public int TotalApplications { get; set; }
    public int SuccessfulApplications { get; set; }
}

/// <summary>
/// A suggested new transformation rule
/// </summary>
public class SuggestedRule
{
    public string Name { get; set; }
    public string Pattern { get; set; }
    public string Replacement { get; set; }
    public double Confidence { get; set; }
    public List<TransformationRecord> SupportingExamples { get; set; }
}