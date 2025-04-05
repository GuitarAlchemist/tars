using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for classifying content
/// </summary>
public class ContentClassifierService : IContentClassifierService
{
    private readonly ILogger<ContentClassifierService> _logger;
    private readonly IMetascriptService _metascriptService;
    private readonly List<ClassificationRule> _rules = new();
    private readonly Dictionary<string, ContentClassification> _classificationCache = new();
    private readonly string _rulesFilePath;
    private bool _rulesLoaded = false;

    /// <summary>
    /// Initializes a new instance of the <see cref="ContentClassifierService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="metascriptService">The metascript service</param>
    public ContentClassifierService(
        ILogger<ContentClassifierService> logger,
        IMetascriptService metascriptService)
    {
        _logger = logger;
        _metascriptService = metascriptService;
        _rulesFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "ClassificationRules.json");
    }

    /// <inheritdoc/>
    public async Task<ContentClassification> ClassifyContentAsync(string content, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Classifying content of length {Length}", content?.Length ?? 0);

            // Check if we have a cached classification
            var cacheKey = $"{content?.GetHashCode() ?? 0}_{string.Join("_", options?.Select(kv => $"{kv.Key}={kv.Value}") ?? Array.Empty<string>())}";
            if (_classificationCache.TryGetValue(cacheKey, out var cachedClassification))
            {
                _logger.LogInformation("Using cached classification for content");
                return cachedClassification;
            }

            // Ensure rules are loaded
            await EnsureRulesLoadedAsync();

            // Create a new classification
            var classification = new ContentClassification
            {
                Content = content ?? string.Empty,
                ClassificationSource = "rule-based"
            };

            // Apply rules to determine the primary category
            var categoryScores = new Dictionary<ContentCategory, double>();
            foreach (var rule in _rules.Where(r => r.IsEnabled))
            {
                var score = CalculateRuleScore(content ?? string.Empty, rule);
                if (score >= rule.MinConfidence)
                {
                    if (!categoryScores.ContainsKey(rule.Category))
                    {
                        categoryScores[rule.Category] = 0;
                    }
                    categoryScores[rule.Category] += score * rule.Weight;

                    // Add tags from the rule
                    foreach (var tag in rule.Tags)
                    {
                        if (!classification.Tags.Contains(tag))
                        {
                            classification.Tags.Add(tag);
                        }
                    }
                }
            }

            // Determine the primary category
            if (categoryScores.Any())
            {
                var primaryCategory = categoryScores.OrderByDescending(kv => kv.Value).First();
                classification.PrimaryCategory = primaryCategory.Key;
                classification.ConfidenceScore = Math.Min(1.0, primaryCategory.Value);

                // Determine secondary categories
                foreach (var category in categoryScores.OrderByDescending(kv => kv.Value).Skip(1).Take(3))
                {
                    if (category.Value >= 0.3) // Only include categories with a reasonable score
                    {
                        classification.SecondaryCategories.Add(category.Key);
                    }
                }
            }
            else
            {
                classification.PrimaryCategory = ContentCategory.Unknown;
                classification.ConfidenceScore = 0.0;
            }

            // Calculate relevance and quality scores
            classification.RelevanceScore = await CalculateRelevanceScoreAsync(content ?? string.Empty, options);
            classification.QualityScore = await CalculateQualityScoreAsync(content ?? string.Empty, options);

            // Add to cache
            _classificationCache[cacheKey] = classification;

            _logger.LogInformation("Classified content as {Category} with confidence {Confidence}",
                classification.PrimaryCategory, classification.ConfidenceScore);
            return classification;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error classifying content");
            return new ContentClassification
            {
                Content = content ?? string.Empty,
                PrimaryCategory = ContentCategory.Unknown,
                ConfidenceScore = 0.0,
                RelevanceScore = 0.0,
                QualityScore = 0.0,
                ClassificationSource = "error"
            };
        }
    }

    /// <inheritdoc/>
    public async Task<ContentClassificationBatch> ClassifyDocumentAsync(DocumentParsingResult document, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Classifying document: {DocumentPath}", document.DocumentPath);

            // Create a batch for the document
            var batch = new ContentClassificationBatch
            {
                Source = document.DocumentPath,
                Metadata = new Dictionary<string, string>
                {
                    { "DocumentType", document.DocumentType.ToString() },
                    { "Title", document.Title }
                }
            };

            // Classify each section
            foreach (var section in document.Sections)
            {
                var sectionOptions = new Dictionary<string, string>(options ?? new Dictionary<string, string>())
                {
                    { "SectionHeading", section.Heading },
                    { "SectionType", section.ContentType.ToString() }
                };

                var classification = await ClassifyContentAsync(section.RawContent, sectionOptions);
                classification.Metadata["SectionId"] = section.Id;
                classification.Metadata["SectionHeading"] = section.Heading;
                classification.Metadata["SectionType"] = section.ContentType.ToString();

                batch.Classifications.Add(classification);
            }

            // Classify code blocks
            foreach (var section in document.Sections)
            {
                foreach (var codeBlock in section.CodeBlocks)
                {
                    var codeOptions = new Dictionary<string, string>(options ?? new Dictionary<string, string>())
                    {
                        { "Language", codeBlock.Language },
                        { "SectionId", section.Id },
                        { "SectionHeading", section.Heading }
                    };

                    var classification = await ClassifyContentAsync(codeBlock.Code, codeOptions);
                    classification.Metadata["CodeBlockId"] = codeBlock.Id;
                    classification.Metadata["Language"] = codeBlock.Language;
                    classification.Metadata["SectionId"] = section.Id;
                    classification.Metadata["SectionHeading"] = section.Heading;
                    classification.PrimaryCategory = ContentCategory.CodeExample; // Override for code blocks

                    batch.Classifications.Add(classification);
                }
            }

            _logger.LogInformation("Classified document with {ClassificationCount} classifications",
                batch.Classifications.Count);
            return batch;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error classifying document: {DocumentPath}", document.DocumentPath);
            return new ContentClassificationBatch
            {
                Source = document.DocumentPath,
                Metadata = new Dictionary<string, string>
                {
                    { "Error", ex.Message }
                }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<ContentClassificationBatch> ClassifySectionsAsync(IEnumerable<ContentSection> sections, Dictionary<string, string>? options = null)
    {
        try
        {
            var sectionsList = sections.ToList();
            _logger.LogInformation("Classifying {Count} content sections", sectionsList.Count);

            // Create a batch for the sections
            var batch = new ContentClassificationBatch
            {
                Source = "content-sections",
                Metadata = new Dictionary<string, string>
                {
                    { "SectionCount", sectionsList.Count.ToString() }
                }
            };

            // Classify each section
            foreach (var section in sectionsList)
            {
                var sectionOptions = new Dictionary<string, string>(options ?? new Dictionary<string, string>())
                {
                    { "SectionHeading", section.Heading },
                    { "SectionType", section.ContentType.ToString() }
                };

                var classification = await ClassifyContentAsync(section.RawContent, sectionOptions);
                classification.Metadata["SectionId"] = section.Id;
                classification.Metadata["SectionHeading"] = section.Heading;
                classification.Metadata["SectionType"] = section.ContentType.ToString();

                batch.Classifications.Add(classification);
            }

            _logger.LogInformation("Classified {Count} sections", batch.Classifications.Count);
            return batch;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error classifying sections");
            return new ContentClassificationBatch
            {
                Source = "content-sections",
                Metadata = new Dictionary<string, string>
                {
                    { "Error", ex.Message }
                }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<double> CalculateRelevanceScoreAsync(string content, Dictionary<string, string>? context = null)
    {
        try
        {
            _logger.LogInformation("Calculating relevance score for content of length {Length}", content?.Length ?? 0);

            // If no context is provided, assume medium relevance
            if (context == null || !context.Any())
            {
                return 0.5;
            }

            // Simple relevance scoring based on keyword matching
            if (string.IsNullOrWhiteSpace(content))
            {
                return 0.0;
            }

            double relevanceScore = 0.5; // Default to medium relevance

            // Extract keywords from context
            var keywords = new List<string>();
            if (context.TryGetValue("Keywords", out var keywordsStr))
            {
                keywords.AddRange(keywordsStr.Split(',', ';').Select(k => k.Trim().ToLowerInvariant()));
            }

            if (context.TryGetValue("Topic", out var topic))
            {
                keywords.AddRange(topic.Split(' ').Where(w => w.Length > 3).Select(w => w.ToLowerInvariant()));
            }

            // If we have keywords, calculate relevance based on keyword matches
            if (keywords.Any())
            {
                var contentLower = content.ToLowerInvariant();
                int matches = keywords.Count(k => contentLower.Contains(k));
                relevanceScore = Math.Min(1.0, (double)matches / keywords.Count * 1.5);
            }

            // Ensure the score is between 0 and 1
            relevanceScore = Math.Max(0, Math.Min(1, relevanceScore));

            _logger.LogInformation("Calculated relevance score: {RelevanceScore}", relevanceScore);
            return relevanceScore;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating relevance score");
            return 0.5; // Default to medium relevance
        }
    }

    /// <inheritdoc/>
    public async Task<double> CalculateQualityScoreAsync(string content, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Calculating quality score for content of length {Length}", content?.Length ?? 0);

            // Simple quality scoring based on content length, structure, and complexity
            if (string.IsNullOrWhiteSpace(content))
            {
                return 0.0;
            }

            // Calculate basic metrics
            var length = content.Length;
            var sentences = Regex.Split(content, @"(?<=[.!?])\s+").Length;
            var words = Regex.Matches(content, @"\b\w+\b").Count;
            var uniqueWords = Regex.Matches(content, @"\b\w+\b")
                .Select(m => m.Value.ToLowerInvariant())
                .Distinct()
                .Count();
            var codeBlocks = Regex.Matches(content, @"```[a-zA-Z0-9]*\n[\s\S]*?\n```").Count;
            var bulletPoints = Regex.Matches(content, @"^\s*[-*]\s+", RegexOptions.Multiline).Count;
            var headings = Regex.Matches(content, @"^#{1,6}\s+.+$", RegexOptions.Multiline).Count;

            // Calculate quality score components
            var lengthScore = Math.Min(1.0, length / 1000.0); // Longer content is generally better, up to a point
            var structureScore = Math.Min(1.0, (headings + bulletPoints + codeBlocks) / 10.0); // Structured content is better
            var vocabularyScore = uniqueWords > 0 ? Math.Min(1.0, (double)uniqueWords / words) : 0.0; // Diverse vocabulary is better
            var sentenceScore = sentences > 0 ? Math.Min(1.0, (double)words / sentences / 20.0) : 0.0; // Reasonable sentence length

            // Combine scores with weights
            var qualityScore = (lengthScore * 0.3) + (structureScore * 0.3) + (vocabularyScore * 0.2) + (sentenceScore * 0.2);

            // Adjust based on options
            if (options != null)
            {
                if (options.TryGetValue("ContentType", out var contentType))
                {
                    // Adjust score based on content type
                    if (contentType.Equals("CodeExample", StringComparison.OrdinalIgnoreCase) && codeBlocks > 0)
                    {
                        qualityScore += 0.2; // Bonus for code examples that actually contain code
                    }
                    else if (contentType.Equals("Concept", StringComparison.OrdinalIgnoreCase) && headings > 0)
                    {
                        qualityScore += 0.1; // Bonus for well-structured concepts
                    }
                }
            }

            // Ensure the score is between 0 and 1
            qualityScore = Math.Max(0, Math.Min(1, qualityScore));

            _logger.LogInformation("Calculated quality score: {QualityScore}", qualityScore);
            return qualityScore;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating quality score");
            return 0.5; // Default to medium quality
        }
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<ClassificationRule>> GetClassificationRulesAsync()
    {
        await EnsureRulesLoadedAsync();
        return _rules;
    }

    /// <inheritdoc/>
    public async Task<ClassificationRule> AddClassificationRuleAsync(ClassificationRule rule)
    {
        try
        {
            _logger.LogInformation("Adding classification rule: {RuleName}", rule.Name);

            // Ensure rules are loaded
            await EnsureRulesLoadedAsync();

            // Add the rule
            _rules.Add(rule);

            // Save the rules
            await SaveRulesAsync();

            _logger.LogInformation("Added classification rule: {RuleName}", rule.Name);
            return rule;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding classification rule: {RuleName}", rule.Name);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<ClassificationRule> UpdateClassificationRuleAsync(ClassificationRule rule)
    {
        try
        {
            _logger.LogInformation("Updating classification rule: {RuleName}", rule.Name);

            // Ensure rules are loaded
            await EnsureRulesLoadedAsync();

            // Find the rule
            var existingRule = _rules.FirstOrDefault(r => r.Id == rule.Id);
            if (existingRule == null)
            {
                throw new ArgumentException($"Rule with ID {rule.Id} not found");
            }

            // Update the rule
            var index = _rules.IndexOf(existingRule);
            _rules[index] = rule;

            // Save the rules
            await SaveRulesAsync();

            _logger.LogInformation("Updated classification rule: {RuleName}", rule.Name);
            return rule;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating classification rule: {RuleName}", rule.Name);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> DeleteClassificationRuleAsync(string ruleId)
    {
        try
        {
            _logger.LogInformation("Deleting classification rule: {RuleId}", ruleId);

            // Ensure rules are loaded
            await EnsureRulesLoadedAsync();

            // Find the rule
            var existingRule = _rules.FirstOrDefault(r => r.Id == ruleId);
            if (existingRule == null)
            {
                return false;
            }

            // Remove the rule
            _rules.Remove(existingRule);

            // Save the rules
            await SaveRulesAsync();

            _logger.LogInformation("Deleted classification rule: {RuleId}", ruleId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deleting classification rule: {RuleId}", ruleId);
            return false;
        }
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<string>> GetTagsAsync(string content, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Getting tags for content of length {Length}", content?.Length ?? 0);

            // Classify the content to get tags
            var classification = await ClassifyContentAsync(content, options);

            // Return the tags
            return classification.Tags;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting tags for content");
            return Enumerable.Empty<string>();
        }
    }

    private async Task EnsureRulesLoadedAsync()
    {
        if (_rulesLoaded)
        {
            return;
        }

        try
        {
            _logger.LogInformation("Loading classification rules");

            // Create default rules if the file doesn't exist
            if (!File.Exists(_rulesFilePath))
            {
                _logger.LogInformation("Classification rules file not found, creating default rules");
                _rules.Clear();
                _rules.AddRange(CreateDefaultRules());
                await SaveRulesAsync();
            }
            else
            {
                // Load rules from file
                var json = await File.ReadAllTextAsync(_rulesFilePath);
                var rules = JsonSerializer.Deserialize<List<ClassificationRule>>(json);
                if (rules != null)
                {
                    _rules.Clear();
                    _rules.AddRange(rules);
                }
            }

            _rulesLoaded = true;
            _logger.LogInformation("Loaded {RuleCount} classification rules", _rules.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading classification rules");
            _rules.Clear();
            _rules.AddRange(CreateDefaultRules());
        }
    }

    private async Task SaveRulesAsync()
    {
        try
        {
            _logger.LogInformation("Saving {RuleCount} classification rules", _rules.Count);

            // Create the directory if it doesn't exist
            var directory = Path.GetDirectoryName(_rulesFilePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Save the rules to file
            var json = JsonSerializer.Serialize(_rules, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(_rulesFilePath, json);

            _logger.LogInformation("Saved classification rules to {FilePath}", _rulesFilePath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving classification rules");
        }
    }

    private double CalculateRuleScore(string content, ClassificationRule rule)
    {
        if (string.IsNullOrWhiteSpace(content))
        {
            return 0.0;
        }

        // Calculate keyword score
        double keywordScore = 0.0;
        if (rule.Keywords.Any())
        {
            var keywordMatches = rule.Keywords
                .Select(keyword => Regex.Matches(content, $@"\b{Regex.Escape(keyword)}\b", RegexOptions.IgnoreCase).Count)
                .Sum();
            keywordScore = Math.Min(1.0, keywordMatches / (double)Math.Max(1, rule.Keywords.Count));
        }

        // Calculate pattern score
        double patternScore = 0.0;
        if (rule.Patterns.Any())
        {
            var patternMatches = rule.Patterns
                .Select(pattern => Regex.IsMatch(content, pattern, RegexOptions.IgnoreCase))
                .Count(matched => matched);
            patternScore = patternMatches / (double)rule.Patterns.Count;
        }

        // Combine scores
        double score = rule.Keywords.Any() && rule.Patterns.Any()
            ? (keywordScore + patternScore) / 2.0
            : rule.Keywords.Any() ? keywordScore : patternScore;

        return score;
    }

    private List<ClassificationRule> CreateDefaultRules()
    {
        return new List<ClassificationRule>
        {
            new ClassificationRule
            {
                Name = "Concept Rule",
                Description = "Identifies conceptual explanations",
                Category = ContentCategory.Concept,
                Keywords = new List<string> { "concept", "explanation", "understand", "theory", "principle", "idea", "notion" },
                Patterns = new List<string> { @"what\s+is", @"how\s+does.+work", @"why\s+is" },
                Tags = new List<string> { "concept", "explanation" }
            },
            new ClassificationRule
            {
                Name = "Code Example Rule",
                Description = "Identifies code examples",
                Category = ContentCategory.CodeExample,
                Keywords = new List<string> { "example", "code", "snippet", "implementation", "function", "method", "class" },
                Patterns = new List<string> { @"```[a-zA-Z0-9]*\n[\s\S]*?\n```", @"public\s+class", @"public\s+void", @"function\s+\w+" },
                Tags = new List<string> { "code", "example", "implementation" }
            },
            new ClassificationRule
            {
                Name = "Algorithm Rule",
                Description = "Identifies algorithm descriptions",
                Category = ContentCategory.Algorithm,
                Keywords = new List<string> { "algorithm", "complexity", "time", "space", "O(n)", "O(1)", "O(log n)", "O(n log n)", "O(n²)" },
                Patterns = new List<string> { @"O\([a-zA-Z0-9\s\^]+\)", @"step\s+\d+", @"algorithm\s+works" },
                Tags = new List<string> { "algorithm", "complexity" }
            },
            new ClassificationRule
            {
                Name = "Design Pattern Rule",
                Description = "Identifies design pattern descriptions",
                Category = ContentCategory.DesignPattern,
                Keywords = new List<string> { "pattern", "design pattern", "singleton", "factory", "observer", "strategy", "decorator", "adapter" },
                Patterns = new List<string> { @"design\s+pattern", @"implements\s+the\s+\w+\s+pattern" },
                Tags = new List<string> { "design pattern", "architecture" }
            },
            new ClassificationRule
            {
                Name = "Architecture Rule",
                Description = "Identifies architecture descriptions",
                Category = ContentCategory.Architecture,
                Keywords = new List<string> { "architecture", "system", "component", "module", "service", "layer", "tier", "microservice" },
                Patterns = new List<string> { @"system\s+architecture", @"component\s+diagram", @"service\s+oriented" },
                Tags = new List<string> { "architecture", "system design" }
            },
            new ClassificationRule
            {
                Name = "Best Practice Rule",
                Description = "Identifies best practices",
                Category = ContentCategory.BestPractice,
                Keywords = new List<string> { "best practice", "recommended", "should", "avoid", "prefer", "better", "improve" },
                Patterns = new List<string> { @"best\s+practice", @"should\s+(not\s+)?use", @"recommended\s+approach" },
                Tags = new List<string> { "best practice", "recommendation" }
            },
            new ClassificationRule
            {
                Name = "Tutorial Rule",
                Description = "Identifies tutorials or guides",
                Category = ContentCategory.Tutorial,
                Keywords = new List<string> { "tutorial", "guide", "how to", "step by step", "walkthrough", "learn" },
                Patterns = new List<string> { @"step\s+\d+", @"how\s+to", @"tutorial" },
                Tags = new List<string> { "tutorial", "guide", "how-to" }
            },
            new ClassificationRule
            {
                Name = "API Documentation Rule",
                Description = "Identifies API documentation",
                Category = ContentCategory.ApiDoc,
                Keywords = new List<string> { "api", "endpoint", "parameter", "return", "response", "request", "method", "property" },
                Patterns = new List<string> { @"GET\s+/\w+", @"POST\s+/\w+", @"public\s+\w+\s+\w+\(", @"@param", @"@return" },
                Tags = new List<string> { "api", "documentation" }
            },
            new ClassificationRule
            {
                Name = "Question Rule",
                Description = "Identifies questions",
                Category = ContentCategory.Question,
                Keywords = new List<string> { "question", "ask", "how", "what", "why", "when", "where", "who", "which" },
                Patterns = new List<string> { @"^[^.!?]*\?", @"can\s+you", @"how\s+do\s+I" },
                Tags = new List<string> { "question", "inquiry" }
            },
            new ClassificationRule
            {
                Name = "Answer Rule",
                Description = "Identifies answers",
                Category = ContentCategory.Answer,
                Keywords = new List<string> { "answer", "response", "solution", "result", "output" },
                Patterns = new List<string> { @"^(Yes|No)", @"the\s+answer\s+is", @"to\s+solve\s+this" },
                Tags = new List<string> { "answer", "response" }
            },
            new ClassificationRule
            {
                Name = "Insight Rule",
                Description = "Identifies insights or reflections",
                Category = ContentCategory.Insight,
                Keywords = new List<string> { "insight", "reflection", "learned", "realized", "discovered", "understand", "perspective" },
                Patterns = new List<string> { @"I\s+realized", @"key\s+takeaway", @"important\s+insight" },
                Tags = new List<string> { "insight", "reflection", "learning" }
            },
            new ClassificationRule
            {
                Name = "Problem Rule",
                Description = "Identifies problem descriptions",
                Category = ContentCategory.Problem,
                Keywords = new List<string> { "problem", "issue", "bug", "error", "exception", "fail", "crash", "incorrect" },
                Patterns = new List<string> { @"the\s+problem\s+is", @"error\s+message", @"exception\s+occurred" },
                Tags = new List<string> { "problem", "issue", "bug" }
            },
            new ClassificationRule
            {
                Name = "Solution Rule",
                Description = "Identifies solution descriptions",
                Category = ContentCategory.Solution,
                Keywords = new List<string> { "solution", "fix", "resolve", "solved", "fixed", "workaround", "approach" },
                Patterns = new List<string> { @"the\s+solution\s+is", @"to\s+fix\s+this", @"can\s+be\s+resolved" },
                Tags = new List<string> { "solution", "fix", "resolution" }
            },
            new ClassificationRule
            {
                Name = "Testing Rule",
                Description = "Identifies testing approaches",
                Category = ContentCategory.Testing,
                Keywords = new List<string> { "test", "unit test", "integration test", "e2e test", "assert", "mock", "stub", "verify" },
                Patterns = new List<string> { @"test\s+case", @"assert\.", @"\[Test\]", @"\[Fact\]" },
                Tags = new List<string> { "testing", "verification", "validation" }
            },
            new ClassificationRule
            {
                Name = "Performance Rule",
                Description = "Identifies performance optimizations",
                Category = ContentCategory.Performance,
                Keywords = new List<string> { "performance", "optimize", "efficient", "speed", "fast", "slow", "bottleneck", "latency" },
                Patterns = new List<string> { @"performance\s+issue", @"optimize\s+for", @"reduce\s+latency" },
                Tags = new List<string> { "performance", "optimization", "efficiency" }
            }
        };
    }
}
