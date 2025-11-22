using System.Text.Json;
using System.Text.RegularExpressions;
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
    private readonly List<ClassificationRule> _rules = [];
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
    public ContentClassification ClassifyContent(string content, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Classifying content synchronously of length {Length}", content?.Length ?? 0);

            // Check if we have a cached classification
            var cacheKey = $"{content?.GetHashCode() ?? 0}_{string.Join("_", options?.Select(kv => $"{kv.Key}={kv.Value}") ?? Array.Empty<string>())}";
            if (_classificationCache.TryGetValue(cacheKey, out var cachedClassification))
            {
                _logger.LogInformation("Using cached classification for content");
                return cachedClassification;
            }

            // Ensure rules are loaded
            if (!_rulesLoaded)
            {
                LoadRulesSync();
            }

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
                var primaryCategory = categoryScores.OrderByDescending(kv => kv.Value).First().Key;
                classification.PrimaryCategory = primaryCategory;
                classification.ConfidenceScore = categoryScores[primaryCategory] / categoryScores.Sum(kv => kv.Value);

                // Determine secondary categories
                foreach (var category in categoryScores.OrderByDescending(kv => kv.Value).Skip(1).Take(2).Select(kv => kv.Key))
                {
                    classification.SecondaryCategories.Add(category);
                }
            }
            else
            {
                classification.PrimaryCategory = ContentCategory.Unknown;
                classification.ConfidenceScore = 0.0;
            }

            // Calculate relevance and quality scores
            classification.RelevanceScore = CalculateRelevanceScoreSync(content ?? string.Empty, options);
            classification.QualityScore = CalculateQualityScoreSync(content ?? string.Empty, options);

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
                ClassificationSource = "error",
                Metadata = new Dictionary<string, string>
                {
                    { "Error", ex.Message }
                }
            };
        }
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

            // Move the CPU-bound relevance calculation to a background thread
            return await Task.Run(() =>
            {
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

                var relevanceScore = 0.5; // Default to medium relevance

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
                    var matches = keywords.Count(k => contentLower.Contains(k));
                    relevanceScore = Math.Min(1.0, (double)matches / keywords.Count * 1.5);
                }

                // Ensure the score is between 0 and 1
                relevanceScore = Math.Max(0, Math.Min(1, relevanceScore));

                return relevanceScore;
            });
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

            // Move CPU-bound quality calculations to a background thread
            return await Task.Run(() =>
            {
                if (string.IsNullOrWhiteSpace(content))
                {
                    return 0.0;
                }

                // Calculate basic metrics
                var length = content.Length;
                var lines = content.Split('\n').Length;
                var words = content.Split([' ', '\n', '\r', '\t'], StringSplitOptions.RemoveEmptyEntries).Length;
                var sentences = content.Split(['.', '!', '?'], StringSplitOptions.RemoveEmptyEntries).Length;
                var uniqueWords = content.Split([' ', '\n', '\r', '\t'], StringSplitOptions.RemoveEmptyEntries)
                    .Select(w => w.ToLowerInvariant())
                    .Distinct()
                    .Count();

                // Count structural elements
                var headings = Regex.Matches(content, @"^#+\s+.+$", RegexOptions.Multiline).Count;
                var bulletPoints = Regex.Matches(content, @"^[-*]\s+.+$", RegexOptions.Multiline).Count;
                var codeBlocks = Regex.Matches(content, @"```[\s\S]*?```").Count;

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
            });
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
    public async Task<IEnumerable<string>> GetTagsAsync(string? content, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Getting tags for content of length {Length}", content?.Length ?? 0);

            // Handle null content
            if (string.IsNullOrEmpty(content))
            {
                _logger.LogWarning("Content is null or empty, returning empty tags collection");
                return Array.Empty<string>();
            }

            // Classify the content to get tags
            var classification = await ClassifyContentAsync(content, options);

            // Return the tags
            return classification.Tags;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting tags for content");
            return Array.Empty<string>();
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

    private double CalculateRelevanceScoreSync(string content, Dictionary<string, string>? context = null)
    {
        try
        {
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

            var relevanceScore = 0.5; // Default to medium relevance

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
                var matches = keywords.Count(k => contentLower.Contains(k));
                relevanceScore = Math.Min(1.0, (double)matches / keywords.Count * 1.5);
            }

            // Ensure the score is between 0 and 1
            relevanceScore = Math.Max(0, Math.Min(1, relevanceScore));

            return relevanceScore;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating relevance score");
            return 0.5; // Default to medium relevance
        }
    }

    private double CalculateQualityScoreSync(string content, Dictionary<string, string>? options = null)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return 0.0;
            }

            // Calculate basic metrics
            var length = content.Length;
            var lines = content.Split('\n').Length;
            var words = content.Split([' ', '\n', '\r', '\t'], StringSplitOptions.RemoveEmptyEntries).Length;
            var sentences = content.Split(['.', '!', '?'], StringSplitOptions.RemoveEmptyEntries).Length;
            var uniqueWords = content.Split([' ', '\n', '\r', '\t'], StringSplitOptions.RemoveEmptyEntries)
                .Select(w => w.ToLowerInvariant())
                .Distinct()
                .Count();

            // Count structural elements
            var headings = Regex.Matches(content, @"^#+\s+.+$", RegexOptions.Multiline).Count;
            var bulletPoints = Regex.Matches(content, @"^[-*]\s+.+$", RegexOptions.Multiline).Count;
            var codeBlocks = Regex.Matches(content, @"```[\s\S]*?```").Count;

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

            return qualityScore;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating quality score");
            return 0.5; // Default to medium quality
        }
    }

    private void LoadRulesSync()
    {
        if (_rulesLoaded)
        {
            return;
        }

        try
        {
            _logger.LogInformation("Loading classification rules synchronously");

            // Create default rules if the file doesn't exist
            if (!File.Exists(_rulesFilePath))
            {
                _logger.LogInformation("Classification rules file not found, creating default rules");
                _rules.Clear();
                _rules.AddRange(CreateDefaultRules());
            }
            else
            {
                // Load rules from file
                var json = File.ReadAllText(_rulesFilePath);
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
            _rulesLoaded = true;
        }
    }

    private double CalculateRuleScore(string content, ClassificationRule rule)
    {
        if (string.IsNullOrWhiteSpace(content))
        {
            return 0.0;
        }

        // Calculate keyword score
        var keywordScore = 0.0;
        if (rule.Keywords.Any())
        {
            var keywordMatches = rule.Keywords
                .Select(keyword => Regex.Matches(content, $@"\b{Regex.Escape(keyword)}\b", RegexOptions.IgnoreCase).Count)
                .Sum();
            keywordScore = Math.Min(1.0, keywordMatches / (double)Math.Max(1, rule.Keywords.Count));
        }

        // Calculate pattern score
        var patternScore = 0.0;
        if (rule.Patterns.Any())
        {
            var patternMatches = rule.Patterns
                .Select(pattern => Regex.IsMatch(content, pattern, RegexOptions.IgnoreCase))
                .Count(matched => matched);
            patternScore = patternMatches / (double)rule.Patterns.Count;
        }

        // Combine scores
        var score = rule.Keywords.Any() && rule.Patterns.Any()
            ? (keywordScore + patternScore) / 2.0
            : rule.Keywords.Any() ? keywordScore : patternScore;

        return score;
    }

    private List<ClassificationRule> CreateDefaultRules()
    {
        return
        [
            new ClassificationRule
            {
                Name = "Concept Rule",
                Description = "Identifies conceptual explanations",
                Category = ContentCategory.Concept,
                Keywords = ["concept", "explanation", "understand", "theory", "principle", "idea", "notion"],
                Patterns = [@"what\s+is", @"how\s+does.+work", @"why\s+is"],
                Tags = ["concept", "explanation"]
            },

            new ClassificationRule
            {
                Name = "Code Example Rule",
                Description = "Identifies code examples",
                Category = ContentCategory.CodeExample,
                Keywords = ["example", "code", "snippet", "implementation", "function", "method", "class"],
                Patterns = [@"```[a-zA-Z0-9]*\n[\s\S]*?\n```", @"public\s+class", @"public\s+void", @"function\s+\w+"],
                Tags = ["code", "example", "implementation"]
            },

            new ClassificationRule
            {
                Name = "Algorithm Rule",
                Description = "Identifies algorithm descriptions",
                Category = ContentCategory.Algorithm,
                Keywords =
                [
                    "algorithm", "complexity", "time", "space", "O(n)", "O(1)", "O(log n)", "O(n log n)", "O(n²)"
                ],
                Patterns = [@"O\([a-zA-Z0-9\s\^]+\)", @"step\s+\d+", @"algorithm\s+works"],
                Tags = ["algorithm", "complexity"]
            },

            new ClassificationRule
            {
                Name = "Design Pattern Rule",
                Description = "Identifies design pattern descriptions",
                Category = ContentCategory.DesignPattern,
                Keywords =
                [
                    "pattern", "design pattern", "singleton", "factory", "observer", "strategy", "decorator", "adapter"
                ],
                Patterns = [@"design\s+pattern", @"implements\s+the\s+\w+\s+pattern"],
                Tags = ["design pattern", "architecture"]
            },

            new ClassificationRule
            {
                Name = "Architecture Rule",
                Description = "Identifies architecture descriptions",
                Category = ContentCategory.Architecture,
                Keywords =
                    ["architecture", "system", "component", "module", "service", "layer", "tier", "microservice"],
                Patterns = [@"system\s+architecture", @"component\s+diagram", @"service\s+oriented"],
                Tags = ["architecture", "system design"]
            },

            new ClassificationRule
            {
                Name = "Best Practice Rule",
                Description = "Identifies best practices",
                Category = ContentCategory.BestPractice,
                Keywords = ["best practice", "recommended", "should", "avoid", "prefer", "better", "improve"],
                Patterns = [@"best\s+practice", @"should\s+(not\s+)?use", @"recommended\s+approach"],
                Tags = ["best practice", "recommendation"]
            },

            new ClassificationRule
            {
                Name = "Tutorial Rule",
                Description = "Identifies tutorials or guides",
                Category = ContentCategory.Tutorial,
                Keywords = ["tutorial", "guide", "how to", "step by step", "walkthrough", "learn"],
                Patterns = [@"step\s+\d+", @"how\s+to", @"tutorial"],
                Tags = ["tutorial", "guide", "how-to"]
            },

            new ClassificationRule
            {
                Name = "API Documentation Rule",
                Description = "Identifies API documentation",
                Category = ContentCategory.ApiDoc,
                Keywords = ["api", "endpoint", "parameter", "return", "response", "request", "method", "property"],
                Patterns = [@"GET\s+/\w+", @"POST\s+/\w+", @"public\s+\w+\s+\w+\(", @"@param", @"@return"],
                Tags = ["api", "documentation"]
            },

            new ClassificationRule
            {
                Name = "Question Rule",
                Description = "Identifies questions",
                Category = ContentCategory.Question,
                Keywords =
                    ["question", "ask", "how", "what", "why", "when", "where", "who", "which"],
                Patterns = [@"^[^.!?]*\?", @"can\s+you", @"how\s+do\s+I"],
                Tags = ["question", "inquiry"]
            },

            new ClassificationRule
            {
                Name = "Answer Rule",
                Description = "Identifies answers",
                Category = ContentCategory.Answer,
                Keywords = ["answer", "response", "solution", "result", "output"],
                Patterns = [@"^(Yes|No)", @"the\s+answer\s+is", @"to\s+solve\s+this"],
                Tags = ["answer", "response"]
            },

            new ClassificationRule
            {
                Name = "Insight Rule",
                Description = "Identifies insights or reflections",
                Category = ContentCategory.Insight,
                Keywords = ["insight", "reflection", "learned", "realized", "discovered", "understand", "perspective"],
                Patterns = [@"I\s+realized", @"key\s+takeaway", @"important\s+insight"],
                Tags = ["insight", "reflection", "learning"]
            },

            new ClassificationRule
            {
                Name = "Problem Rule",
                Description = "Identifies problem descriptions",
                Category = ContentCategory.Problem,
                Keywords = ["problem", "issue", "bug", "error", "exception", "fail", "crash", "incorrect"],
                Patterns = [@"the\s+problem\s+is", @"error\s+message", @"exception\s+occurred"],
                Tags = ["problem", "issue", "bug"]
            },

            new ClassificationRule
            {
                Name = "Solution Rule",
                Description = "Identifies solution descriptions",
                Category = ContentCategory.Solution,
                Keywords = ["solution", "fix", "resolve", "solved", "fixed", "workaround", "approach"],
                Patterns = [@"the\s+solution\s+is", @"to\s+fix\s+this", @"can\s+be\s+resolved"],
                Tags = ["solution", "fix", "resolution"]
            },

            new ClassificationRule
            {
                Name = "Testing Rule",
                Description = "Identifies testing approaches",
                Category = ContentCategory.Testing,
                Keywords = ["test", "unit test", "integration test", "e2e test", "assert", "mock", "stub", "verify"],
                Patterns = [@"test\s+case", @"assert\.", @"\[Test\]", @"\[Fact\]"],
                Tags = ["testing", "verification", "validation"]
            },

            new ClassificationRule
            {
                Name = "Performance Rule",
                Description = "Identifies performance optimizations",
                Category = ContentCategory.Performance,
                Keywords = ["performance", "optimize", "efficient", "speed", "fast", "slow", "bottleneck", "latency"],
                Patterns = [@"performance\s+issue", @"optimize\s+for", @"reduce\s+latency"],
                Tags = ["performance", "optimization", "efficiency"]
            }
        ];
    }
}
