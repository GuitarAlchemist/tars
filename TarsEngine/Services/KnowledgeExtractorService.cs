using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for extracting knowledge from content
/// </summary>
public class KnowledgeExtractorService : IKnowledgeExtractorService
{
    private readonly ILogger<KnowledgeExtractorService> _logger;
    private readonly IDocumentParserService _documentParserService;
    private readonly IContentClassifierService _contentClassifierService;
    private readonly List<TarsEngine.Models.KnowledgeValidationRule> _validationRules = [];

    // Regular expressions for pattern extraction
    private static readonly Regex _codePatternRegex = new(@"(?:public|private|protected|internal|static)?\s+(?:class|interface|struct|enum|record)\s+(\w+)", RegexOptions.Compiled);
    private static readonly Regex _functionPatternRegex = new(@"(?:public|private|protected|internal|static)?\s+\w+\s+(\w+)\s*\(", RegexOptions.Compiled);
    private static readonly Regex _conceptRegex = new(@"(?:is|are|refers to|means|defined as)\s+(?:a|an|the)?\s+([^.!?]+)", RegexOptions.Compiled);
    private static readonly Regex _insightRegex = new(@"(?:realized|discovered|found|learned|understood)\s+that\s+([^.!?]+)", RegexOptions.Compiled);

    /// <summary>
    /// Initializes a new instance of the <see cref="KnowledgeExtractorService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="documentParserService">The document parser service</param>
    /// <param name="contentClassifierService">The content classifier service</param>
    public KnowledgeExtractorService(
        ILogger<KnowledgeExtractorService> logger,
        IDocumentParserService documentParserService,
        IContentClassifierService contentClassifierService)
    {
        _logger = logger;
        _documentParserService = documentParserService;
        _contentClassifierService = contentClassifierService;
        InitializeValidationRules();
    }

    /// <inheritdoc/>
    public async Task<KnowledgeExtractionResult> ExtractFromTextAsync(string content, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Extracting knowledge from text content of length {Length}", content?.Length ?? 0);

            if (string.IsNullOrWhiteSpace(content))
            {
                return new KnowledgeExtractionResult
                {
                    Source = "text",
                    Errors = { "Content is empty or whitespace" }
                };
            }

            var result = new KnowledgeExtractionResult
            {
                Source = "text",
                Metadata = new Dictionary<string, string>
                {
                    { "ContentLength", content.Length.ToString() },
                    { "ContentType", "Text" }
                }
            };

            // Extract knowledge items from text
            var items = new List<TarsEngine.Models.KnowledgeItem>();

            // Extract concepts
            var concepts = ExtractConcepts(content);
            items.AddRange(concepts);

            // Extract insights
            var insights = ExtractInsights(content);
            items.AddRange(insights);

            // Add options to metadata
            if (options != null)
            {
                foreach (var option in options)
                {
                    result.Metadata[option.Key] = option.Value;
                }
            }

            result.Items = items;

            _logger.LogInformation("Extracted {ItemCount} knowledge items from text", items.Count);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting knowledge from text");
            return new KnowledgeExtractionResult
            {
                Source = "text",
                Errors = { $"Error extracting knowledge from text: {ex.Message}" }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<KnowledgeExtractionResult> ExtractFromDocumentAsync(DocumentParsingResult document, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Extracting knowledge from document: {DocumentPath}", document.DocumentPath);

            var result = new KnowledgeExtractionResult
            {
                Source = document.DocumentPath,
                Metadata = new Dictionary<string, string>
                {
                    { "DocumentType", document.DocumentType.ToString() },
                    { "Title", document.Title }
                }
            };

            // Extract knowledge items from each section
            var items = new List<TarsEngine.Models.KnowledgeItem>();
            foreach (var section in document.Sections)
            {
                var sectionItems = await ExtractFromSectionAsync(section, document);
                items.AddRange(sectionItems);
            }

            // Add options to metadata
            if (options != null)
            {
                foreach (var option in options)
                {
                    result.Metadata[option.Key] = option.Value;
                }
            }

            result.Items = items;

            _logger.LogInformation("Extracted {ItemCount} knowledge items from document", items.Count);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting knowledge from document: {DocumentPath}", document.DocumentPath);
            return new KnowledgeExtractionResult
            {
                Source = document.DocumentPath,
                Errors = { $"Error extracting knowledge from document: {ex.Message}" }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<KnowledgeExtractionResult> ExtractFromCodeAsync(string code, string language, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Extracting knowledge from code of language: {Language}", language);

            if (string.IsNullOrWhiteSpace(code))
            {
                return new KnowledgeExtractionResult
                {
                    Source = "code",
                    Errors = { "Code is empty or whitespace" }
                };
            }

            var result = new KnowledgeExtractionResult
            {
                Source = "code",
                Metadata = new Dictionary<string, string>
                {
                    { "Language", language },
                    { "CodeLength", code.Length.ToString() }
                }
            };

            // Extract knowledge items from code
            var items = new List<TarsEngine.Models.KnowledgeItem>();

            // Extract code patterns based on language
            var codePatterns = ExtractCodePatterns(code, language);
            items.AddRange(codePatterns);

            // Add options to metadata
            if (options != null)
            {
                foreach (var option in options)
                {
                    result.Metadata[option.Key] = option.Value;
                }
            }

            result.Items = items;

            _logger.LogInformation("Extracted {ItemCount} knowledge items from code", items.Count);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting knowledge from code");
            return new KnowledgeExtractionResult
            {
                Source = "code",
                Errors = { $"Error extracting knowledge from code: {ex.Message}" }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<KnowledgeExtractionResult> ExtractFromClassificationAsync(ContentClassification classification, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Extracting knowledge from classification: {Category}", classification.PrimaryCategory);

            var result = new KnowledgeExtractionResult
            {
                Source = "classification",
                Metadata = new Dictionary<string, string>
                {
                    { "Category", classification.PrimaryCategory.ToString() },
                    { "Confidence", classification.ConfidenceScore.ToString() },
                    { "Relevance", classification.RelevanceScore.ToString() },
                    { "Quality", classification.QualityScore.ToString() }
                }
            };

            // Extract knowledge items based on classification category
            var items = new List<TarsEngine.Models.KnowledgeItem>();
            var item = CreateKnowledgeItemFromClassification(classification);
            if (item != null)
            {
                items.Add(item);
            }

            // Add options to metadata
            if (options != null)
            {
                foreach (var option in options)
                {
                    result.Metadata[option.Key] = option.Value;
                }
            }

            result.Items = items;

            _logger.LogInformation("Extracted {ItemCount} knowledge items from classification", items.Count);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting knowledge from classification");
            return new KnowledgeExtractionResult
            {
                Source = "classification",
                Errors = { $"Error extracting knowledge from classification: {ex.Message}" }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<KnowledgeExtractionResult> ExtractFromClassificationBatchAsync(ContentClassificationBatch batch, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Extracting knowledge from classification batch with {Count} classifications", batch.Classifications.Count);

            var result = new KnowledgeExtractionResult
            {
                Source = batch.Source,
                Metadata = new Dictionary<string, string>
                {
                    { "BatchId", batch.Id },
                    { "ClassificationCount", batch.Classifications.Count.ToString() }
                }
            };

            // Extract knowledge items from each classification
            var items = new List<TarsEngine.Models.KnowledgeItem>();
            foreach (var classification in batch.Classifications)
            {
                var extractionResult = await ExtractFromClassificationAsync(classification, options);
                items.AddRange(extractionResult.Items);
            }

            // Add batch metadata to result
            foreach (var meta in batch.Metadata)
            {
                result.Metadata[meta.Key] = meta.Value;
            }

            // Add options to metadata
            if (options != null)
            {
                foreach (var option in options)
                {
                    result.Metadata[option.Key] = option.Value;
                }
            }

            result.Items = items;

            _logger.LogInformation("Extracted {ItemCount} knowledge items from classification batch", items.Count);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting knowledge from classification batch");
            return new KnowledgeExtractionResult
            {
                Source = batch.Source,
                Errors = { $"Error extracting knowledge from classification batch: {ex.Message}" }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<TarsEngine.Models.KnowledgeValidationResult> ValidateKnowledgeItemAsync(TarsEngine.Models.KnowledgeItem item, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Validating knowledge item: {ItemId}", item.Id);

            var result = new TarsEngine.Models.KnowledgeValidationResult
            {
                Item = item,
                IsValid = true
            };

            // Apply validation rules
            foreach (var rule in _validationRules.Where(r => r.IsEnabled && (r.ApplicableTypes.Contains(item.Type) || !r.ApplicableTypes.Any())))
            {
                var isValid = ValidateAgainstRule(item, rule);
                if (!isValid)
                {
                    result.IsValid = false;
                    result.Issues.Add(new TarsEngine.Models.ValidationIssue
                    {
                        RuleId = rule.Id,
                        RuleName = rule.Name,
                        Message = rule.ErrorMessage,
                        Severity = rule.Severity
                    });
                }
            }

            _logger.LogInformation("Validated knowledge item {ItemId}: {IsValid}", item.Id, result.IsValid);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating knowledge item: {ItemId}", item.Id);
            return new KnowledgeValidationResult
            {
                Item = item,
                IsValid = false,
                Issues =
                [
                    new TarsEngine.Models.ValidationIssue
                    {
                        RuleId = "error",
                        RuleName = "Error",
                        Message = $"Error validating knowledge item: {ex.Message}",
                        Severity = ValidationSeverity.Error
                    }
                ]
            };
        }
    }

    /// <inheritdoc/>
    public async Task<List<TarsEngine.Models.KnowledgeRelationship>> DetectRelationshipsAsync(List<TarsEngine.Models.KnowledgeItem> items, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Detecting relationships between {ItemCount} knowledge items", items.Count);

            var relationships = new List<TarsEngine.Models.KnowledgeRelationship>();

            // Skip if there are too few items
            if (items.Count < 2)
            {
                return relationships;
            }

            // Detect relationships based on content similarity
            for (int i = 0; i < items.Count; i++)
            {
                for (int j = i + 1; j < items.Count; j++)
                {
                    var item1 = items[i];
                    var item2 = items[j];

                    // Skip if items are the same
                    if (item1.Id == item2.Id)
                    {
                        continue;
                    }

                    // Detect relationship type and strength
                    var (type, strength) = DetectRelationship(item1, item2);
                    if (type != RelationshipType.Unknown && strength > 0.3)
                    {
                        relationships.Add(new KnowledgeRelationship
                        {
                            SourceId = item1.Id,
                            TargetId = item2.Id,
                            Type = type,
                            Strength = strength,
                            Description = $"{item1.Type} {type.ToString().ToLowerInvariant()} {item2.Type}"
                        });
                    }
                }
            }

            _logger.LogInformation("Detected {RelationshipCount} relationships between knowledge items", relationships.Count);
            return relationships;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting relationships between knowledge items");
            return [];
        }
    }

    /// <inheritdoc/>
    public async Task<List<KnowledgeValidationRule>> GetValidationRulesAsync()
    {
        return _validationRules;
    }

    /// <inheritdoc/>
    public async Task<KnowledgeValidationRule> AddValidationRuleAsync(KnowledgeValidationRule rule)
    {
        try
        {
            _logger.LogInformation("Adding validation rule: {RuleName}", rule.Name);

            // Add the rule
            _validationRules.Add(rule);

            _logger.LogInformation("Added validation rule: {RuleName}", rule.Name);
            return rule;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding validation rule: {RuleName}", rule.Name);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<KnowledgeValidationRule> UpdateValidationRuleAsync(KnowledgeValidationRule rule)
    {
        try
        {
            _logger.LogInformation("Updating validation rule: {RuleName}", rule.Name);

            // Find the rule
            var existingRule = _validationRules.FirstOrDefault(r => r.Id == rule.Id);
            if (existingRule == null)
            {
                throw new ArgumentException($"Rule with ID {rule.Id} not found");
            }

            // Update the rule
            var index = _validationRules.IndexOf(existingRule);
            _validationRules[index] = rule;

            _logger.LogInformation("Updated validation rule: {RuleName}", rule.Name);
            return rule;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating validation rule: {RuleName}", rule.Name);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> DeleteValidationRuleAsync(string ruleId)
    {
        try
        {
            _logger.LogInformation("Deleting validation rule: {RuleId}", ruleId);

            // Find the rule
            var existingRule = _validationRules.FirstOrDefault(r => r.Id == ruleId);
            if (existingRule == null)
            {
                return false;
            }

            // Remove the rule
            _validationRules.Remove(existingRule);

            _logger.LogInformation("Deleted validation rule: {RuleId}", ruleId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deleting validation rule: {RuleId}", ruleId);
            return false;
        }
    }

    private async Task<List<TarsEngine.Models.KnowledgeItem>> ExtractFromSectionAsync(ContentSection section, DocumentParsingResult document)
    {
        var items = new List<TarsEngine.Models.KnowledgeItem>();

        // Extract based on content type
        switch (section.ContentType)
        {
            case ContentType.Text:
                var textItems = await ExtractFromTextAsync(section.RawContent);
                items.AddRange(textItems.Items);
                break;

            case ContentType.Code:
                foreach (var codeBlock in section.CodeBlocks)
                {
                    var codeItems = await ExtractFromCodeAsync(codeBlock.Code, codeBlock.Language);
                    items.AddRange(codeItems.Items);
                }
                break;

            case ContentType.Concept:
                var conceptItems = ExtractConcepts(section.RawContent);
                items.AddRange(conceptItems);
                break;

            case ContentType.Insight:
                var insightItems = ExtractInsights(section.RawContent);
                items.AddRange(insightItems);
                break;

            case ContentType.Question:
                var questionItem = new TarsEngine.Models.KnowledgeItem
                {
                    Type = TarsEngine.Models.KnowledgeType.Question,
                    Content = section.RawContent,
                    Source = document.DocumentPath,
                    Context = section.Heading,
                    Confidence = 0.9,
                    Relevance = 0.7,
                    Tags = { "question" }
                };
                items.Add(questionItem);
                break;

            case ContentType.Answer:
                var answerItem = new TarsEngine.Models.KnowledgeItem
                {
                    Type = TarsEngine.Models.KnowledgeType.Answer,
                    Content = section.RawContent,
                    Source = document.DocumentPath,
                    Context = section.Heading,
                    Confidence = 0.9,
                    Relevance = 0.8,
                    Tags = { "answer" }
                };
                items.Add(answerItem);
                break;

            case ContentType.Example:
                var exampleItem = new TarsEngine.Models.KnowledgeItem
                {
                    Type = TarsEngine.Models.KnowledgeType.CodePattern,
                    Content = section.RawContent,
                    Source = document.DocumentPath,
                    Context = section.Heading,
                    Confidence = 0.9,
                    Relevance = 0.8,
                    Tags = { "example" }
                };
                items.Add(exampleItem);
                break;
        }

        // Extract from code blocks in any section type
        foreach (var codeBlock in section.CodeBlocks)
        {
            var codeItems = await ExtractFromCodeAsync(codeBlock.Code, codeBlock.Language);
            items.AddRange(codeItems.Items);
        }

        // Set source and context for all items
        foreach (var item in items)
        {
            if (string.IsNullOrEmpty(item.Source))
            {
                item.Source = document.DocumentPath;
            }
            if (string.IsNullOrEmpty(item.Context))
            {
                item.Context = section.Heading;
            }
        }

        return items;
    }

    private List<TarsEngine.Models.KnowledgeItem> ExtractConcepts(string content)
    {
        var concepts = new List<TarsEngine.Models.KnowledgeItem>();

        // Extract concepts using regex
        var matches = _conceptRegex.Matches(content);
        foreach (Match match in matches)
        {
            if (match.Groups.Count > 1)
            {
                var conceptText = match.Groups[1].Value.Trim();
                if (!string.IsNullOrWhiteSpace(conceptText))
                {
                    var item = new TarsEngine.Models.KnowledgeItem
                    {
                        Type = TarsEngine.Models.KnowledgeType.Concept,
                        Content = conceptText,
                        Confidence = 0.8,
                        Relevance = 0.7,
                        Tags = { "concept", "definition" }
                    };
                    concepts.Add(item);
                }
            }
        }

        return concepts;
    }

    private List<TarsEngine.Models.KnowledgeItem> ExtractInsights(string content)
    {
        var insights = new List<TarsEngine.Models.KnowledgeItem>();

        // Extract insights using regex
        var matches = _insightRegex.Matches(content);
        foreach (Match match in matches)
        {
            if (match.Groups.Count > 1)
            {
                var insightText = match.Groups[1].Value.Trim();
                if (!string.IsNullOrWhiteSpace(insightText))
                {
                    var item = new TarsEngine.Models.KnowledgeItem
                    {
                        Type = TarsEngine.Models.KnowledgeType.Insight,
                        Content = insightText,
                        Confidence = 0.7,
                        Relevance = 0.6,
                        Tags = { "insight", "reflection" }
                    };
                    insights.Add(item);
                }
            }
        }

        return insights;
    }

    private List<TarsEngine.Models.KnowledgeItem> ExtractCodePatterns(string code, string language)
    {
        var patterns = new List<TarsEngine.Models.KnowledgeItem>();

        // Extract class/interface definitions
        var classMatches = _codePatternRegex.Matches(code);
        foreach (Match match in classMatches)
        {
            if (match.Groups.Count > 1)
            {
                var className = match.Groups[1].Value.Trim();
                if (!string.IsNullOrWhiteSpace(className))
                {
                    var item = new TarsEngine.Models.KnowledgeItem
                    {
                        Type = TarsEngine.Models.KnowledgeType.CodePattern,
                        Content = $"Class/Interface: {className}",
                        Confidence = 0.9,
                        Relevance = 0.8,
                        Tags = { "code", "class", language }
                    };
                    patterns.Add(item);
                }
            }
        }

        // Extract function/method definitions
        var functionMatches = _functionPatternRegex.Matches(code);
        foreach (Match match in functionMatches)
        {
            if (match.Groups.Count > 1)
            {
                var functionName = match.Groups[1].Value.Trim();
                if (!string.IsNullOrWhiteSpace(functionName))
                {
                    var item = new TarsEngine.Models.KnowledgeItem
                    {
                        Type = TarsEngine.Models.KnowledgeType.CodePattern,
                        Content = $"Function/Method: {functionName}",
                        Confidence = 0.9,
                        Relevance = 0.7,
                        Tags = { "code", "function", language }
                    };
                    patterns.Add(item);
                }
            }
        }

        return patterns;
    }

    private TarsEngine.Models.KnowledgeItem? CreateKnowledgeItemFromClassification(ContentClassification classification)
    {
        // Skip if content is empty
        if (string.IsNullOrWhiteSpace(classification.Content))
        {
            return null;
        }

        // Map classification category to knowledge type
        var knowledgeType = classification.PrimaryCategory switch
        {
            ContentCategory.Concept => TarsEngine.Models.KnowledgeType.Concept,
            ContentCategory.CodeExample => TarsEngine.Models.KnowledgeType.CodePattern,
            ContentCategory.Algorithm => TarsEngine.Models.KnowledgeType.Algorithm,
            ContentCategory.DesignPattern => TarsEngine.Models.KnowledgeType.DesignPattern,
            ContentCategory.BestPractice => TarsEngine.Models.KnowledgeType.BestPractice,
            ContentCategory.ApiDoc => TarsEngine.Models.KnowledgeType.ApiUsage,
            ContentCategory.Question => TarsEngine.Models.KnowledgeType.Question,
            ContentCategory.Answer => TarsEngine.Models.KnowledgeType.Answer,
            ContentCategory.Insight => TarsEngine.Models.KnowledgeType.Insight,
            ContentCategory.Performance => TarsEngine.Models.KnowledgeType.Performance,
            ContentCategory.Security => TarsEngine.Models.KnowledgeType.Security,
            ContentCategory.Testing => TarsEngine.Models.KnowledgeType.Testing,
            _ => TarsEngine.Models.KnowledgeType.Unknown
        };

        // Skip unknown types
        if (knowledgeType == TarsEngine.Models.KnowledgeType.Unknown)
        {
            return null;
        }

        // Create knowledge item
        var item = new TarsEngine.Models.KnowledgeItem
        {
            Type = knowledgeType,
            Content = classification.Content,
            Confidence = classification.ConfidenceScore,
            Relevance = classification.RelevanceScore,
            Tags =
            [
                ..classification.Tags,
                classification.PrimaryCategory.ToString().ToLowerInvariant()
            ]
        };

        // Add tag for the primary category

        return item;
    }

    private (RelationshipType Type, double Strength) DetectRelationship(TarsEngine.Models.KnowledgeItem item1, TarsEngine.Models.KnowledgeItem item2)
    {
        // Check for same type
        if (item1.Type == item2.Type)
        {
            // Calculate content similarity
            var similarity = CalculateContentSimilarity(item1.Content, item2.Content);
            if (similarity > 0.7)
            {
                return (RelationshipType.IsSimilarTo, similarity);
            }
        }

        // Check for question-answer relationship
        if (item1.Type == TarsEngine.Models.KnowledgeType.Question && item2.Type == TarsEngine.Models.KnowledgeType.Answer)
        {
            return (RelationshipType.Answers, 0.9);
        }
        if (item1.Type == TarsEngine.Models.KnowledgeType.Answer && item2.Type == TarsEngine.Models.KnowledgeType.Question)
        {
            return (RelationshipType.Questions, 0.9);
        }

        // Check for implementation relationship
        if (item1.Type == TarsEngine.Models.KnowledgeType.CodePattern && item2.Type == TarsEngine.Models.KnowledgeType.Algorithm)
        {
            return (RelationshipType.Implements, 0.8);
        }
        if (item1.Type == TarsEngine.Models.KnowledgeType.Algorithm && item2.Type == TarsEngine.Models.KnowledgeType.CodePattern)
        {
            return (RelationshipType.IsImplementedBy, 0.8);
        }

        // Check for concept-example relationship
        if (item1.Type == TarsEngine.Models.KnowledgeType.Concept && item2.Type == TarsEngine.Models.KnowledgeType.CodePattern)
        {
            return (RelationshipType.IsImplementedBy, 0.7);
        }
        if (item1.Type == TarsEngine.Models.KnowledgeType.CodePattern && item2.Type == TarsEngine.Models.KnowledgeType.Concept)
        {
            return (RelationshipType.Implements, 0.7);
        }

        // Default to related with weak strength
        return (RelationshipType.RelatedTo, 0.3);
    }

    private double CalculateContentSimilarity(string content1, string content2)
    {
        // Simple word overlap similarity
        var words1 = content1.ToLowerInvariant().Split([' ', '\t', '\n', '\r', '.', ',', ';', ':', '!', '?'], StringSplitOptions.RemoveEmptyEntries);
        var words2 = content2.ToLowerInvariant().Split([' ', '\t', '\n', '\r', '.', ',', ';', ':', '!', '?'], StringSplitOptions.RemoveEmptyEntries);

        var set1 = new HashSet<string>(words1);
        var set2 = new HashSet<string>(words2);

        var intersection = set1.Intersect(set2).Count();
        var union = set1.Union(set2).Count();

        return union > 0 ? (double)intersection / union : 0;
    }

    private bool ValidateAgainstRule(TarsEngine.Models.KnowledgeItem item, TarsEngine.Models.KnowledgeValidationRule rule)
    {
        // Simple validation based on content length
        if (rule.ValidationCriteria.Contains("MinLength"))
        {
            var minLength = int.Parse(rule.ValidationCriteria.Split('=')[1]);
            return item.Content.Length >= minLength;
        }

        // Validation based on confidence score
        if (rule.ValidationCriteria.Contains("MinConfidence"))
        {
            var minConfidence = double.Parse(rule.ValidationCriteria.Split('=')[1]);
            return item.Confidence >= minConfidence;
        }

        // Validation based on relevance score
        if (rule.ValidationCriteria.Contains("MinRelevance"))
        {
            var minRelevance = double.Parse(rule.ValidationCriteria.Split('=')[1]);
            return item.Relevance >= minRelevance;
        }

        // Default to valid
        return true;
    }

    private void InitializeValidationRules()
    {
        _validationRules.Add(new KnowledgeValidationRule
        {
            Name = "Minimum Content Length",
            Description = "Ensures that knowledge items have a minimum content length",
            ApplicableTypes =
            [
                TarsEngine.Models.KnowledgeType.Concept, TarsEngine.Models.KnowledgeType.Insight,
                TarsEngine.Models.KnowledgeType.BestPractice
            ],
            ValidationCriteria = "MinLength=10",
            ErrorMessage = "Content is too short (minimum 10 characters)",
            Severity = ValidationSeverity.Warning
        });

        _validationRules.Add(new KnowledgeValidationRule
        {
            Name = "Minimum Confidence",
            Description = "Ensures that knowledge items have a minimum confidence score",
            ApplicableTypes = [],
            ValidationCriteria = "MinConfidence=0.5",
            ErrorMessage = "Confidence score is too low (minimum 0.5)",
            Severity = ValidationSeverity.Warning
        });

        _validationRules.Add(new KnowledgeValidationRule
        {
            Name = "Minimum Relevance",
            Description = "Ensures that knowledge items have a minimum relevance score",
            ApplicableTypes = [],
            ValidationCriteria = "MinRelevance=0.3",
            ErrorMessage = "Relevance score is too low (minimum 0.3)",
            Severity = ValidationSeverity.Info
        });
    }
}
