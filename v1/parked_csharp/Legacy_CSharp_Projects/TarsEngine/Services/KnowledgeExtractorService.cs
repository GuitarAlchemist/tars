using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

using ModelKnowledgeItem = TarsEngine.Models.KnowledgeItem;
using ModelKnowledgeType = TarsEngine.Models.KnowledgeType;

namespace TarsEngine.Services;

/// <summary>
/// Service for extracting knowledge from content
/// </summary>
public class KnowledgeExtractorService : IKnowledgeExtractorService
{
    private readonly ILogger<KnowledgeExtractorService> _logger;
    private readonly IDocumentParserService _documentParserService;
    private readonly IContentClassifierService _contentClassifierService;
    private readonly List<KnowledgeValidationRule> _validationRules = [];

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
            var items = new List<ModelKnowledgeItem>();

            // Extract concepts and insights
            var concepts = await ExtractConceptsAsync(content);
            var insights = await ExtractInsightsAsync(content);

            items.AddRange(concepts);
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

            // Move CPU-intensive document processing to a background thread
            return await Task.Run(async () =>
            {
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
            });
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
    public Task<KnowledgeExtractionResult> ExtractFromCodeAsync(string code, string language, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Extracting knowledge from code of language: {Language}", language);

            if (string.IsNullOrWhiteSpace(code))
            {
                return Task.FromResult(new KnowledgeExtractionResult
                {
                    Source = "code",
                    Errors = { "Code is empty or whitespace" }
                });
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
            return Task.FromResult(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting knowledge from code");
            return Task.FromResult(new KnowledgeExtractionResult
            {
                Source = "code",
                Errors = { $"Error extracting knowledge from code: {ex.Message}" }
            });
        }
    }

    /// <inheritdoc/>
    public Task<KnowledgeExtractionResult> ExtractFromClassificationAsync(ContentClassification classification, Dictionary<string, string>? options = null)
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
            return Task.FromResult(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting knowledge from classification");
            return Task.FromResult(new KnowledgeExtractionResult
            {
                Source = "classification",
                Errors = { $"Error extracting knowledge from classification: {ex.Message}" }
            });
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
    public Task<KnowledgeValidationResult> ValidateKnowledgeItemAsync(TarsEngine.Models.KnowledgeItem item, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Validating knowledge item: {ItemId}", item.Id);

            var result = new KnowledgeValidationResult
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
            return Task.FromResult(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating knowledge item: {ItemId}", item.Id);
            return Task.FromResult(new KnowledgeValidationResult
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
            });
        }
    }

    /// <inheritdoc/>
    public Task<List<KnowledgeRelationship>> DetectRelationshipsAsync(List<ModelKnowledgeItem> items, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Detecting relationships between {ItemCount} knowledge items", items.Count);

            var relationships = new List<KnowledgeRelationship>();

            // Skip if there are too few items
            if (items.Count < 2)
            {
                return Task.FromResult(relationships);
            }

            // Detect relationships based on content similarity
            for (var i = 0; i < items.Count; i++)
            {
                for (var j = i + 1; j < items.Count; j++)
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
            return Task.FromResult(relationships);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting relationships between knowledge items");
            return Task.FromResult(new List<KnowledgeRelationship>());
        }
    }

    /// <inheritdoc/>
    public Task<List<KnowledgeValidationRule>> GetValidationRulesAsync()
    {
        return Task.FromResult(_validationRules);
    }

    /// <inheritdoc/>
    public Task<KnowledgeValidationRule> AddValidationRuleAsync(KnowledgeValidationRule rule)
    {
        _logger.LogInformation("Adding validation rule: {RuleName}", rule.Name);
        _validationRules.Add(rule);
        return Task.FromResult(rule);
    }

    /// <inheritdoc/>
    public Task<KnowledgeValidationRule> UpdateValidationRuleAsync(KnowledgeValidationRule rule)
    {
        _logger.LogInformation("Updating validation rule: {RuleName}", rule.Name);
        var existingRule = _validationRules.FirstOrDefault(r => r.Id == rule.Id)
            ?? throw new ArgumentException($"Rule with ID {rule.Id} not found");

        var index = _validationRules.IndexOf(existingRule);
        _validationRules[index] = rule;
        return Task.FromResult(rule);
    }

    /// <inheritdoc/>
    public Task<bool> DeleteValidationRuleAsync(string ruleId)
    {
        _logger.LogInformation("Deleting validation rule: {RuleId}", ruleId);
        var existingRule = _validationRules.FirstOrDefault(r => r.Id == ruleId);
        if (existingRule == null) return Task.FromResult(false);

        _validationRules.Remove(existingRule);
        return Task.FromResult(true);
    }

    private async Task<List<TarsEngine.Models.KnowledgeItem>> ExtractFromSectionAsync(ContentSection section, DocumentParsingResult document)
    {
        var items = new List<TarsEngine.Models.KnowledgeItem>();

        // Move CPU-intensive processing to a background thread
        return await Task.Run(() =>
        {
            // Extract based on content type
            switch (section.ContentType)
            {
                case ContentType.Text:
                    var textResult = ExtractFromTextAsync(section.RawContent).Result;
                    items.AddRange(textResult.Items);
                    break;

                case ContentType.Code:
                    foreach (var codeBlock in section.CodeBlocks)
                    {
                        var codeResult = ExtractFromCodeAsync(codeBlock.Code, codeBlock.Language).Result;
                        items.AddRange(codeResult.Items);
                    }
                    break;

                case ContentType.Concept:
                    var conceptItems = ExtractConceptsAsync(section.RawContent).Result;
                    items.AddRange(conceptItems);
                    break;

                case ContentType.Insight:
                    var insightItems = ExtractInsightsAsync(section.RawContent).Result;
                    items.AddRange(insightItems);
                    break;

                case ContentType.Question:
                    var questionItem = new TarsEngine.Models.KnowledgeItem
                    {
                        Type = ModelKnowledgeType.Question,
                        Content = section.RawContent,
                        Source = document.DocumentPath,
                        Context = section.Heading,
                        Confidence = 0.9,
                        Relevance = 0.7,
                        Tags = ["question"]
                    };
                    items.Add(questionItem);
                    break;

                case ContentType.Answer:
                    var answerItem = new TarsEngine.Models.KnowledgeItem
                    {
                        Type = ModelKnowledgeType.Answer,
                        Content = section.RawContent,
                        Source = document.DocumentPath,
                        Context = section.Heading,
                        Confidence = 0.9,
                        Relevance = 0.8,
                        Tags = ["answer"]
                    };
                    items.Add(answerItem);
                    break;

                case ContentType.Example:
                    var exampleItem = new TarsEngine.Models.KnowledgeItem
                    {
                        Type = ModelKnowledgeType.CodePattern,
                        Content = section.RawContent,
                        Source = document.DocumentPath,
                        Context = section.Heading,
                        Confidence = 0.9,
                        Relevance = 0.8,
                        Tags = ["example"]
                    };
                    items.Add(exampleItem);
                    break;
            }

            // Extract from code blocks in any section type
            foreach (var codeBlock in section.CodeBlocks)
            {
                var codeResult = ExtractFromCodeAsync(codeBlock.Code, codeBlock.Language).Result;
                items.AddRange(codeResult.Items);
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
        });
    }

    private async Task<List<ModelKnowledgeItem>> ExtractConceptsAsync(string content)
    {
        try
        {
            _logger.LogDebug("Extracting concepts from content of length {Length}", content?.Length ?? 0);

            // Since concept extraction is CPU-intensive, move it to a background thread
            return await Task.Run(() =>
            {
                var concepts = new List<ModelKnowledgeItem>();

                // Extract concepts using the content classifier
                var classification = _contentClassifierService.ClassifyContent(content ?? string.Empty);

                // Only create a concept if the classification is of type Concept
                if (classification.PrimaryCategory == ContentCategory.Concept)
                {
                    concepts.Add(new ModelKnowledgeItem
                    {
                        Type = ModelKnowledgeType.Concept,
                        Content = classification.Content,
                        Context = classification.PrimaryCategory.ToString(),
                        Confidence = classification.ConfidenceScore,
                        Relevance = classification.RelevanceScore,
                        Tags = classification.Tags.ToList()
                    });
                }

                _logger.LogDebug("Extracted {ConceptCount} concepts", concepts.Count);
                return concepts;
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting concepts from content");
            return new List<ModelKnowledgeItem>();
        }
    }

    private async Task<List<ModelKnowledgeItem>> ExtractInsightsAsync(string content)
    {
        try
        {
            _logger.LogDebug("Extracting insights from content of length {Length}", content?.Length ?? 0);

            // Since insight extraction is CPU-intensive, move it to a background thread
            return await Task.Run(() =>
            {
                var insights = new List<ModelKnowledgeItem>();

                // Extract insights using the content classifier
                var classification = _contentClassifierService.ClassifyContentAsync(content ?? string.Empty).Result;

                // Create insights based on classification
                if (classification.PrimaryCategory == ContentCategory.Insight)
                {
                    insights.Add(new ModelKnowledgeItem
                    {
                        Type = ModelKnowledgeType.Insight,
                        Content = classification.Content,
                        Context = classification.PrimaryCategory.ToString(),
                        Confidence = classification.ConfidenceScore,
                        Relevance = classification.RelevanceScore,
                        Tags = classification.Tags.ToList()
                    });
                }

                _logger.LogDebug("Extracted {InsightCount} insights", insights.Count);
                return insights;
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting insights from content");
            return [];
        }
    }

    private List<ModelKnowledgeItem> ExtractCodePatterns(string code, string language)
    {
        var patterns = new List<ModelKnowledgeItem>();

        // Extract class/interface definitions
        var classMatches = _codePatternRegex.Matches(code);
        foreach (Match match in classMatches)
        {
            if (match.Groups.Count > 1)
            {
                var className = match.Groups[1].Value.Trim();
                if (!string.IsNullOrWhiteSpace(className))
                {
                    var item = new ModelKnowledgeItem
                    {
                        Type = ModelKnowledgeType.CodePattern,
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
                    var item = new ModelKnowledgeItem
                    {
                        Type = ModelKnowledgeType.CodePattern,
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
            ContentCategory.Concept => ModelKnowledgeType.Concept,
            ContentCategory.CodeExample => ModelKnowledgeType.CodePattern,
            ContentCategory.Algorithm => ModelKnowledgeType.Algorithm,
            ContentCategory.DesignPattern => ModelKnowledgeType.DesignPattern,
            ContentCategory.BestPractice => ModelKnowledgeType.BestPractice,
            ContentCategory.ApiDoc => ModelKnowledgeType.ApiUsage,
            ContentCategory.Question => ModelKnowledgeType.Question,
            ContentCategory.Answer => ModelKnowledgeType.Answer,
            ContentCategory.Insight => ModelKnowledgeType.Insight,
            ContentCategory.Performance => ModelKnowledgeType.Performance,
            ContentCategory.Security => ModelKnowledgeType.Security,
            ContentCategory.Testing => ModelKnowledgeType.Testing,
            _ => ModelKnowledgeType.Unknown
        };

        // Skip unknown types
        if (knowledgeType == ModelKnowledgeType.Unknown)
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
        if (item1.Type == ModelKnowledgeType.Question && item2.Type == ModelKnowledgeType.Answer)
        {
            return (RelationshipType.Answers, 0.9);
        }
        if (item1.Type == ModelKnowledgeType.Answer && item2.Type == ModelKnowledgeType.Question)
        {
            return (RelationshipType.Questions, 0.9);
        }

        // Check for implementation relationship
        if (item1.Type == ModelKnowledgeType.CodePattern && item2.Type == ModelKnowledgeType.Algorithm)
        {
            return (RelationshipType.Implements, 0.8);
        }
        if (item1.Type == ModelKnowledgeType.Algorithm && item2.Type == ModelKnowledgeType.CodePattern)
        {
            return (RelationshipType.IsImplementedBy, 0.8);
        }

        // Check for concept-example relationship
        if (item1.Type == ModelKnowledgeType.Concept && item2.Type == ModelKnowledgeType.CodePattern)
        {
            return (RelationshipType.IsImplementedBy, 0.7);
        }
        if (item1.Type == ModelKnowledgeType.CodePattern && item2.Type == ModelKnowledgeType.Concept)
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

    private bool ValidateAgainstRule(TarsEngine.Models.KnowledgeItem item, KnowledgeValidationRule rule)
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
            Name = "MinimumContentLength",
            Description = "Content must be at least 10 characters long",
            ValidationCriteria = "MinLength=10",
            ErrorMessage = "Content is too short (minimum 10 characters)",
            Severity = ValidationSeverity.Error
        });

        _validationRules.Add(new KnowledgeValidationRule
        {
            Name = "MinimumConfidence",
            Description = "Confidence score must be at least 0.5",
            ValidationCriteria = "MinConfidence=0.5",
            ErrorMessage = "Confidence score is too low (minimum 0.5)",
            Severity = ValidationSeverity.Warning
        });

        _validationRules.Add(new KnowledgeValidationRule
        {
            Name = "RequiredFields",
            Description = "Required fields must be filled",
            ValidationCriteria = "RequiredFields",
            ErrorMessage = "One or more required fields are missing",
            Severity = ValidationSeverity.Error
        });
    }
}
