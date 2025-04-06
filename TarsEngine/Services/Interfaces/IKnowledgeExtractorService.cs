using System.Collections.Generic;
using System.Threading.Tasks;
using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Service for extracting knowledge from content
/// </summary>
public interface IKnowledgeExtractorService
{
    /// <summary>
    /// Extracts knowledge from text content
    /// </summary>
    /// <param name="content">The content to extract knowledge from</param>
    /// <param name="options">Optional extraction options</param>
    /// <returns>The knowledge extraction result</returns>
    Task<KnowledgeExtractionResult> ExtractFromTextAsync(string content, Dictionary<string, string>? options = null);

    /// <summary>
    /// Extracts knowledge from a document parsing result
    /// </summary>
    /// <param name="document">The document parsing result to extract knowledge from</param>
    /// <param name="options">Optional extraction options</param>
    /// <returns>The knowledge extraction result</returns>
    Task<KnowledgeExtractionResult> ExtractFromDocumentAsync(DocumentParsingResult document, Dictionary<string, string>? options = null);

    /// <summary>
    /// Extracts knowledge from code
    /// </summary>
    /// <param name="code">The code to extract knowledge from</param>
    /// <param name="language">The programming language of the code</param>
    /// <param name="options">Optional extraction options</param>
    /// <returns>The knowledge extraction result</returns>
    Task<KnowledgeExtractionResult> ExtractFromCodeAsync(string code, string language, Dictionary<string, string>? options = null);

    /// <summary>
    /// Extracts knowledge from a content classification
    /// </summary>
    /// <param name="classification">The content classification to extract knowledge from</param>
    /// <param name="options">Optional extraction options</param>
    /// <returns>The knowledge extraction result</returns>
    Task<KnowledgeExtractionResult> ExtractFromClassificationAsync(ContentClassification classification, Dictionary<string, string>? options = null);

    /// <summary>
    /// Extracts knowledge from a content classification batch
    /// </summary>
    /// <param name="batch">The content classification batch to extract knowledge from</param>
    /// <param name="options">Optional extraction options</param>
    /// <returns>The knowledge extraction result</returns>
    Task<KnowledgeExtractionResult> ExtractFromClassificationBatchAsync(ContentClassificationBatch batch, Dictionary<string, string>? options = null);

    /// <summary>
    /// Validates a knowledge item
    /// </summary>
    /// <param name="item">The knowledge item to validate</param>
    /// <param name="options">Optional validation options</param>
    /// <returns>The knowledge validation result</returns>
    Task<KnowledgeValidationResult> ValidateKnowledgeItemAsync(KnowledgeItem item, Dictionary<string, string>? options = null);

    /// <summary>
    /// Detects relationships between knowledge items
    /// </summary>
    /// <param name="items">The knowledge items to detect relationships between</param>
    /// <param name="options">Optional detection options</param>
    /// <returns>The detected relationships</returns>
    Task<List<KnowledgeRelationship>> DetectRelationshipsAsync(List<KnowledgeItem> items, Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets the validation rules
    /// </summary>
    /// <returns>The validation rules</returns>
    Task<List<KnowledgeValidationRule>> GetValidationRulesAsync();

    /// <summary>
    /// Adds a validation rule
    /// </summary>
    /// <param name="rule">The rule to add</param>
    /// <returns>The added rule</returns>
    Task<KnowledgeValidationRule> AddValidationRuleAsync(KnowledgeValidationRule rule);

    /// <summary>
    /// Updates a validation rule
    /// </summary>
    /// <param name="rule">The rule to update</param>
    /// <returns>The updated rule</returns>
    Task<KnowledgeValidationRule> UpdateValidationRuleAsync(KnowledgeValidationRule rule);

    /// <summary>
    /// Deletes a validation rule
    /// </summary>
    /// <param name="ruleId">The ID of the rule to delete</param>
    /// <returns>True if the rule was deleted, false otherwise</returns>
    Task<bool> DeleteValidationRuleAsync(string ruleId);
}
