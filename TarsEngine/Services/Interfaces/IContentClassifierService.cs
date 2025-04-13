using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Service for classifying content
/// </summary>
public interface IContentClassifierService
{
    /// <summary>
    /// Classifies content
    /// </summary>
    /// <param name="content">The content to classify</param>
    /// <param name="options">Optional classification options</param>
    /// <returns>The content classification</returns>
    Task<ContentClassification> ClassifyContentAsync(string content, Dictionary<string, string>? options = null);

    /// <summary>
    /// Classifies a document parsing result
    /// </summary>
    /// <param name="document">The document parsing result to classify</param>
    /// <param name="options">Optional classification options</param>
    /// <returns>A batch of content classifications</returns>
    Task<ContentClassificationBatch> ClassifyDocumentAsync(DocumentParsingResult document, Dictionary<string, string>? options = null);

    /// <summary>
    /// Classifies a collection of content sections
    /// </summary>
    /// <param name="sections">The content sections to classify</param>
    /// <param name="options">Optional classification options</param>
    /// <returns>A batch of content classifications</returns>
    Task<ContentClassificationBatch> ClassifySectionsAsync(IEnumerable<ContentSection> sections, Dictionary<string, string>? options = null);

    /// <summary>
    /// Calculates the relevance score for content
    /// </summary>
    /// <param name="content">The content to score</param>
    /// <param name="context">Optional context information</param>
    /// <returns>The relevance score (0-1)</returns>
    Task<double> CalculateRelevanceScoreAsync(string content, Dictionary<string, string>? context = null);

    /// <summary>
    /// Calculates the quality score for content
    /// </summary>
    /// <param name="content">The content to score</param>
    /// <param name="options">Optional scoring options</param>
    /// <returns>The quality score (0-1)</returns>
    Task<double> CalculateQualityScoreAsync(string content, Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets the classification rules
    /// </summary>
    /// <returns>The classification rules</returns>
    Task<IEnumerable<ClassificationRule>> GetClassificationRulesAsync();

    /// <summary>
    /// Adds a classification rule
    /// </summary>
    /// <param name="rule">The rule to add</param>
    /// <returns>The added rule</returns>
    Task<ClassificationRule> AddClassificationRuleAsync(ClassificationRule rule);

    /// <summary>
    /// Updates a classification rule
    /// </summary>
    /// <param name="rule">The rule to update</param>
    /// <returns>The updated rule</returns>
    Task<ClassificationRule> UpdateClassificationRuleAsync(ClassificationRule rule);

    /// <summary>
    /// Deletes a classification rule
    /// </summary>
    /// <param name="ruleId">The ID of the rule to delete</param>
    /// <returns>True if the rule was deleted, false otherwise</returns>
    Task<bool> DeleteClassificationRuleAsync(string ruleId);

    /// <summary>
    /// Gets the tags for content
    /// </summary>
    /// <param name="content">The content to tag</param>
    /// <param name="options">Optional tagging options</param>
    /// <returns>The tags</returns>
    Task<IEnumerable<string>> GetTagsAsync(string content, Dictionary<string, string>? options = null);
}
