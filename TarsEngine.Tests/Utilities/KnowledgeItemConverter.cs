using ModelKnowledgeItem = TarsEngine.Models.KnowledgeItem;
using InterfaceKnowledgeItem = TarsEngine.Services.Interfaces.KnowledgeItem;

namespace TarsEngine.Tests.Utilities;

/// <summary>
/// Utility class for converting between different KnowledgeItem types
/// </summary>
public static class KnowledgeItemConverter
{
    /// <summary>
    /// Converts a TarsEngine.Services.Interfaces.KnowledgeItem to a TarsEngine.Models.KnowledgeItem
    /// </summary>
    public static ModelKnowledgeItem ToModelKnowledgeItem(this InterfaceKnowledgeItem item)
    {
        return new ModelKnowledgeItem
        {
            Id = item.Id,
            Type = TarsEngine.Services.Adapters.KnowledgeTypeAdapter.ToModelType(item.Type),
            Content = item.Content,
            Source = item.Source,
            Confidence = item.Confidence,
            CreatedAt = item.Timestamp,
            UpdatedAt = DateTime.UtcNow,
            Tags = item.RelatedItems?.ToList() ?? new List<string>(),
            Metadata = item.Metadata?.ToDictionary(kvp => kvp.Key, kvp => kvp.Value) ?? new Dictionary<string, string>()
        };
    }

    /// <summary>
    /// Converts a TarsEngine.Models.KnowledgeItem to a TarsEngine.Services.Interfaces.KnowledgeItem
    /// </summary>
    public static InterfaceKnowledgeItem ToServiceKnowledgeItem(this ModelKnowledgeItem item)
    {
        return new InterfaceKnowledgeItem
        {
            Id = item.Id,
            Type = TarsEngine.Services.Adapters.KnowledgeTypeAdapter.ToInterfaceType(item.Type),
            Content = item.Content,
            Source = item.Source,
            Confidence = item.Confidence,
            Timestamp = item.CreatedAt,
            RelatedItems = item.Tags?.ToList() ?? new List<string>(),
            Metadata = item.Metadata?.ToDictionary(kvp => kvp.Key, kvp => kvp.Value) ?? new Dictionary<string, string>()
        };
    }

    /// <summary>
    /// Converts a list of TarsEngine.Services.Interfaces.KnowledgeItem to a list of TarsEngine.Models.KnowledgeItem
    /// </summary>
    public static IEnumerable<ModelKnowledgeItem> ToModelKnowledgeItems(this IEnumerable<InterfaceKnowledgeItem> items)
    {
        return items.Select(item => item.ToModelKnowledgeItem());
    }

    /// <summary>
    /// Converts a list of TarsEngine.Models.KnowledgeItem to a list of TarsEngine.Services.Interfaces.KnowledgeItem
    /// </summary>
    public static IEnumerable<InterfaceKnowledgeItem> ToServiceKnowledgeItems(this IEnumerable<ModelKnowledgeItem> items)
    {
        return items.Select(item => item.ToServiceKnowledgeItem());
    }
}