using System;
using System.Collections.Generic;

namespace TarsEngine.Models.Adapters
{
    /// <summary>
    /// Adapter class to resolve ambiguity between TarsEngine.Services.Interfaces.KnowledgeItem and TarsEngine.Models.KnowledgeItem
    /// </summary>
    public class KnowledgeItemAdapter
    {
        // Properties from the service interface
        public string Id { get; set; } = string.Empty;
        public string Content { get; set; } = string.Empty;
        public KnowledgeTypeAdapter Type { get; set; } = new();
        public List<string> Tags { get; set; } = new();
        public double Relevance { get; set; }
        public string Source { get; set; } = string.Empty;
        public string Context { get; set; } = string.Empty;
        public double Confidence { get; set; }

        // Convert from service interface to model
        public static KnowledgeItem ToModel(Services.Interfaces.KnowledgeItem item)
        {
            return new KnowledgeItem
            {
                Id = item.Id,
                Content = item.Content,
                Type = KnowledgeTypeAdapter.ToModel(item.Type),
                Source = item.Source,
                Context = item.Metadata.TryGetValue("Context", out var context) ? context : string.Empty,
                Confidence = item.Confidence,
                CreatedAt = item.Timestamp,
                UpdatedAt = DateTime.UtcNow,
                Tags = item.RelatedItems ?? new List<string>(),
                Metadata = item.Metadata ?? new Dictionary<string, string>()
            };
        }

        // Convert from model to service interface
        public static Services.Interfaces.KnowledgeItem ToService(KnowledgeItem item)
        {
            var serviceItem = new Services.Interfaces.KnowledgeItem
            {
                Id = item.Id,
                Content = item.Content,
                Type = KnowledgeTypeAdapter.ToService(item.Type),
                Source = item.Source,
                Confidence = item.Confidence,
                Timestamp = item.CreatedAt,
                RelatedItems = item.Tags ?? new List<string>(),
                Metadata = item.Metadata ?? new Dictionary<string, string>()
            };

            // Add context to metadata if not empty
            if (!string.IsNullOrEmpty(item.Context))
            {
                serviceItem.Metadata["Context"] = item.Context;
            }

            return serviceItem;
        }
    }
}
