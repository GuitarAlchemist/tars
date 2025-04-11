using System;
using System.Collections.Generic;
using ModelKnowledgeItem = TarsEngine.Models.KnowledgeItem;
using InterfaceKnowledgeItem = TarsEngine.Services.Interfaces.KnowledgeItem;
using ModelKnowledgeType = TarsEngine.Models.KnowledgeType;
using InterfaceKnowledgeType = TarsEngine.Services.Interfaces.KnowledgeType;
using ServicesModelKnowledgeType = TarsEngine.Services.Models.KnowledgeType;

namespace TarsEngine.Services.Adapters
{
    /// <summary>
    /// Adapter for converting between different KnowledgeItem types
    /// </summary>
    public static class KnowledgeItemAdapter
    {
        /// <summary>
        /// Converts a model KnowledgeItem to an interface KnowledgeItem
        /// </summary>
        public static InterfaceKnowledgeItem ToInterface(ModelKnowledgeItem model)
        {
            if (model == null)
                return null;

            return new InterfaceKnowledgeItem
            {
                Id = model.Id,
                Content = model.Content,
                Type = ConvertTypeToInterface(model.Type),
                Source = model.Source,
                Confidence = model.Confidence,
                Timestamp = model.CreatedAt,
                RelatedItems = model.Tags ?? [],
                Metadata = model.Metadata ?? new Dictionary<string, string>()
            };
        }

        /// <summary>
        /// Converts a model KnowledgeItem to an interface KnowledgeItem
        /// </summary>
        public static InterfaceKnowledgeItem ConvertToInterface(ModelKnowledgeItem model)
        {
            if (model == null)
                return null;

            return ToInterface(model);
        }

        /// <summary>
        /// Converts an interface KnowledgeItem to a model KnowledgeItem
        /// </summary>
        public static ModelKnowledgeItem ToModel(InterfaceKnowledgeItem interfaceItem)
        {
            if (interfaceItem == null)
                return null;

            // If it's already a ModelKnowledgeItem, just return it
            if (interfaceItem.GetType().FullName == typeof(ModelKnowledgeItem).FullName)
                return (ModelKnowledgeItem)(object)interfaceItem;

            var result = new ModelKnowledgeItem
            {
                Id = interfaceItem.Id,
                Content = interfaceItem.Content,
                Type = ConvertTypeToModel(interfaceItem.Type),
                Source = interfaceItem.Source,
                Context = interfaceItem.Metadata != null && interfaceItem.Metadata.TryGetValue("Context", out var context) ? context : string.Empty,
                Confidence = interfaceItem.Confidence,
                CreatedAt = interfaceItem.Timestamp,
                UpdatedAt = DateTime.UtcNow,
                Tags = interfaceItem.RelatedItems != null ? [..interfaceItem.RelatedItems] : [],
                Metadata = interfaceItem.Metadata != null ? new Dictionary<string, string>(interfaceItem.Metadata) : new Dictionary<string, string>()
            };

            return result;
        }

        /// <summary>
        /// Converts an interface KnowledgeItem to a model KnowledgeItem
        /// </summary>
        public static ModelKnowledgeItem ConvertToModel(object item)
        {
            if (item == null)
                return null;

            // If it's already a ModelKnowledgeItem, just return it
            if (item.GetType() == typeof(ModelKnowledgeItem))
                return (ModelKnowledgeItem)item;

            // If it's an InterfaceKnowledgeItem, convert it
            if (item is InterfaceKnowledgeItem interfaceItem)
                return ToModel(interfaceItem);

            throw new ArgumentException($"Cannot convert item of type {item.GetType().Name} to ModelKnowledgeItem");
        }

        /// <summary>
        /// Converts a KnowledgeType from the Services namespace to a KnowledgeType from the Models namespace
        /// </summary>
        public static ModelKnowledgeType ConvertTypeToModel(object type)
        {
            if (type == null)
                return ModelKnowledgeType.Unknown;

            // If it's already a ModelKnowledgeType, just return it
            if (type.GetType() == typeof(ModelKnowledgeType))
                return (ModelKnowledgeType)type;

            // If it's a Services.Models.KnowledgeType, convert it based on the name
            if (type is Models.KnowledgeType servicesType)
            {
                return servicesType.ToString() switch
                {
                    "CodePattern" => ModelKnowledgeType.CodePattern,
                    "Concept" => ModelKnowledgeType.Concept,
                    "BestPractice" => ModelKnowledgeType.BestPractice,
                    "Algorithm" => ModelKnowledgeType.Algorithm,
                    "DesignPattern" => ModelKnowledgeType.DesignPattern,
                    "Insight" => ModelKnowledgeType.Insight,
                    "Question" => ModelKnowledgeType.Question,
                    "Answer" => ModelKnowledgeType.Answer,
                    "Tool" => ModelKnowledgeType.Tool,
                    "Resource" => ModelKnowledgeType.Resource,
                    _ => ModelKnowledgeType.Unknown
                };
            }

            // If it's an InterfaceKnowledgeType, convert it based on the integer value
            if (type is InterfaceKnowledgeType interfaceType)
            {
                return (ModelKnowledgeType)(int)interfaceType;
            }

            // Try to convert by string value
            return type.ToString() switch
            {
                "CodePattern" => ModelKnowledgeType.CodePattern,
                "Concept" => ModelKnowledgeType.Concept,
                "BestPractice" => ModelKnowledgeType.BestPractice,
                "Algorithm" => ModelKnowledgeType.Algorithm,
                "DesignPattern" => ModelKnowledgeType.DesignPattern,
                "Insight" => ModelKnowledgeType.Insight,
                "Question" => ModelKnowledgeType.Question,
                "Answer" => ModelKnowledgeType.Answer,
                "Tool" => ModelKnowledgeType.Tool,
                "Resource" => ModelKnowledgeType.Resource,
                _ => ModelKnowledgeType.Unknown
            };
        }

        /// <summary>
        /// Converts a model KnowledgeType to an interface KnowledgeType
        /// </summary>
        public static InterfaceKnowledgeType ConvertTypeToInterface(ModelKnowledgeType type)
        {
            // Direct conversion based on integer value
            return (InterfaceKnowledgeType)(int)type;
        }

        /// <summary>
        /// Converts an interface KnowledgeType to a model KnowledgeType
        /// </summary>
        public static ModelKnowledgeType ConvertTypeToModel(InterfaceKnowledgeType type)
        {
            // Direct conversion based on integer value
            return (ModelKnowledgeType)(int)type;
        }

        /// <summary>
        /// Compares two KnowledgeType values
        /// </summary>
        public static bool AreTypesEqual(ModelKnowledgeType modelType, InterfaceKnowledgeType interfaceType)
        {
            // Compare based on integer values
            return (int)modelType == (int)interfaceType;
        }

        /// <summary>
        /// Converts a Services.Models.KnowledgeType to an interface KnowledgeType
        /// </summary>
        public static InterfaceKnowledgeType ConvertTypeToInterface(ServicesModelKnowledgeType type)
        {
            // Direct conversion based on integer value
            return (InterfaceKnowledgeType)(int)type;
        }

        /// <summary>
        /// Converts an interface KnowledgeType to a Services.Models.KnowledgeType
        /// </summary>
        public static ServicesModelKnowledgeType ConvertTypeToServicesModel(InterfaceKnowledgeType type)
        {
            // Direct conversion based on integer value
            return (ServicesModelKnowledgeType)(int)type;
        }

        /// <summary>
        /// Converts a model KnowledgeType to a Services.Models.KnowledgeType
        /// </summary>
        public static ServicesModelKnowledgeType ConvertTypeToServicesModel(ModelKnowledgeType type)
        {
            // Direct conversion based on integer value
            return (ServicesModelKnowledgeType)(int)type;
        }

        /// <summary>
        /// Converts a Services.Models.KnowledgeType to a model KnowledgeType
        /// </summary>
        public static ModelKnowledgeType ConvertTypeToModel(ServicesModelKnowledgeType type)
        {
            // Direct conversion based on integer value
            return (ModelKnowledgeType)(int)type;
        }
    }
}
