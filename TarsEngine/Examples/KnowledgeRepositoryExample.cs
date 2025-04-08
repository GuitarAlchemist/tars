using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Monads;
using TarsEngine.Services.Adapters;
using ModelKnowledgeItem = TarsEngine.Models.KnowledgeItem;
using InterfaceKnowledgeItem = TarsEngine.Services.Interfaces.KnowledgeItem;
using ModelKnowledgeType = TarsEngine.Models.KnowledgeType;
using InterfaceKnowledgeType = TarsEngine.Services.Interfaces.KnowledgeType;

namespace TarsEngine.Examples
{
    /// <summary>
    /// Example of how to refactor KnowledgeRepository to use adapters and monads
    /// </summary>
    public class KnowledgeRepositoryExample
    {
        private readonly ILogger<KnowledgeRepositoryExample> _logger;
        private readonly List<ModelKnowledgeItem> _knowledgeItems = [];

        public KnowledgeRepositoryExample(ILogger<KnowledgeRepositoryExample> logger)
        {
            _logger = logger;
        }

        #region Original Implementation

        /// <summary>
        /// Original implementation with ambiguous types
        /// </summary>
        public IEnumerable<InterfaceKnowledgeItem> AddItems(IEnumerable<InterfaceKnowledgeItem> items)
        {
            if (items == null)
                return [];

            // Add each item
            var addedItems = new List<ModelKnowledgeItem>();

            foreach (var item in items)
            {
                try
                {
                    // Convert from interface to model
                    var modelItem = new ModelKnowledgeItem
                    {
                        Id = item.Id,
                        Content = item.Content,
                        Type = (ModelKnowledgeType)(int)item.Type,
                        Source = item.Source,
                        Context = item.Metadata.TryGetValue("Context", out var context) ? context : string.Empty,
                        Confidence = item.Confidence,
                        CreatedAt = item.Timestamp,
                        UpdatedAt = DateTime.UtcNow,
                        Tags = item.RelatedItems ?? [],
                        Metadata = item.Metadata ?? new Dictionary<string, string>()
                    };

                    _knowledgeItems.Add(modelItem);
                    addedItems.Add(modelItem);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, $"Error adding knowledge item: {item.Id}");
                }
            }

            // Convert back to interface items
            return addedItems.Select(item => new InterfaceKnowledgeItem
            {
                Id = item.Id,
                Content = item.Content,
                Type = (InterfaceKnowledgeType)(int)item.Type,
                Source = item.Source,
                Confidence = item.Confidence,
                Timestamp = item.CreatedAt,
                RelatedItems = item.Tags ?? [],
                Metadata = item.Metadata ?? new Dictionary<string, string>()
            });
        }

        /// <summary>
        /// Original implementation with ambiguous types
        /// </summary>
        public IEnumerable<InterfaceKnowledgeItem> SearchByType(InterfaceKnowledgeType type)
        {
            var results = _knowledgeItems.Where(item => (int)item.Type == (int)type).ToList();

            return results.Select(item => new InterfaceKnowledgeItem
            {
                Id = item.Id,
                Content = item.Content,
                Type = (InterfaceKnowledgeType)(int)item.Type,
                Source = item.Source,
                Confidence = item.Confidence,
                Timestamp = item.CreatedAt,
                RelatedItems = item.Tags ?? [],
                Metadata = item.Metadata ?? new Dictionary<string, string>()
            });
        }

        #endregion

        #region Adapter Implementation

        /// <summary>
        /// Refactored implementation using adapters
        /// </summary>
        public IEnumerable<InterfaceKnowledgeItem> AddItemsWithAdapter(IEnumerable<InterfaceKnowledgeItem> items)
        {
            if (items == null)
                return [];

            // Add each item
            var addedItems = new List<ModelKnowledgeItem>();

            foreach (var item in items)
            {
                try
                {
                    // Convert from interface to model using adapter
                    var modelItem = KnowledgeItemAdapter.ToModel(item);

                    _knowledgeItems.Add(modelItem);
                    addedItems.Add(modelItem);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, $"Error adding knowledge item: {item.Id}");
                }
            }

            // Convert back to interface items using adapter
            return addedItems.Select(KnowledgeItemAdapter.ToInterface);
        }

        /// <summary>
        /// Refactored implementation using adapters
        /// </summary>
        public IEnumerable<InterfaceKnowledgeItem> SearchByTypeWithAdapter(InterfaceKnowledgeType type)
        {
            // Convert type using adapter
            var modelType = KnowledgeItemAdapter.ConvertTypeToModel(type);

            // Use type comparison helper
            var results = _knowledgeItems.Where(item => KnowledgeItemAdapter.AreTypesEqual(item.Type, type)).ToList();

            // Convert results using adapter
            return results.Select(KnowledgeItemAdapter.ToInterface);
        }

        #endregion

        #region Monad Implementation

        /// <summary>
        /// Advanced implementation using adapters and monads
        /// </summary>
        public Result<IEnumerable<InterfaceKnowledgeItem>, Exception> AddItemsWithMonad(IEnumerable<InterfaceKnowledgeItem> items)
        {
            if (items == null)
                return Result<IEnumerable<InterfaceKnowledgeItem>, Exception>.Success([]);

            try
            {
                // Add each item
                var addedItems = new List<ModelKnowledgeItem>();

                foreach (var item in items)
                {
                    // Use Option monad to handle null items
                    Option<InterfaceKnowledgeItem> itemOption = Monad.FromNullable(item);

                    // Use Result monad to handle conversion errors
                    Result<ModelKnowledgeItem, Exception> conversionResult = itemOption.Match(
                        some: i => Monad.Try(() => KnowledgeItemAdapter.ToModel(i)),
                        none: () => Result<ModelKnowledgeItem, Exception>.Failure(new ArgumentNullException("item"))
                    );

                    // Add item if conversion was successful
                    conversionResult.IfSuccess(modelItem => {
                        _knowledgeItems.Add(modelItem);
                        addedItems.Add(modelItem);
                    }).IfFailure(ex => {
                        _logger.LogError(ex, $"Error adding knowledge item");
                    });
                }

                // Convert back to interface items using adapter
                return Result<IEnumerable<InterfaceKnowledgeItem>, Exception>.Success(
                    addedItems.Select(KnowledgeItemAdapter.ToInterface)
                );
            }
            catch (Exception ex)
            {
                return Result<IEnumerable<InterfaceKnowledgeItem>, Exception>.Failure(ex);
            }
        }

        /// <summary>
        /// Advanced implementation using adapters and monads
        /// </summary>
        public AsyncResult<IEnumerable<InterfaceKnowledgeItem>> SearchByTypeWithMonad(InterfaceKnowledgeType type)
        {
            return AsyncResult<IEnumerable<InterfaceKnowledgeItem>>.FromTask(Task.Run(() => {
                // Convert type using adapter
                var modelType = KnowledgeItemAdapter.ConvertTypeToModel(type);

                // Use type comparison helper
                var results = _knowledgeItems.Where(item => KnowledgeItemAdapter.AreTypesEqual(item.Type, type)).ToList();

                // Convert results using adapter
                return results.Select(KnowledgeItemAdapter.ToInterface);
            }));
        }

        #endregion
    }
}
