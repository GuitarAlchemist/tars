using Microsoft.Extensions.Logging;
using TarsEngine.Services.Abstractions.Knowledge;
using TarsEngine.Services.Abstractions.Models.Knowledge;
using TarsEngine.Services.Core.Base;

namespace TarsEngine.Services.Knowledge
{
    /// <summary>
    /// Implementation of the IKnowledgeRepository interface.
    /// </summary>
    public class KnowledgeRepository : ServiceBase, IKnowledgeRepository
    {
        private readonly Dictionary<string, KnowledgeItem> _items = new();

        /// <summary>
        /// Initializes a new instance of the <see cref="KnowledgeRepository"/> class.
        /// </summary>
        /// <param name="logger">The logger instance.</param>
        public KnowledgeRepository(ILogger<KnowledgeRepository> logger)
            : base(logger)
        {
        }

        /// <inheritdoc/>
        public override string Name => "Knowledge Repository";

        /// <inheritdoc/>
        public Task AddItemAsync(KnowledgeItem item)
        {
            Logger.LogInformation("Adding knowledge item: {ItemId} - {ItemTitle}", item.Id, item.Title);
            
            _items[item.Id] = item;
            
            return Task.CompletedTask;
        }

        /// <inheritdoc/>
        public Task DeleteItemAsync(string id)
        {
            Logger.LogInformation("Deleting knowledge item: {ItemId}", id);
            
            _items.Remove(id);
            
            return Task.CompletedTask;
        }

        /// <inheritdoc/>
        public Task<KnowledgeItem?> GetItemByIdAsync(string id)
        {
            Logger.LogInformation("Getting knowledge item by ID: {ItemId}", id);
            
            _items.TryGetValue(id, out var item);
            
            return Task.FromResult(item);
        }

        /// <inheritdoc/>
        public Task<IEnumerable<KnowledgeItem>> GetItemsByTypeAsync(KnowledgeType type, int limit = 10)
        {
            Logger.LogInformation("Getting knowledge items by type: {Type}, limit: {Limit}", type, limit);
            
            var items = _items.Values
                .Where(i => i.Type == type)
                .Take(limit);
            
            return Task.FromResult(items);
        }

        /// <inheritdoc/>
        public Task<IEnumerable<KnowledgeItem>> SearchAsync(string query, int limit = 10)
        {
            Logger.LogInformation("Searching knowledge items with query: {Query}, limit: {Limit}", query, limit);
            
            var items = _items.Values
                .Where(i => 
                    i.Title.Contains(query, StringComparison.OrdinalIgnoreCase) || 
                    i.Content.Contains(query, StringComparison.OrdinalIgnoreCase) ||
                    i.Tags.Any(t => t.Contains(query, StringComparison.OrdinalIgnoreCase)))
                .Take(limit);
            
            return Task.FromResult(items);
        }

        /// <inheritdoc/>
        public Task UpdateItemAsync(KnowledgeItem item)
        {
            Logger.LogInformation("Updating knowledge item: {ItemId} - {ItemTitle}", item.Id, item.Title);
            
            if (_items.ContainsKey(item.Id))
            {
                item.UpdatedAt = DateTime.UtcNow;
                _items[item.Id] = item;
            }
            else
            {
                Logger.LogWarning("Attempted to update non-existent knowledge item: {ItemId}", item.Id);
            }
            
            return Task.CompletedTask;
        }
    }
}
