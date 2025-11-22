using System;
using System.Collections.Generic;

public class MemoryCache<TKey, TValue>
{
    private readonly Dictionary<TKey, CacheEntry<TValue>> _cache = new Dictionary<TKey, CacheEntry<TValue>>();
    private readonly object _syncLock = new object();
    private readonly TimeSpan _expirationTimeSpan;

    public MemoryCache(TimeSpan expirationTimeSpan)
    {
        _expirationTimeSpan = expirationTimeSpan;
    }

    public TValue Get(TKey key)
    {
        if (_cache.TryGetValue(key, out var cacheEntry))
        {
            if (DateTime.UtcNow - cacheEntry.LastAccessed > _expirationTimeSpan)
            {
                Remove(key);
                return default(TValue); // Return default value for type
            }
            else
            {
                cacheEntry.LastAccessed = DateTime.UtcNow;
                return cacheEntry.Value;
            }
        }
        return default(TValue); // Return default value for type
    }

    public void Add(TKey key, TValue value)
    {
        _cache.Add(key, new CacheEntry<TValue>(value));
    }

    public void Remove(TKey key)
    {
        _cache.Remove(key);
    }

    public void Clear()
    {
        _cache.Clear();
    }

    private class CacheEntry<T>
    {
        public T Value { get; set; }
        public DateTime LastAccessed { get; set; }

        public CacheEntry(T value)
        {
            Value = value;
            LastAccessed = DateTime.UtcNow;
        }
    }
}