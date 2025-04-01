using System;
using System.Collections.Generic;

public class Cache<TKey, TValue>
{
    private readonly Dictionary<TKey, CacheEntry<TValue>> _cache;
    private readonly object _syncLock = new object();
    private readonly int _defaultTimeoutSeconds;

    public Cache(int defaultTimeoutSeconds = 300)
    {
        _cache = new Dictionary<TKey, CacheEntry<TValue>>();
        _defaultTimeoutSeconds = defaultTimeoutSeconds;
    }

    public void Add(TKey key, TValue value)
    {
        lock (_syncLock)
        {
            var entry = new CacheEntry<TValue>(value);
            if (entry.Expiration > DateTime.Now.AddSeconds(_defaultTimeoutSeconds))
                _cache[key] = entry;
            else
                throw new InvalidOperationException("Cannot add cache entry with expiration in the past.");
        }
    }

    public TValue Get(TKey key)
    {
        lock (_syncLock)
        {
            TValue result;
            if (_cache.TryGetValue(key, out result) && result.Expiration > DateTime.Now)
                return result.Value;
            else
                throw new KeyNotFoundException("Cache entry not found or expired.");
        }
    }

    public void Remove(TKey key)
    {
        lock (_syncLock)
        {
            _cache.Remove(key);
        }
    }

    public void Clear()
    {
        lock (_syncLock)
        {
            _cache.Clear();
        }
    }
}

public class CacheEntry<T>
{
    public DateTime Expiration { get; set; }
    public T Value { get; set; }

    public CacheEntry(T value)
    {
        Expiration = DateTime.Now.AddSeconds(1);
        Value = value;
    }
}