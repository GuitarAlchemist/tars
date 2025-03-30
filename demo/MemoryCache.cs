using System;
using System.Collections.Generic;

public class Cache<TKey, TValue>
{
    private readonly Dictionary<TKey, CachedItem<TValue>> _cache = new Dictionary<TKey, CachedItem<TValue>>();
    private readonly Dictionary<TKey, DateTime> _expires = new Dictionary<TKey, DateTime>();
    private readonly int _defaultExpirationSeconds;

    public Cache(int defaultExpirationSeconds)
    {
        _defaultExpirationSeconds = defaultExpirationSeconds;
    }

    public void Add(TKey key, TValue value)
    {
        var cachedItem = new CachedItem<TValue>(value);
        _cache.Add(key, cachedItem);
        _expires.Add(key, DateTime.UtcNow.AddSeconds(_defaultExpirationSeconds));
    }

    public TValue Get(TKey key)
    {
        if (!_cache.ContainsKey(key))
            return default(TValue);

        var cachedItem = _cache[key];
        var now = DateTime.UtcNow;

        if (now > _expires[key])
        {
            Remove(key);
            return default(TValue); // or throw an exception, depending on your requirements
        }

        return cachedItem.Value;
    }

    public void Remove(TKey key)
    {
        if (_cache.ContainsKey(key))
        {
            _cache.Remove(key);
            _expires.Remove(key);
        }
    }

    public void Clear()
    {
        _cache.Clear();
        _expires.Clear();
    }

    private class CachedItem<T>
    {
        public T Value { get; set; }
        public DateTime Expiration { get; set; }

        public CachedItem(T value)
        {
            Value = value;
            Expiration = DateTime.UtcNow.AddSeconds(30); // default expiration time
        }
    }
}