using System;
using System.Collections.Generic;
using System.Linq;

namespace TarsCliMinimal.Examples
{
    /// <summary>
    /// Example class with methods that use generic types
    /// </summary>
    public class GenericTypeExample
    {
        /// <summary>
        /// Calculates the average of a list of integers
        /// </summary>
        public double Average(List<int> numbers)
        {
            if (numbers == null || numbers.Count == 0)
                return 0;
                
            return numbers.Average();
        }
        
        /// <summary>
        /// Finds the maximum value in a list of integers
        /// </summary>
        public int FindMax(List<int> numbers)
        {
            if (numbers == null || numbers.Count == 0)
                return 0;
                
            return numbers.Max();
        }
        
        /// <summary>
        /// Counts the number of occurrences of a key in a dictionary
        /// </summary>
        public int CountOccurrences<TKey, TValue>(Dictionary<TKey, TValue> dictionary, TKey key)
        {
            if (dictionary == null)
                return 0;
                
            return dictionary.ContainsKey(key) ? 1 : 0;
        }
        
        /// <summary>
        /// Converts a dictionary to a list of key-value pairs
        /// </summary>
        public List<KeyValuePair<string, int>> ConvertToList(Dictionary<string, int> dictionary)
        {
            if (dictionary == null)
                return new List<KeyValuePair<string, int>>();
                
            return dictionary.ToList();
        }
        
        /// <summary>
        /// Merges two lists into a single list
        /// </summary>
        public List<T> MergeLists<T>(List<T> list1, List<T> list2)
        {
            if (list1 == null)
                list1 = new List<T>();
                
            if (list2 == null)
                list2 = new List<T>();
                
            var result = new List<T>(list1);
            result.AddRange(list2);
            return result;
        }
        
        /// <summary>
        /// Filters a list based on a predicate
        /// </summary>
        public List<T> FilterList<T>(List<T> list, Func<T, bool> predicate)
        {
            if (list == null)
                return new List<T>();
                
            return list.Where(predicate).ToList();
        }
        
        /// <summary>
        /// Creates a dictionary from two lists
        /// </summary>
        public Dictionary<TKey, TValue> CreateDictionary<TKey, TValue>(List<TKey> keys, List<TValue> values)
        {
            if (keys == null || values == null)
                return new Dictionary<TKey, TValue>();
                
            var result = new Dictionary<TKey, TValue>();
            int count = Math.Min(keys.Count, values.Count);
            
            for (int i = 0; i < count; i++)
            {
                result[keys[i]] = values[i];
            }
            
            return result;
        }
    }
}
