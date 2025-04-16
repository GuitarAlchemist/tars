using System;
using System.Collections.Generic;
using System.Linq;

namespace TarsCli.Examples
{
    /// <summary>
    /// Example class with methods that use generic types
    /// </summary>
    public class GenericTypeExample
    {
        /// <summary>
        /// Calculates the average of a list of integers
        /// </summary>
        /// <param name="numbers">List of integers</param>
        /// <returns>Average value</returns>
        public double Average(List<int> numbers)
        {
            if (numbers == null || !numbers.Any())
            {
                return 0;
            }

            return numbers.Average();
        }

        /// <summary>
        /// Finds the maximum value in a list of integers
        /// </summary>
        /// <param name="numbers">List of integers</param>
        /// <returns>Maximum value</returns>
        public int FindMax(List<int> numbers)
        {
            if (numbers == null || !numbers.Any())
            {
                return 0;
            }

            return numbers.Max();
        }

        /// <summary>
        /// Finds the minimum value in a list of integers
        /// </summary>
        /// <param name="numbers">List of integers</param>
        /// <returns>Minimum value</returns>
        public int FindMin(List<int> numbers)
        {
            if (numbers == null || !numbers.Any())
            {
                return 0;
            }

            return numbers.Min();
        }

        /// <summary>
        /// Filters a list of integers to only include even numbers
        /// </summary>
        /// <param name="numbers">List of integers</param>
        /// <returns>List of even numbers</returns>
        public List<int> FilterEven(List<int> numbers)
        {
            if (numbers == null)
            {
                return new List<int>();
            }

            return numbers.Where(n => n % 2 == 0).ToList();
        }

        /// <summary>
        /// Converts a dictionary to a list of key-value pairs
        /// </summary>
        /// <param name="dictionary">Dictionary to convert</param>
        /// <returns>List of key-value pairs</returns>
        public List<string> DictionaryToList(Dictionary<string, int> dictionary)
        {
            if (dictionary == null)
            {
                return new List<string>();
            }

            return dictionary.Select(kvp => $"{kvp.Key}: {kvp.Value}").ToList();
        }

        /// <summary>
        /// Merges two dictionaries
        /// </summary>
        /// <param name="first">First dictionary</param>
        /// <param name="second">Second dictionary</param>
        /// <returns>Merged dictionary</returns>
        public Dictionary<string, int> MergeDictionaries(Dictionary<string, int> first, Dictionary<string, int> second)
        {
            var result = new Dictionary<string, int>(first);
            
            foreach (var kvp in second)
            {
                if (result.ContainsKey(kvp.Key))
                {
                    result[kvp.Key] += kvp.Value;
                }
                else
                {
                    result[kvp.Key] = kvp.Value;
                }
            }

            return result;
        }
    }
}
