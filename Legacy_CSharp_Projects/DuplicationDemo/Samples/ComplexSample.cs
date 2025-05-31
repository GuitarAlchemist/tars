using System;
using System.Collections.Generic;
using System.Linq;

namespace DuplicationDemo.Samples
{
    public class ComplexSample
    {
        // Method 1 with duplicated code
        public void ProcessData(List<int> data)
        {
            // Duplicated block 1 - Data validation
            if (data == null)
            {
                throw new ArgumentNullException(nameof(data));
            }

            if (data.Count == 0)
            {
                Console.WriteLine("No data to process");
                return;
            }

            // Process the data
            var sum = 0;
            foreach (var item in data)
            {
                sum += item;
            }

            Console.WriteLine($"Sum: {sum}");
            Console.WriteLine($"Average: {sum / data.Count}");
        }

        // Method 2 with duplicated code
        public void AnalyzeData(List<int> data)
        {
            // Duplicated block 1 - Data validation
            if (data == null)
            {
                throw new ArgumentNullException(nameof(data));
            }

            if (data.Count == 0)
            {
                Console.WriteLine("No data to analyze");
                return;
            }

            // Analyze the data
            var min = data.Min();
            var max = data.Max();

            Console.WriteLine($"Min: {min}");
            Console.WriteLine($"Max: {max}");
        }

        // Method 3 with semantically similar code
        public void FilterData(List<int> items)
        {
            // Semantically similar to duplicated block 1
            if (items == null)
            {
                throw new ArgumentNullException(nameof(items));
            }

            if (!items.Any())
            {
                Console.WriteLine("No items to filter");
                return;
            }

            // Filter the data
            var evenItems = items.Where(i => i % 2 == 0).ToList();
            var oddItems = items.Where(i => i % 2 != 0).ToList();

            Console.WriteLine($"Even items: {string.Join(", ", evenItems)}");
            Console.WriteLine($"Odd items: {string.Join(", ", oddItems)}");
        }

        // Method 4 with duplicated code
        public void SortData(List<int> data)
        {
            // Duplicated block 1 - Data validation
            if (data == null)
            {
                throw new ArgumentNullException(nameof(data));
            }

            if (data.Count == 0)
            {
                Console.WriteLine("No data to sort");
                return;
            }

            // Sort the data
            var sortedData = data.OrderBy(i => i).ToList();

            Console.WriteLine($"Sorted data: {string.Join(", ", sortedData)}");
        }

        // Method 5 with duplicated code within the method
        public void ProcessLargeData(List<int> data)
        {
            // Duplicated block 1 - Data validation
            if (data == null)
            {
                throw new ArgumentNullException(nameof(data));
            }

            if (data.Count == 0)
            {
                Console.WriteLine("No data to process");
                return;
            }

            // Process the first half
            var firstHalf = data.Take(data.Count / 2).ToList();
            var sum1 = 0;
            foreach (var item in firstHalf)
            {
                sum1 += item;
            }

            Console.WriteLine($"First half sum: {sum1}");
            Console.WriteLine($"First half average: {sum1 / firstHalf.Count}");

            // Process the second half (duplicated code within the method)
            var secondHalf = data.Skip(data.Count / 2).ToList();
            var sum2 = 0;
            foreach (var item in secondHalf)
            {
                sum2 += item;
            }

            Console.WriteLine($"Second half sum: {sum2}");
            Console.WriteLine($"Second half average: {sum2 / secondHalf.Count}");
        }
    }
}
