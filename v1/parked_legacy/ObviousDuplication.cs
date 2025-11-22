using System;
using System.Collections.Generic;

namespace ObviousDuplication
{
    public class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Obvious Duplication Demo");
            
            // Duplicated block 1
            int a = 1;
            int b = 2;
            int c = a + b;
            Console.WriteLine($"The sum of {a} and {b} is {c}");
            
            // Some other code
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine(i);
            }
            
            // Duplicated block 2 (exact copy of block 1)
            int x = 1;
            int y = 2;
            int z = x + y;
            Console.WriteLine($"The sum of {x} and {y} is {z}");
            
            ProcessData();
            AnalyzeData();
        }
        
        static void ProcessData()
        {
            // Duplicated validation block 1
            var data = new List<int> { 1, 2, 3 };
            
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
            foreach (var item in data)
            {
                Console.WriteLine($"Processing item: {item}");
            }
        }
        
        static void AnalyzeData()
        {
            // Duplicated validation block 2 (exact copy of validation block 1)
            var data = new List<int> { 4, 5, 6 };
            
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
            foreach (var item in data)
            {
                Console.WriteLine($"Analyzing item: {item}");
            }
        }
    }
}
