using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;

namespace DuplicationDemo;

class Program
{
    static async Task Main(string[] args)
    {
        // Setup logging
        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder.AddConsole();
        });
        var logger = loggerFactory.CreateLogger<DuplicationAnalyzer>();

        // Create the analyzer
        var analyzer = new DuplicationAnalyzer(logger);

        // Create a demo file
        var demoFilePath = "demo.cs";
        await File.WriteAllTextAsync(demoFilePath, CreateDemoCode());
        Console.WriteLine($"Created demo file: {demoFilePath}");

        // Run the analyzer on the demo file
        Console.WriteLine("\nAnalyzing duplication in the demo file...");
        var demoResult = await analyzer.AnalyzeFileAsync(demoFilePath);

        // Display the results
        Console.WriteLine(demoResult.GetReport());

        // Create the complex sample file
        var complexSamplePath = "ComplexSample.cs";
        await File.WriteAllTextAsync(complexSamplePath, CreateComplexSampleCode());
        Console.WriteLine($"Created complex sample file: {complexSamplePath}");

        // Run the analyzer on the complex sample
        Console.WriteLine("\nAnalyzing duplication in the complex sample...");
        var complexResult = await analyzer.AnalyzeFileAsync(complexSamplePath);

        // Display the results
        Console.WriteLine(complexResult.GetReport());

        // Generate HTML reports
        var demoHtmlPath = "demo_duplication_report.html";
        await File.WriteAllTextAsync(demoHtmlPath, demoResult.GetHtmlReport());
        Console.WriteLine($"\nDemo HTML report saved to {demoHtmlPath}");

        var complexHtmlPath = "complex_duplication_report.html";
        await File.WriteAllTextAsync(complexHtmlPath, complexResult.GetHtmlReport());
        Console.WriteLine($"\nComplex sample HTML report saved to {complexHtmlPath}");

        // Open the HTML reports in the default browser
        Console.WriteLine("\nOpening HTML reports in browser...");
        try
        {
            // Open demo report
            var demoPsi = new System.Diagnostics.ProcessStartInfo
            {
                FileName = demoHtmlPath,
                UseShellExecute = true
            };
            System.Diagnostics.Process.Start(demoPsi);

            // Wait a moment before opening the second report
            await Task.Delay(1000);

            // Open complex sample report
            var complexPsi = new System.Diagnostics.ProcessStartInfo
            {
                FileName = complexHtmlPath,
                UseShellExecute = true
            };
            System.Diagnostics.Process.Start(complexPsi);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error opening HTML reports: {ex.Message}");
        }

        Console.WriteLine("\nPress any key to exit...");
        Console.ReadKey();
    }

    static string CreateDemoCode()
    {
        return @"using System;

namespace DuplicationDemo
{
    public class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine(""Hello, World!"");

            // Duplicated code block 1
            int a = 1;
            int b = 2;
            int c = a + b;
            Console.WriteLine($""The sum of {a} and {b} is {c}"");

            // Some other code
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine(i);
            }

            // Duplicated code block 2
            int x = 1;
            int y = 2;
            int z = x + y;
            Console.WriteLine($""The sum of {x} and {y} is {z}"");

            // Semantically similar code
            var first = 10;
            var second = 20;
            var result = first + second;
            Console.WriteLine($""Adding {first} and {second} gives {result}"");
        }

        static void AnotherMethod()
        {
            // Duplicated code block 3
            int a = 1;
            int b = 2;
            int c = a + b;
            Console.WriteLine($""The sum of {a} and {b} is {c}"");
        }
    }
}";
    }

    static string CreateComplexSampleCode()
    {
        return @"using System;
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
                Console.WriteLine(""No data to process"");
                return;
            }

            // Process the data
            var sum = 0;
            foreach (var item in data)
            {
                sum += item;
            }

            Console.WriteLine($""Sum: {sum}"");
            Console.WriteLine($""Average: {sum / data.Count}"");
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
                Console.WriteLine(""No data to analyze"");
                return;
            }

            // Analyze the data
            var min = data.Min();
            var max = data.Max();

            Console.WriteLine($""Min: {min}"");
            Console.WriteLine($""Max: {max}"");
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
                Console.WriteLine(""No items to filter"");
                return;
            }

            // Filter the data
            var evenItems = items.Where(i => i % 2 == 0).ToList();
            var oddItems = items.Where(i => i % 2 != 0).ToList();

            Console.WriteLine($""Even items: {string.Join("", "", evenItems)}"");
            Console.WriteLine($""Odd items: {string.Join("", "", oddItems)}"");
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
                Console.WriteLine(""No data to sort"");
                return;
            }

            // Sort the data
            var sortedData = data.OrderBy(i => i).ToList();

            Console.WriteLine($""Sorted data: {string.Join("", "", sortedData)}"");
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
                Console.WriteLine(""No data to process"");
                return;
            }

            // Process the first half
            var firstHalf = data.Take(data.Count / 2).ToList();
            var sum1 = 0;
            foreach (var item in firstHalf)
            {
                sum1 += item;
            }

            Console.WriteLine($""First half sum: {sum1}"");
            Console.WriteLine($""First half average: {sum1 / firstHalf.Count}"");

            // Process the second half (duplicated code within the method)
            var secondHalf = data.Skip(data.Count / 2).ToList();
            var sum2 = 0;
            foreach (var item in secondHalf)
            {
                sum2 += item;
            }

            Console.WriteLine($""Second half sum: {sum2}"");
            Console.WriteLine($""Second half average: {sum2 / secondHalf.Count}"");
        }
    }
}";
    }
}