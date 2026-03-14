using System;
using System.IO;
using System.Threading.Tasks;
using DuplicationAnalyzerTests;

namespace DuplicationAnalyzerDemo;

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("=== Code Duplication Detection Demo ===");
        Console.WriteLine();
        
        if (args.Length < 1)
        {
            Console.WriteLine("Usage: DuplicationAnalyzerDemo <file_path> [language] [type] [output] [output_path]");
            Console.WriteLine("  file_path: Path to the file or directory to analyze");
            Console.WriteLine("  language: Programming language (C# or F#, default: C#)");
            Console.WriteLine("  type: Duplication type (token, semantic, or all, default: all)");
            Console.WriteLine("  output: Output format (console, json, csv, or html, default: console)");
            Console.WriteLine("  output_path: Path to save the output file (optional)");
            return;
        }
        
        var filePath = args[0];
        var language = args.Length > 1 ? args[1] : "C#";
        var type = args.Length > 2 ? args[2] : "all";
        var output = args.Length > 3 ? args[3] : "console";
        var outputPath = args.Length > 4 ? args[4] : string.Empty;
        
        // Create a demo file if no file is specified
        if (filePath == "demo")
        {
            filePath = CreateDemoFile();
            Console.WriteLine($"Created demo file: {filePath}");
        }
        
        // Run the demo
        var command = new DuplicationDemoCommand();
        var result = await command.RunAsync(filePath, language, type, output, outputPath);
        
        // Display the result
        Console.WriteLine(result);
    }
    
    static string CreateDemoFile()
    {
        var demoCode = @"
using System;

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
        
        var filePath = Path.Combine(Path.GetTempPath(), "DuplicationDemo.cs");
        File.WriteAllText(filePath, demoCode);
        return filePath;
    }
}
