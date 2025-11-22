using System;
using System.Threading.Tasks;
using DuplicationAnalyzerTests;

class RunDemo
{
    static async Task Main(string[] args)
    {
        var command = new DuplicationDemoCommand();
        var result = await command.RunAsync("demo.cs", "C#", "all", "console");
        Console.WriteLine(result);
    }
}
