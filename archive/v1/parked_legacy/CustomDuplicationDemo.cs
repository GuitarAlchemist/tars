using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;

class CustomDuplicationDemo
{
    static async Task Main(string[] args)
    {
        // Setup logging
        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder.AddConsole();
        });
        var logger = loggerFactory.CreateLogger<DuplicationDemo.DuplicationAnalyzer>();
        
        // Create the analyzer
        var analyzer = new DuplicationDemo.DuplicationAnalyzer(logger);
        
        // Analyze the complex duplication demo file
        var complexFilePath = "ComplexDuplicationDemo.cs";
        Console.WriteLine($"\nAnalyzing duplication in {complexFilePath}...");
        var result = await analyzer.AnalyzeFileAsync(complexFilePath);
        
        // Display the results
        Console.WriteLine(result.GetReport());
        
        // Generate HTML report
        var htmlReportPath = "complex_duplication_report.html";
        await File.WriteAllTextAsync(htmlReportPath, result.GetHtmlReport());
        Console.WriteLine($"\nHTML report saved to {htmlReportPath}");
        
        // Open the HTML report in the default browser
        Console.WriteLine("\nOpening HTML report in browser...");
        try
        {
            var psi = new System.Diagnostics.ProcessStartInfo
            {
                FileName = htmlReportPath,
                UseShellExecute = true
            };
            System.Diagnostics.Process.Start(psi);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error opening HTML report: {ex.Message}");
        }
        
        Console.WriteLine("\nPress any key to exit...");
        Console.ReadKey();
    }
}
