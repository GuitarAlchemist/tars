using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services
{
    /// <summary>
    /// Service for demonstrating TARS capabilities
    /// </summary>
    public class DemoService
    {
        private readonly ILogger<DemoService> _logger;
        private readonly IConfiguration _configuration;
        private readonly SelfImprovementService _selfImprovementService;
        private readonly OllamaService _ollamaService;
        private readonly HuggingFaceService _huggingFaceService;
        private readonly LanguageSpecificationService _languageSpecificationService;
        private readonly DocumentationService _documentationService;
        private readonly string _demoDir;

        public DemoService(
            ILogger<DemoService> logger,
            IConfiguration configuration,
            SelfImprovementService selfImprovementService,
            OllamaService ollamaService,
            HuggingFaceService huggingFaceService,
            LanguageSpecificationService languageSpecificationService,
            DocumentationService documentationService)
        {
            _logger = logger;
            _configuration = configuration;
            _selfImprovementService = selfImprovementService;
            _ollamaService = ollamaService;
            _huggingFaceService = huggingFaceService;
            _languageSpecificationService = languageSpecificationService;
            _documentationService = documentationService;
            
            // Create demo directory
            var projectRoot = _configuration["Tars:ProjectRoot"] ?? Directory.GetCurrentDirectory();
            _demoDir = Path.Combine(projectRoot, "demo");
            Directory.CreateDirectory(_demoDir);
        }

        /// <summary>
        /// Run a demonstration of TARS capabilities
        /// </summary>
        public async Task<bool> RunDemoAsync(string demoType, string model)
        {
            try
            {
                _logger.LogInformation($"Running {demoType} demo with model {model}");
                
                CliSupport.WriteHeader($"TARS Demonstration - {demoType}");
                Console.WriteLine($"Model: {model}");
                Console.WriteLine();
                
                switch (demoType.ToLowerInvariant())
                {
                    case "self-improvement":
                        return await RunSelfImprovementDemoAsync(model);
                    case "code-generation":
                        return await RunCodeGenerationDemoAsync(model);
                    case "language-specs":
                        return await RunLanguageSpecsDemoAsync();
                    case "all":
                        var success = await RunSelfImprovementDemoAsync(model);
                        if (!success) return false;
                        
                        success = await RunCodeGenerationDemoAsync(model);
                        if (!success) return false;
                        
                        success = await RunLanguageSpecsDemoAsync();
                        if (!success) return false;
                        
                        return true;
                    default:
                        CliSupport.WriteColorLine($"Unknown demo type: {demoType}", ConsoleColor.Red);
                        Console.WriteLine("Available demo types:");
                        Console.WriteLine("  - self-improvement: Demonstrate code analysis and improvement");
                        Console.WriteLine("  - code-generation: Demonstrate code generation capabilities");
                        Console.WriteLine("  - language-specs: Demonstrate language specification generation");
                        Console.WriteLine("  - all: Run all demos");
                        return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error running {demoType} demo");
                CliSupport.WriteColorLine($"Error running demo: {ex.Message}", ConsoleColor.Red);
                return false;
            }
        }

        /// <summary>
        /// Run a demonstration of self-improvement capabilities
        /// </summary>
        private async Task<bool> RunSelfImprovementDemoAsync(string model)
        {
            try
            {
                CliSupport.WriteColorLine("Self-Improvement Demonstration", ConsoleColor.Cyan);
                Console.WriteLine("This demo will show how TARS can analyze and improve code.");
                Console.WriteLine();
                
                // Create a demo file with code issues
                var demoFilePath = Path.Combine(_demoDir, "demo_code_with_issues.cs");
                await File.WriteAllTextAsync(demoFilePath, GetDemoCodeWithIssues());
                
                CliSupport.WriteColorLine("Created demo file with code issues:", ConsoleColor.Yellow);
                Console.WriteLine(demoFilePath);
                Console.WriteLine();
                
                // Display the original code
                CliSupport.WriteColorLine("Original Code:", ConsoleColor.Yellow);
                var originalCode = await File.ReadAllTextAsync(demoFilePath);
                Console.WriteLine(originalCode);
                Console.WriteLine();
                
                // Step 1: Analyze the code
                CliSupport.WriteColorLine("Step 1: Analyzing the code...", ConsoleColor.Green);
                Console.WriteLine();
                
                var analyzeSuccess = await _selfImprovementService.AnalyzeFile(demoFilePath, model);
                if (!analyzeSuccess)
                {
                    CliSupport.WriteColorLine("Failed to analyze the code.", ConsoleColor.Red);
                    return false;
                }
                
                Console.WriteLine();
                
                // Step 2: Propose improvements
                CliSupport.WriteColorLine("Step 2: Proposing improvements...", ConsoleColor.Green);
                Console.WriteLine();
                
                var proposeSuccess = await _selfImprovementService.ProposeImprovement(demoFilePath, model, true);
                if (!proposeSuccess)
                {
                    CliSupport.WriteColorLine("Failed to propose improvements.", ConsoleColor.Red);
                    return false;
                }
                
                Console.WriteLine();
                
                // Step 3: Show the improved code
                CliSupport.WriteColorLine("Step 3: Improved Code:", ConsoleColor.Green);
                var improvedCode = await File.ReadAllTextAsync(demoFilePath);
                Console.WriteLine(improvedCode);
                
                CliSupport.WriteColorLine("Self-Improvement Demo Completed Successfully!", ConsoleColor.Cyan);
                Console.WriteLine();
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running self-improvement demo");
                CliSupport.WriteColorLine($"Error running self-improvement demo: {ex.Message}", ConsoleColor.Red);
                return false;
            }
        }

        /// <summary>
        /// Run a demonstration of code generation capabilities
        /// </summary>
        private async Task<bool> RunCodeGenerationDemoAsync(string model)
        {
            try
            {
                CliSupport.WriteColorLine("Code Generation Demonstration", ConsoleColor.Cyan);
                Console.WriteLine("This demo will show how TARS can generate code based on natural language descriptions.");
                Console.WriteLine();
                
                // Step 1: Generate a simple class
                CliSupport.WriteColorLine("Step 1: Generating a simple class...", ConsoleColor.Green);
                Console.WriteLine();
                
                var prompt = "Generate a C# class called 'WeatherForecast' with properties for Date (DateTime), TemperatureC (int), TemperatureF (calculated from TemperatureC), and Summary (string). Include appropriate XML documentation comments.";
                Console.WriteLine($"Prompt: {prompt}");
                Console.WriteLine();
                
                var response = await _ollamaService.GenerateCompletion(prompt, model);
                
                // Extract the code from the response
                var code = ExtractCodeFromResponse(response);
                
                // Save the generated code to a file
                var demoFilePath = Path.Combine(_demoDir, "WeatherForecast.cs");
                await File.WriteAllTextAsync(demoFilePath, code);
                
                CliSupport.WriteColorLine("Generated Code:", ConsoleColor.Yellow);
                Console.WriteLine(code);
                Console.WriteLine();
                
                // Step 2: Generate a more complex example
                CliSupport.WriteColorLine("Step 2: Generating a more complex example...", ConsoleColor.Green);
                Console.WriteLine();
                
                prompt = "Generate a C# implementation of a simple in-memory cache with generic type support. Include methods for Add, Get, Remove, and Clear. Implement a time-based expiration mechanism for cache entries.";
                Console.WriteLine($"Prompt: {prompt}");
                Console.WriteLine();
                
                response = await _ollamaService.GenerateCompletion(prompt, model);
                
                // Extract the code from the response
                code = ExtractCodeFromResponse(response);
                
                // Save the generated code to a file
                demoFilePath = Path.Combine(_demoDir, "MemoryCache.cs");
                await File.WriteAllTextAsync(demoFilePath, code);
                
                CliSupport.WriteColorLine("Generated Code:", ConsoleColor.Yellow);
                Console.WriteLine(code);
                Console.WriteLine();
                
                CliSupport.WriteColorLine("Code Generation Demo Completed Successfully!", ConsoleColor.Cyan);
                Console.WriteLine();
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running code generation demo");
                CliSupport.WriteColorLine($"Error running code generation demo: {ex.Message}", ConsoleColor.Red);
                return false;
            }
        }

        /// <summary>
        /// Run a demonstration of language specification generation
        /// </summary>
        private async Task<bool> RunLanguageSpecsDemoAsync()
        {
            try
            {
                CliSupport.WriteColorLine("Language Specification Demonstration", ConsoleColor.Cyan);
                Console.WriteLine("This demo will show how TARS can generate language specifications for its DSL.");
                Console.WriteLine();
                
                // Step 1: Generate EBNF specification
                CliSupport.WriteColorLine("Step 1: Generating EBNF specification...", ConsoleColor.Green);
                Console.WriteLine();
                
                var ebnf = await _languageSpecificationService.GenerateEbnfAsync();
                var ebnfPath = Path.Combine(_demoDir, "tars_grammar.ebnf");
                await File.WriteAllTextAsync(ebnfPath, ebnf);
                
                CliSupport.WriteColorLine($"EBNF specification saved to: {ebnfPath}", ConsoleColor.Yellow);
                Console.WriteLine("Preview:");
                Console.WriteLine(ebnf.Substring(0, Math.Min(500, ebnf.Length)) + "...");
                Console.WriteLine();
                
                // Step 2: Generate BNF specification
                CliSupport.WriteColorLine("Step 2: Generating BNF specification...", ConsoleColor.Green);
                Console.WriteLine();
                
                var bnf = await _languageSpecificationService.GenerateBnfAsync();
                var bnfPath = Path.Combine(_demoDir, "tars_grammar.bnf");
                await File.WriteAllTextAsync(bnfPath, bnf);
                
                CliSupport.WriteColorLine($"BNF specification saved to: {bnfPath}", ConsoleColor.Yellow);
                Console.WriteLine("Preview:");
                Console.WriteLine(bnf.Substring(0, Math.Min(500, bnf.Length)) + "...");
                Console.WriteLine();
                
                // Step 3: Generate JSON schema
                CliSupport.WriteColorLine("Step 3: Generating JSON schema...", ConsoleColor.Green);
                Console.WriteLine();
                
                var schema = await _languageSpecificationService.GenerateJsonSchemaAsync();
                var schemaPath = Path.Combine(_demoDir, "tars_schema.json");
                await File.WriteAllTextAsync(schemaPath, schema);
                
                CliSupport.WriteColorLine($"JSON schema saved to: {schemaPath}", ConsoleColor.Yellow);
                Console.WriteLine("Preview:");
                Console.WriteLine(schema.Substring(0, Math.Min(500, schema.Length)) + "...");
                Console.WriteLine();
                
                // Step 4: Generate markdown documentation
                CliSupport.WriteColorLine("Step 4: Generating markdown documentation...", ConsoleColor.Green);
                Console.WriteLine();
                
                var docs = await _languageSpecificationService.GenerateMarkdownDocumentationAsync();
                var docsPath = Path.Combine(_demoDir, "tars_dsl_docs.md");
                await File.WriteAllTextAsync(docsPath, docs);
                
                CliSupport.WriteColorLine($"Markdown documentation saved to: {docsPath}", ConsoleColor.Yellow);
                Console.WriteLine("Preview:");
                Console.WriteLine(docs.Substring(0, Math.Min(500, docs.Length)) + "...");
                Console.WriteLine();
                
                CliSupport.WriteColorLine("Language Specification Demo Completed Successfully!", ConsoleColor.Cyan);
                Console.WriteLine();
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running language specification demo");
                CliSupport.WriteColorLine($"Error running language specification demo: {ex.Message}", ConsoleColor.Red);
                return false;
            }
        }

        /// <summary>
        /// Extract code from a response
        /// </summary>
        private string ExtractCodeFromResponse(string response)
        {
            // Look for code blocks
            var codeBlockStart = response.IndexOf("```csharp");
            if (codeBlockStart == -1)
            {
                codeBlockStart = response.IndexOf("```c#");
            }
            if (codeBlockStart == -1)
            {
                codeBlockStart = response.IndexOf("```");
            }
            
            if (codeBlockStart != -1)
            {
                var codeStart = response.IndexOf('\n', codeBlockStart) + 1;
                var codeBlockEnd = response.IndexOf("```", codeStart);
                
                if (codeBlockEnd != -1)
                {
                    return response.Substring(codeStart, codeBlockEnd - codeStart).Trim();
                }
            }
            
            // If no code block is found, return the entire response
            return response;
        }

        /// <summary>
        /// Get demo code with issues
        /// </summary>
        private string GetDemoCodeWithIssues()
        {
            return @"using System;
using System.Collections.Generic;

namespace DemoCode
{
    public class Program
    {
        public static void Main(string[] args)
        {
            // This is a demo program with some issues to be improved
            Console.WriteLine(""Hello, World!"");
            
            // Issue 1: Magic numbers
            int timeout = 300;
            
            // Issue 2: Inefficient string concatenation in loop
            string result = """";
            for (int i = 0; i < 100; i++)
            {
                result += i.ToString();
            }
            
            // Issue 3: Empty catch block
            try
            {
                int x = int.Parse(""abc"");
            }
            catch (Exception)
            {
                // Empty catch block
            }
            
            // Issue 4: Unused variable
            var unusedList = new List<string>();
            
            Console.WriteLine(result);
            Console.WriteLine($""Timeout is set to {timeout} seconds"");
        }
    }
}";
        }
    }
}
