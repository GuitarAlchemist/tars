using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Threading.Tasks;

namespace TarsTestGenerator
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            Console.WriteLine("TARS Test Generator");
            Console.WriteLine("A tool for generating unit tests for C# code");
            Console.WriteLine();

            // Create a root command
            var rootCommand = new RootCommand("TARS Test Generator - A tool for generating unit tests for C# code");

            // Add options
            var inputFileOption = new Option<FileInfo>(
                "--input-file",
                "The input C# file to generate tests for"
            )
            {
                IsRequired = true
            };

            var outputFileOption = new Option<FileInfo>(
                "--output-file",
                "The output file to write the generated tests to"
            );

            var classNameOption = new Option<string>(
                "--class-name",
                "The name of the class to generate tests for (optional)"
            );

            rootCommand.AddOption(inputFileOption);
            rootCommand.AddOption(outputFileOption);
            rootCommand.AddOption(classNameOption);

            rootCommand.SetHandler(async (FileInfo inputFile, FileInfo outputFile, string className) =>
            {
                await ExecuteAsync(inputFile, outputFile, className);
            }, inputFileOption, outputFileOption, classNameOption);

            // Execute the command
            return await rootCommand.InvokeAsync(args);
        }

        private static async Task ExecuteAsync(FileInfo inputFile, FileInfo outputFile, string className)
        {
            try
            {
                if (!inputFile.Exists)
                {
                    Console.WriteLine($"Input file {inputFile.FullName} does not exist");
                    return;
                }

                string sourceCode = await File.ReadAllTextAsync(inputFile.FullName);

                // Create the test generator
                var testGenerator = new ImprovedCSharpTestGenerator();

                // Generate the tests
                string testCode = testGenerator.GenerateTests(sourceCode, className);

                if (string.IsNullOrEmpty(testCode))
                {
                    Console.WriteLine("Failed to generate tests");
                    return;
                }

                // Write the tests to the output file or console
                if (outputFile != null)
                {
                    await File.WriteAllTextAsync(outputFile.FullName, testCode);
                    Console.WriteLine($"Tests written to {outputFile.FullName}");
                }
                else
                {
                    Console.WriteLine(testCode);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error executing test generator: {ex.Message}");
            }
        }
    }
}
