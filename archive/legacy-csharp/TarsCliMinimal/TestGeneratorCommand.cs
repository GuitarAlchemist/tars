using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsCliMinimal
{
    /// <summary>
    /// Command for generating tests using the ImprovedCSharpTestGenerator
    /// </summary>
    public class TestGeneratorCommand : Command
    {
        private readonly ILogger<TestGeneratorCommand> _logger;

        public TestGeneratorCommand(ILogger<TestGeneratorCommand> logger)
            : base("generate-tests", "Generate tests for C# code")
        {
            _logger = logger;

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

            AddOption(inputFileOption);
            AddOption(outputFileOption);
            AddOption(classNameOption);

            this.SetHandler(async (FileInfo inputFile, FileInfo outputFile, string className) =>
            {
                await ExecuteAsync(inputFile, outputFile, className);
            }, inputFileOption, outputFileOption, classNameOption);
        }

        private async Task ExecuteAsync(FileInfo inputFile, FileInfo outputFile, string className)
        {
            try
            {
                if (!inputFile.Exists)
                {
                    _logger.LogError($"Input file {inputFile.FullName} does not exist");
                    return;
                }

                string sourceCode = await File.ReadAllTextAsync(inputFile.FullName);

                // Create the test generator
                var testGeneratorLogger = _logger.GetLogger<ImprovedCSharpTestGenerator>();
                var testGenerator = new ImprovedCSharpTestGenerator(testGeneratorLogger);

                // Generate the tests
                string testCode = testGenerator.GenerateTests(sourceCode, className);

                if (string.IsNullOrEmpty(testCode))
                {
                    _logger.LogError("Failed to generate tests");
                    return;
                }

                // Write the tests to the output file or console
                if (outputFile != null)
                {
                    await File.WriteAllTextAsync(outputFile.FullName, testCode);
                    _logger.LogInformation($"Tests written to {outputFile.FullName}");
                }
                else
                {
                    Console.WriteLine(testCode);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error executing test generator command");
            }
        }
    }
}
