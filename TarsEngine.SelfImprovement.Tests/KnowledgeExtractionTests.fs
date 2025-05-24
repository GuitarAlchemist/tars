namespace TarsEngine.SelfImprovement.Tests

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Xunit
open Xunit.Abstractions
open Moq
open TarsEngine.SelfImprovement

/// <summary>
/// Tests for the knowledge extraction functionality.
///
/// These tests demonstrate how TARS extracts structured knowledge from exploration chats and documentation.
/// The knowledge extraction process identifies:
/// - Concepts: Key definitions and terminology
/// - Insights: Important observations and realizations
/// - Code Patterns: Reusable code snippets and examples
///
/// This knowledge is then used by the autonomous improvement system to enhance the codebase.
/// </summary>
module KnowledgeExtractionTests =

    /// <summary>
    /// Test class that provides access to test output
    /// </summary>
    type KnowledgeExtractionTestsWithOutput(output: ITestOutputHelper) =

        /// <summary>
        /// Demonstrates how TARS extracts concepts from text content.
        ///
        /// Concepts are key definitions and terminology that form the foundation of knowledge.
        /// They are identified using pattern matching on phrases like "concept of X", "X is defined as", etc.
        /// </summary>
        [<Fact>]
        member _.``DEMO: Extract concepts from exploration text`` () =
            // Arrange - Sample text that might appear in exploration chats
            let content = "The concept of knowledge extraction is important. AI is defined as artificial intelligence."
            output.WriteLine("Input text for concept extraction:")
            output.WriteLine(content)
            output.WriteLine("")

            // Act - Extract concepts using the KnowledgeExtractor
            let concepts = KnowledgeExtractor.extractConcepts content

            // Display extracted concepts
            output.WriteLine("Extracted concepts:")
            concepts |> List.iter (fun c ->
                output.WriteLine($"- {c.Content} (Confidence: {c.Confidence})")
            )
            output.WriteLine("")

            // Assert - Verify that concepts were correctly extracted
            Assert.NotEmpty(concepts)
            Assert.Contains(concepts, fun c -> c.Content.Contains("knowledge extraction"))
            Assert.Contains(concepts, fun c -> c.Content.Contains("AI"))

            // Explain the significance
            output.WriteLine("These concepts will be stored in the knowledge base and used to improve the codebase.")

        /// <summary>
        /// Demonstrates how TARS extracts insights from text content.
        ///
        /// Insights are important observations and realizations that provide deeper understanding.
        /// They are identified using pattern matching on phrases like "I realized that X", "key insight is that X", etc.
        /// </summary>
        [<Fact>]
        member _.``DEMO: Extract insights from exploration text`` () =
            // Arrange - Sample text that might appear in exploration chats
            let content = "I realized that pattern matching is powerful. It's important to note that functional programming promotes immutability."
            output.WriteLine("Input text for insight extraction:")
            output.WriteLine(content)
            output.WriteLine("")

            // Act - Extract insights using the KnowledgeExtractor
            let insights = KnowledgeExtractor.extractInsights content

            // Display extracted insights
            output.WriteLine("Extracted insights:")
            insights |> List.iter (fun i ->
                output.WriteLine($"- {i.Content} (Confidence: {i.Confidence})")
            )
            output.WriteLine("")

            // Assert - Verify that insights were correctly extracted
            Assert.NotEmpty(insights)
            Assert.Contains(insights, fun i -> i.Content.Contains("pattern matching is powerful"))
            Assert.Contains(insights, fun i -> i.Content.Contains("functional programming promotes immutability"))

            // Explain the significance
            output.WriteLine("These insights will guide the autonomous improvement system in making better code enhancement decisions.")

        /// <summary>
        /// Demonstrates how TARS extracts code patterns from text content.
        ///
        /// Code patterns are reusable code snippets and examples that demonstrate best practices.
        /// They are identified by extracting code blocks from markdown content.
        /// </summary>
        [<Fact>]
        member _.``DEMO: Extract code patterns from exploration text`` () =
            // Arrange - Sample text with code blocks that might appear in exploration chats
            let content = "Here's an example:\n```csharp\npublic class Example { }\n```\nAnd another:\n```fsharp\nlet add x y = x + y\n```"
            output.WriteLine("Input text with code blocks:")
            output.WriteLine(content)
            output.WriteLine("")

            // Act - Extract code patterns using the KnowledgeExtractor
            let codePatterns = KnowledgeExtractor.extractCodePatterns content

            // Display extracted code patterns
            output.WriteLine("Extracted code patterns:")
            codePatterns |> List.iter (fun c ->
                output.WriteLine($"- Language pattern: {c.Content} (Confidence: {c.Confidence})")
            )
            output.WriteLine("")

            // Assert - Verify that code patterns were correctly extracted
            Assert.Equal(2, codePatterns.Length)
            Assert.Contains(codePatterns, fun c -> c.Content.Contains("public class Example"))
            Assert.Contains(codePatterns, fun c -> c.Content.Contains("let add x y = x + y"))

            // Explain the significance
            output.WriteLine("These code patterns will be used as templates for code generation and improvement.")

        /// <summary>
        /// Demonstrates how TARS identifies the source type of a file based on its path.
        ///
        /// Different source types (Chat, Reflection, Documentation, etc.) are processed differently
        /// and have different levels of authority in the knowledge base.
        /// </summary>
        [<Fact>]
        member _.``DEMO: Identify source types from file paths`` () =
            // Arrange - Sample file paths from different source types
            let paths = [
                "docs/Explorations/v1/Chats/example.md", "Chat"
                "docs/Explorations/Reflections/example.md", "Reflection"
                "docs/features/example.md", "Feature"
                "docs/architecture/example.md", "Architecture"
                "docs/tutorials/example.md", "Tutorial"
                "docs/example.md", "Documentation"
            ]

            output.WriteLine("File path classification:")

            // Act & Assert - Verify that source types are correctly identified
            for (path, expectedType) in paths do
                let sourceType = ExplorationFileProcessor.determineSourceType path
                let sourceTypeStr = sourceType.ToString()

                output.WriteLine($"- {path} → {sourceTypeStr}")
                Assert.Equal(expectedType, sourceTypeStr)

            // Explain the significance
            output.WriteLine("")
            output.WriteLine("Source type classification helps TARS prioritize knowledge from different sources.")
            output.WriteLine("For example, insights from Chats might be weighted differently than official Documentation.")
