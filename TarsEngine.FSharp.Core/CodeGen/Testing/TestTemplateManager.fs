namespace TarsEngine.FSharp.Core.CodeGen.Testing

open System
open System.Collections.Generic
open System.IO
open Microsoft.Extensions.Logging

/// <summary>
/// Represents a test template.
/// </summary>
type TestTemplate = {
    /// <summary>
    /// The name of the template.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the template.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The language of the template.
    /// </summary>
    Language: string
    
    /// <summary>
    /// The test framework of the template.
    /// </summary>
    TestFramework: string
    
    /// <summary>
    /// The content of the template.
    /// </summary>
    Content: string
    
    /// <summary>
    /// The placeholders in the template.
    /// </summary>
    Placeholders: Map<string, string>
}

/// <summary>
/// Manager for test templates.
/// </summary>
type TestTemplateManager(logger: ILogger<TestTemplateManager>) =
    
    // Dictionary of templates by language and framework
    let templates = Dictionary<string, Dictionary<string, TestTemplate>>()
    
    // Initialize with default templates
    do
        // Add C# xUnit template
        let csharpXUnitTemplate = {
            Name = "CSharpXUnitTemplate"
            Description = "Template for C# xUnit tests"
            Language = "csharp"
            TestFramework = "xunit"
            Content = """
                using System;
                using Xunit;
                
                namespace ${Namespace}
                {
                    public class ${TestClassName}
                    {
                        [Fact]
                        public void ${TestMethodName}()
                        {
                            // Arrange
                            ${ArrangeCode}
                            
                            // Act
                            ${ActCode}
                            
                            // Assert
                            ${AssertCode}
                        }
                    }
                }
                """
            Placeholders = Map.ofList [
                "Namespace", "The namespace of the test class"
                "TestClassName", "The name of the test class"
                "TestMethodName", "The name of the test method"
                "ArrangeCode", "The code to arrange the test"
                "ActCode", "The code to act in the test"
                "AssertCode", "The code to assert in the test"
            ]
        }
        
        // Add C# NUnit template
        let csharpNUnitTemplate = {
            Name = "CSharpNUnitTemplate"
            Description = "Template for C# NUnit tests"
            Language = "csharp"
            TestFramework = "nunit"
            Content = """
                using System;
                using NUnit.Framework;
                
                namespace ${Namespace}
                {
                    [TestFixture]
                    public class ${TestClassName}
                    {
                        [Test]
                        public void ${TestMethodName}()
                        {
                            // Arrange
                            ${ArrangeCode}
                            
                            // Act
                            ${ActCode}
                            
                            // Assert
                            ${AssertCode}
                        }
                    }
                }
                """
            Placeholders = Map.ofList [
                "Namespace", "The namespace of the test class"
                "TestClassName", "The name of the test class"
                "TestMethodName", "The name of the test method"
                "ArrangeCode", "The code to arrange the test"
                "ActCode", "The code to act in the test"
                "AssertCode", "The code to assert in the test"
            ]
        }
        
        // Add C# MSTest template
        let csharpMSTestTemplate = {
            Name = "CSharpMSTestTemplate"
            Description = "Template for C# MSTest tests"
            Language = "csharp"
            TestFramework = "mstest"
            Content = """
                using System;
                using Microsoft.VisualStudio.TestTools.UnitTesting;
                
                namespace ${Namespace}
                {
                    [TestClass]
                    public class ${TestClassName}
                    {
                        [TestMethod]
                        public void ${TestMethodName}()
                        {
                            // Arrange
                            ${ArrangeCode}
                            
                            // Act
                            ${ActCode}
                            
                            // Assert
                            ${AssertCode}
                        }
                    }
                }
                """
            Placeholders = Map.ofList [
                "Namespace", "The namespace of the test class"
                "TestClassName", "The name of the test class"
                "TestMethodName", "The name of the test method"
                "ArrangeCode", "The code to arrange the test"
                "ActCode", "The code to act in the test"
                "AssertCode", "The code to assert in the test"
            ]
        }
        
        // Add F# xUnit template
        let fsharpXUnitTemplate = {
            Name = "FSharpXUnitTemplate"
            Description = "Template for F# xUnit tests"
            Language = "fsharp"
            TestFramework = "xunit"
            Content = """
                module ${ModuleName}

                open System
                open Xunit
                
                [<Fact>]
                let ``${TestName}`` () =
                    // Arrange
                    ${ArrangeCode}
                    
                    // Act
                    ${ActCode}
                    
                    // Assert
                    ${AssertCode}
                """
            Placeholders = Map.ofList [
                "ModuleName", "The name of the test module"
                "TestName", "The name of the test"
                "ArrangeCode", "The code to arrange the test"
                "ActCode", "The code to act in the test"
                "AssertCode", "The code to assert in the test"
            ]
        }
        
        // Add F# NUnit template
        let fsharpNUnitTemplate = {
            Name = "FSharpNUnitTemplate"
            Description = "Template for F# NUnit tests"
            Language = "fsharp"
            TestFramework = "nunit"
            Content = """
                module ${ModuleName}

                open System
                open NUnit.Framework
                
                [<Test>]
                let ``${TestName}`` () =
                    // Arrange
                    ${ArrangeCode}
                    
                    // Act
                    ${ActCode}
                    
                    // Assert
                    ${AssertCode}
                """
            Placeholders = Map.ofList [
                "ModuleName", "The name of the test module"
                "TestName", "The name of the test"
                "ArrangeCode", "The code to arrange the test"
                "ActCode", "The code to act in the test"
                "AssertCode", "The code to assert in the test"
            ]
        }
        
        // Add F# MSTest template
        let fsharpMSTestTemplate = {
            Name = "FSharpMSTestTemplate"
            Description = "Template for F# MSTest tests"
            Language = "fsharp"
            TestFramework = "mstest"
            Content = """
                namespace ${Namespace}

                open System
                open Microsoft.VisualStudio.TestTools.UnitTesting
                
                [<TestClass>]
                type ${TestClassName}() =
                    
                    [<TestMethod>]
                    member _.``${TestMethodName}`` () =
                        // Arrange
                        ${ArrangeCode}
                        
                        // Act
                        ${ActCode}
                        
                        // Assert
                        ${AssertCode}
                """
            Placeholders = Map.ofList [
                "Namespace", "The namespace of the test class"
                "TestClassName", "The name of the test class"
                "TestMethodName", "The name of the test method"
                "ArrangeCode", "The code to arrange the test"
                "ActCode", "The code to act in the test"
                "AssertCode", "The code to assert in the test"
            ]
        }
        
        // Add templates to the dictionary
        let addTemplate (template: TestTemplate) =
            if not (templates.ContainsKey(template.Language)) then
                templates.Add(template.Language, Dictionary<string, TestTemplate>())
            
            templates.[template.Language].[template.TestFramework] <- template
        
        addTemplate csharpXUnitTemplate
        addTemplate csharpNUnitTemplate
        addTemplate csharpMSTestTemplate
        addTemplate fsharpXUnitTemplate
        addTemplate fsharpNUnitTemplate
        addTemplate fsharpMSTestTemplate
    
    /// <summary>
    /// Gets a template by language and test framework.
    /// </summary>
    /// <param name="language">The language of the template.</param>
    /// <param name="testFramework">The test framework of the template.</param>
    /// <returns>The template, if found.</returns>
    member _.GetTemplate(language: string, testFramework: string) =
        if templates.ContainsKey(language) && templates.[language].ContainsKey(testFramework) then
            Some templates.[language].[testFramework]
        else
            None
    
    /// <summary>
    /// Gets all templates for a language.
    /// </summary>
    /// <param name="language">The language to get templates for.</param>
    /// <returns>The list of templates for the language.</returns>
    member _.GetTemplatesForLanguage(language: string) =
        if templates.ContainsKey(language) then
            templates.[language].Values |> Seq.toList
        else
            []
    
    /// <summary>
    /// Gets all templates for a test framework.
    /// </summary>
    /// <param name="testFramework">The test framework to get templates for.</param>
    /// <returns>The list of templates for the test framework.</returns>
    member _.GetTemplatesForTestFramework(testFramework: string) =
        templates.Values
        |> Seq.collect (fun dict -> dict.Values)
        |> Seq.filter (fun template -> template.TestFramework = testFramework)
        |> Seq.toList
    
    /// <summary>
    /// Gets all templates.
    /// </summary>
    /// <returns>The list of all templates.</returns>
    member _.GetAllTemplates() =
        templates.Values
        |> Seq.collect (fun dict -> dict.Values)
        |> Seq.toList
    
    /// <summary>
    /// Adds a template.
    /// </summary>
    /// <param name="template">The template to add.</param>
    member _.AddTemplate(template: TestTemplate) =
        if not (templates.ContainsKey(template.Language)) then
            templates.Add(template.Language, Dictionary<string, TestTemplate>())
        
        templates.[template.Language].[template.TestFramework] <- template
    
    /// <summary>
    /// Removes a template.
    /// </summary>
    /// <param name="language">The language of the template to remove.</param>
    /// <param name="testFramework">The test framework of the template to remove.</param>
    /// <returns>Whether the template was removed.</returns>
    member _.RemoveTemplate(language: string, testFramework: string) =
        if templates.ContainsKey(language) && templates.[language].ContainsKey(testFramework) then
            templates.[language].Remove(testFramework)
        else
            false
    
    /// <summary>
    /// Generates a test from a template.
    /// </summary>
    /// <param name="template">The template to use.</param>
    /// <param name="placeholderValues">The values for placeholders.</param>
    /// <returns>The generated test code.</returns>
    member _.GenerateTest(template: TestTemplate, placeholderValues: Map<string, string>) =
        try
            logger.LogInformation("Generating test from template: {TemplateName}", template.Name)
            
            // Replace placeholders in the template
            let mutable generatedCode = template.Content
            
            for placeholder in template.Placeholders do
                let placeholderName = placeholder.Key
                let placeholderDescription = placeholder.Value
                
                // Get the value for the placeholder
                match placeholderValues.TryFind placeholderName with
                | Some value ->
                    // Replace the placeholder with the value
                    generatedCode <- generatedCode.Replace($"${{{placeholderName}}}", value)
                | None ->
                    logger.LogWarning("No value provided for placeholder: {PlaceholderName}", placeholderName)
            
            generatedCode
        with
        | ex ->
            logger.LogError(ex, "Error generating test from template: {TemplateName}", template.Name)
            $"Error generating test: {ex.Message}"
