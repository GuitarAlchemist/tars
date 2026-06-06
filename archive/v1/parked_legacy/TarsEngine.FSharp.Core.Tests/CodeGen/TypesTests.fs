module TarsEngine.FSharp.Core.Tests.CodeGen.TypesTests

open System
open Xunit
open TarsEngine.FSharp.Core.CodeGen

/// <summary>
/// Tests for the CodeGen.Types module.
/// </summary>
type TypesTests() =
    /// <summary>
    /// Test that a CodeTemplate can be created with valid values.
    /// </summary>
    [<Fact>]
    member _.``CodeTemplate can be created with valid values``() =
        // Arrange
        let id = Guid.NewGuid()
        let name = "Test Template"
        let description = "A test template"
        let language = Language.FSharp
        let content = "let {{name}} = {{value}}"
        let parameters = ["name"; "value"]
        let metadata = Map.empty
        
        // Act
        let template = {
            Id = id
            Name = name
            Description = description
            Language = language
            Content = content
            Parameters = parameters
            Metadata = metadata
        }
        
        // Assert
        Assert.Equal(id, template.Id)
        Assert.Equal(name, template.Name)
        Assert.Equal(description, template.Description)
        Assert.Equal(language, template.Language)
        Assert.Equal(content, template.Content)
        Assert.Equal(parameters, template.Parameters)
        Assert.Equal(metadata, template.Metadata)
    
    /// <summary>
    /// Test that a CodeGeneration can be created with valid values.
    /// </summary>
    [<Fact>]
    member _.``CodeGeneration can be created with valid values``() =
        // Arrange
        let id = Guid.NewGuid()
        let templateId = Guid.NewGuid()
        let parameters = Map.ofList [("name", "testVar"); ("value", "42")]
        let generatedCode = "let testVar = 42"
        let generationTime = DateTime.UtcNow
        let metadata = Map.empty
        
        // Act
        let generation = {
            Id = id
            TemplateId = templateId
            Parameters = parameters
            GeneratedCode = generatedCode
            GenerationTime = generationTime
            Metadata = metadata
        }
        
        // Assert
        Assert.Equal(id, generation.Id)
        Assert.Equal(templateId, generation.TemplateId)
        Assert.Equal(parameters, generation.Parameters)
        Assert.Equal(generatedCode, generation.GeneratedCode)
        Assert.Equal(generationTime, generation.GenerationTime)
        Assert.Equal(metadata, generation.Metadata)
