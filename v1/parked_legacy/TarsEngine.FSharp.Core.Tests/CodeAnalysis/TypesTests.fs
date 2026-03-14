module TarsEngine.FSharp.Core.Tests.CodeAnalysis.TypesTests

open System
open Xunit
open TarsEngine.FSharp.Core.CodeAnalysis

/// <summary>
/// Tests for the CodeAnalysis.Types module.
/// </summary>
type TypesTests() =
    /// <summary>
    /// Test that a CodeFile can be created with valid values.
    /// </summary>
    [<Fact>]
    member _.``CodeFile can be created with valid values``() =
        // Arrange
        let path = "C:/test/file.cs"
        let content = "public class Test {}"
        let language = Language.CSharp
        let lastModified = DateTime.UtcNow
        let metadata = Map.empty
        
        // Act
        let codeFile = {
            Path = path
            Content = content
            Language = language
            LastModified = lastModified
            Metadata = metadata
        }
        
        // Assert
        Assert.Equal(path, codeFile.Path)
        Assert.Equal(content, codeFile.Content)
        Assert.Equal(language, codeFile.Language)
        Assert.Equal(lastModified, codeFile.LastModified)
        Assert.Equal(metadata, codeFile.Metadata)
    
    /// <summary>
    /// Test that a CodeIssue can be created with valid values.
    /// </summary>
    [<Fact>]
    member _.``CodeIssue can be created with valid values``() =
        // Arrange
        let id = Guid.NewGuid()
        let filePath = "C:/test/file.cs"
        let issueType = IssueType.Performance
        let severity = IssueSeverity.Warning
        let message = "This code could be optimized"
        let lineNumber = 42
        let columnNumber = 10
        let code = "slow_code"
        let metadata = Map.empty
        
        // Act
        let codeIssue = {
            Id = id
            FilePath = filePath
            Type = issueType
            Severity = severity
            Message = message
            LineNumber = lineNumber
            ColumnNumber = columnNumber
            Code = code
            Metadata = metadata
        }
        
        // Assert
        Assert.Equal(id, codeIssue.Id)
        Assert.Equal(filePath, codeIssue.FilePath)
        Assert.Equal(issueType, codeIssue.Type)
        Assert.Equal(severity, codeIssue.Severity)
        Assert.Equal(message, codeIssue.Message)
        Assert.Equal(lineNumber, codeIssue.LineNumber)
        Assert.Equal(columnNumber, codeIssue.ColumnNumber)
        Assert.Equal(code, codeIssue.Code)
        Assert.Equal(metadata, codeIssue.Metadata)
