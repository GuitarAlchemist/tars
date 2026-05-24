namespace TarsEngine.Tests.SelfImprovement

open System
open System.Collections.Generic
open System.IO
open System.Text.Json
open Xunit
open TarsEngine.SelfImprovement
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Logging.Abstractions

module RetroactionLoopTests =
    
    [<Fact>]
    let ``Create pattern should return valid pattern`` () =
        // Arrange
        let name = "Test Pattern"
        let description = "A test pattern"
        let pattern = "for\\s*\\(int\\s+i"
        let replacement = "foreach (var i in"
        let context = "CSharp"
        
        // Act
        let result = RetroactionLoop.createPattern name description pattern replacement context
        
        // Assert
        Assert.Equal(name, result.Name)
        Assert.Equal(description, result.Description)
        Assert.Equal(pattern, result.Pattern)
        Assert.Equal(replacement, result.Replacement)
        Assert.Equal(context, result.Context)
        Assert.Equal(0.5, result.Score) // Default score
        Assert.Equal(0, result.SuccessCount)
        Assert.Equal(0, result.FailureCount)
        Assert.Equal(0, result.UsageCount)
        Assert.Empty(result.Tags)
        Assert.NotNull(result.Metadata)
    
    [<Fact>]
    let ``Add pattern should update state`` () =
        // Arrange
        let state = RetroactionLoop.createState()
        let pattern = RetroactionLoop.createPattern "Test" "Test" "Test" "Test" "Test"
        
        // Act
        let result = RetroactionLoop.addPattern state pattern
        
        // Assert
        Assert.Single(result.Patterns)
        Assert.Equal(pattern, result.Patterns.[0])
        Assert.True(result.LastUpdated >= state.LastUpdated)
    
    [<Fact>]
    let ``Apply pattern should replace matching code`` () =
        // Arrange
        let pattern = RetroactionLoop.createPattern 
                        "Replace for loop with foreach" 
                        "Replace for loops with foreach loops" 
                        "for\\s*\\(int\\s+i\\s*=\\s*0;\\s*i\\s*<\\s*(\\w+)\\.Length;\\s*i\\+\\+\\)" 
                        "foreach (var item in $1)" 
                        "CSharp"
        
        let code = "for (int i = 0; i < array.Length; i++) { Console.WriteLine(array[i]); }"
        
        // Act
        let (success, result) = RetroactionLoop.applyPattern pattern code
        
        // Assert
        Assert.True(success)
        Assert.Equal("foreach (var item in array) { Console.WriteLine(array[i]); }", result)
    
    [<Fact>]
    let ``Apply patterns should apply multiple patterns`` () =
        // Arrange
        let pattern1 = RetroactionLoop.createPattern 
                        "Replace for loop with foreach" 
                        "Replace for loops with foreach loops" 
                        "for\\s*\\(int\\s+i\\s*=\\s*0;\\s*i\\s*<\\s*(\\w+)\\.Length;\\s*i\\+\\+\\)" 
                        "foreach (var item in $1)" 
                        "CSharp"
        
        let pattern2 = RetroactionLoop.createPattern 
                        "Replace array indexing with item" 
                        "Replace array indexing with item variable in foreach loops" 
                        "(foreach\\s*\\(var\\s+item\\s+in\\s+\\w+\\))\\s*\\{\\s*Console\\.WriteLine\\((\\w+)\\[i\\]\\);\\s*\\}" 
                        "$1 { Console.WriteLine(item); }" 
                        "CSharp"
        
        let state = RetroactionLoop.createState()
        let state = RetroactionLoop.addPattern state pattern1
        let state = RetroactionLoop.addPattern state pattern2
        
        // Update scores to ensure patterns are applied
        let updatedPatterns =
            state.Patterns
            |> List.map (fun p -> { p with Score = 0.8 })
        
        let state = { state with Patterns = updatedPatterns }
        
        let code = "for (int i = 0; i < array.Length; i++) { Console.WriteLine(array[i]); }"
        
        // Act
        let (result, _) = RetroactionLoop.applyPatterns state "CSharp" code
        
        // Assert
        Assert.Equal("foreach (var item in array) { Console.WriteLine(item); }", result)
    
    [<Fact>]
    let ``Record event should update pattern score`` () =
        // Arrange
        let pattern = RetroactionLoop.createPattern "Test" "Test" "Test" "Test" "Test"
        let state = RetroactionLoop.createState()
        let state = RetroactionLoop.addPattern state pattern
        
        let event = RetroactionLoop.createEvent pattern.Id "Feedback" 0.5 "Test"
        
        // Act
        let result = RetroactionLoop.recordEvent state event
        
        // Assert
        Assert.Single(result.Events)
        Assert.Equal(event, result.Events.[0])
        
        // Check that the pattern score was updated
        let updatedPattern = result.Patterns |> List.find (fun p -> p.Id = pattern.Id)
        Assert.Equal(0.55, updatedPattern.Score) // 0.5 + (0.5 * 0.1)
    
    [<Fact>]
    let ``Record pattern application should update pattern statistics`` () =
        // Arrange
        let pattern = RetroactionLoop.createPattern "Test" "Test" "Test" "Test" "Test"
        let state = RetroactionLoop.createState()
        let state = RetroactionLoop.addPattern state pattern
        
        let result = 
            { PatternId = pattern.Id
              Success = true
              Context = "Test"
              BeforeCode = "Before"
              AfterCode = "After"
              Timestamp = DateTime.UtcNow
              Metrics = new Dictionary<string, float>() }
        
        // Act
        let updatedState = RetroactionLoop.recordPatternApplication state result
        
        // Assert
        Assert.Single(updatedState.ApplicationResults)
        Assert.Equal(result, updatedState.ApplicationResults.[0])
        
        // Check that the pattern statistics were updated
        let updatedPattern = updatedState.Patterns |> List.find (fun p -> p.Id = pattern.Id)
        Assert.Equal(1, updatedPattern.UsageCount)
        Assert.Equal(1, updatedPattern.SuccessCount)
        Assert.Equal(0, updatedPattern.FailureCount)
        Assert.True(updatedPattern.LastUsed.IsSome)
    
    [<Fact>]
    let ``Analyze pattern performance should adjust scores based on success rate`` () =
        // Arrange
        let pattern = RetroactionLoop.createPattern "Test" "Test" "Test" "Test" "Test"
        let state = RetroactionLoop.createState()
        let state = RetroactionLoop.addPattern state pattern
        
        // Add some application results
        let successResult = 
            { PatternId = pattern.Id
              Success = true
              Context = "Test"
              BeforeCode = "Before"
              AfterCode = "After"
              Timestamp = DateTime.UtcNow
              Metrics = new Dictionary<string, float>() }
        
        let failureResult = 
            { PatternId = pattern.Id
              Success = false
              Context = "Test"
              BeforeCode = "Before"
              AfterCode = "Before" // No change
              Timestamp = DateTime.UtcNow
              Metrics = new Dictionary<string, float>() }
        
        let state = RetroactionLoop.recordPatternApplication state successResult
        let state = RetroactionLoop.recordPatternApplication state successResult
        let state = RetroactionLoop.recordPatternApplication state failureResult
        
        // Act
        let result = RetroactionLoop.analyzePatternPerformance state
        
        // Assert
        let updatedPattern = result.Patterns |> List.find (fun p -> p.Id = pattern.Id)
        
        // Score should be adjusted based on success rate (2/3 = 0.67)
        // New score = (old score * 0.7) + (success rate * 0.3)
        // = (0.5 * 0.7) + (0.67 * 0.3) = 0.35 + 0.2 = 0.55
        Assert.Equal(0.55, Math.Round(updatedPattern.Score, 2))
    
    [<Fact>]
    let ``Get top patterns should return patterns sorted by score`` () =
        // Arrange
        let pattern1 = { RetroactionLoop.createPattern "Test1" "Test1" "Test1" "Test1" "Test" with Score = 0.8 }
        let pattern2 = { RetroactionLoop.createPattern "Test2" "Test2" "Test2" "Test2" "Test" with Score = 0.6 }
        let pattern3 = { RetroactionLoop.createPattern "Test3" "Test3" "Test3" "Test3" "Test" with Score = 0.9 }
        
        let state = RetroactionLoop.createState()
        let state = RetroactionLoop.addPattern state pattern1
        let state = RetroactionLoop.addPattern state pattern2
        let state = RetroactionLoop.addPattern state pattern3
        
        // Act
        let result = RetroactionLoop.getTopPatterns state "Test" 2
        
        // Assert
        Assert.Equal(2, result.Length)
        Assert.Equal(pattern3.Id, result.[0].Id) // Highest score first
        Assert.Equal(pattern1.Id, result.[1].Id) // Second highest score
    
    [<Fact>]
    let ``Get matching patterns should return patterns that match the code`` () =
        // Arrange
        let pattern1 = RetroactionLoop.createPattern 
                        "Replace for loop with foreach" 
                        "Replace for loops with foreach loops" 
                        "for\\s*\\(int\\s+i" 
                        "foreach (var item in" 
                        "CSharp"
        
        let pattern2 = RetroactionLoop.createPattern 
                        "Replace while loop with for" 
                        "Replace while loops with for loops" 
                        "while\\s*\\(" 
                        "for (" 
                        "CSharp"
        
        let state = RetroactionLoop.createState()
        let state = RetroactionLoop.addPattern state pattern1
        let state = RetroactionLoop.addPattern state pattern2
        
        let code = "for (int i = 0; i < array.Length; i++) { Console.WriteLine(array[i]); }"
        
        // Act
        let result = RetroactionLoop.getMatchingPatterns state "CSharp" code
        
        // Assert
        Assert.Single(result)
        Assert.Equal(pattern1.Id, result.[0].Id)
    
    [<Fact>]
    let ``Get statistics should return correct statistics`` () =
        // Arrange
        let pattern1 = { RetroactionLoop.createPattern "Test1" "Test1" "Test1" "Test1" "CSharp" with Score = 0.8 }
        let pattern2 = { RetroactionLoop.createPattern "Test2" "Test2" "Test2" "Test2" "FSharp" with Score = 0.4 }
        
        let state = RetroactionLoop.createState()
        let state = RetroactionLoop.addPattern state pattern1
        let state = RetroactionLoop.addPattern state pattern2
        
        let result1 = 
            { PatternId = pattern1.Id
              Success = true
              Context = "CSharp"
              BeforeCode = "Before"
              AfterCode = "After"
              Timestamp = DateTime.UtcNow
              Metrics = new Dictionary<string, float>() }
        
        let result2 = 
            { PatternId = pattern2.Id
              Success = false
              Context = "FSharp"
              BeforeCode = "Before"
              AfterCode = "Before"
              Timestamp = DateTime.UtcNow
              Metrics = new Dictionary<string, float>() }
        
        let state = RetroactionLoop.recordPatternApplication state result1
        let state = RetroactionLoop.recordPatternApplication state result2
        
        // Act
        let stats = RetroactionLoop.getStatistics state
        
        // Assert
        Assert.Equal(2, stats.TotalPatterns)
        Assert.Equal(1, stats.ActivePatterns) // Only pattern1 has score >= 0.6
        Assert.Equal(2, stats.TotalApplications)
        Assert.Equal(1, stats.SuccessfulApplications)
        Assert.Equal(0.5, stats.SuccessRate)
        Assert.Equal(0.6, stats.AveragePatternScore)
        
        // Check context counts
        Assert.Equal(1, stats.PatternsByContext.["CSharp"])
        Assert.Equal(1, stats.PatternsByContext.["FSharp"])
