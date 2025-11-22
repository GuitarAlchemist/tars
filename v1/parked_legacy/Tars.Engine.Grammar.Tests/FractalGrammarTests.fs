namespace Tars.Engine.Grammar.Tests

open System
open Xunit
open Tars.Engine.Grammar.FractalGrammar
open Tars.Engine.Grammar.FractalGrammarParser
open Tars.Engine.Grammar.FractalGrammarIntegration

/// Comprehensive tests for TARS Fractal Grammar System
module FractalGrammarTests =

    [<Fact>]
    let ``FractalGrammarEngine should create default properties`` () =
        // Arrange
        let engine = FractalGrammarEngine()
        
        // Act
        let properties = engine.CreateDefaultProperties()
        
        // Assert
        Assert.True(properties.Dimension > 0.0)
        Assert.True(properties.ScalingFactor > 0.0)
        Assert.True(properties.IterationDepth > 0)
        Assert.True(properties.RecursionLimit > 0)
        Assert.NotEmpty(properties.CompositionRules)

    [<Fact>]
    let ``FractalGrammarEngine should apply scale transformation`` () =
        // Arrange
        let engine = FractalGrammarEngine()
        let pattern = "test pattern"
        let transformation = Scale 2.0
        
        // Act
        let result = engine.ApplyTransformation(pattern, transformation)
        
        // Assert
        Assert.NotEqual(pattern, result)
        Assert.True(result.Length >= pattern.Length)

    [<Fact>]
    let ``FractalGrammarEngine should apply rotation transformation`` () =
        // Arrange
        let engine = FractalGrammarEngine()
        let pattern = "word1 word2 word3"
        let transformation = Rotate 90.0
        
        // Act
        let result = engine.ApplyTransformation(pattern, transformation)
        
        // Assert
        Assert.NotEqual(pattern, result)
        Assert.Contains("word", result)

    [<Fact>]
    let ``FractalGrammarEngine should apply recursive transformation`` () =
        // Arrange
        let engine = FractalGrammarEngine()
        let pattern = "base"
        let transformation = Recursive (3, Scale 1.1)
        
        // Act
        let result = engine.ApplyTransformation(pattern, transformation)
        
        // Assert
        Assert.NotEqual(pattern, result)

    [<Fact>]
    let ``FractalGrammarEngine should apply composition transformation`` () =
        // Arrange
        let engine = FractalGrammarEngine()
        let pattern = "test"
        let transformation = Compose [Scale 1.5; Rotate 45.0]
        
        // Act
        let result = engine.ApplyTransformation(pattern, transformation)
        
        // Assert
        Assert.NotEqual(pattern, result)

    [<Fact>]
    let ``FractalGrammarEngine should apply conditional transformation`` () =
        // Arrange
        let engine = FractalGrammarEngine()
        let shortPattern = "short"
        let longPattern = "this is a very long pattern with many words"
        let transformation = Conditional ("length_gt_10", Scale 2.0, Scale 0.5)
        
        // Act
        let shortResult = engine.ApplyTransformation(shortPattern, transformation)
        let longResult = engine.ApplyTransformation(longPattern, transformation)
        
        // Assert
        Assert.NotEqual(shortPattern, shortResult)
        Assert.NotEqual(longPattern, longResult)
        Assert.True(shortResult.Length < longResult.Length)

    [<Fact>]
    let ``FractalGrammarEngine should generate fractal node`` () =
        // Arrange
        let engine = FractalGrammarEngine()
        let rule = {
            Name = "test_rule"
            BasePattern = "pattern"
            RecursivePattern = Some "pattern pattern"
            TerminationCondition = "max_depth_5"
            Transformations = [Scale 0.8]
            Properties = engine.CreateDefaultProperties()
            Dependencies = []
        }
        
        // Act
        let node = engine.GenerateFractalNode(rule, 0, None)
        
        // Assert
        Assert.Equal("test_rule", node.Name)
        Assert.Equal(0, node.Level)
        Assert.True(node.ParentId.IsNone)
        Assert.NotEmpty(node.Pattern)

    [<Fact>]
    let ``FractalGrammarEngine should generate complete fractal grammar`` () =
        // Arrange
        let engine = FractalGrammarEngine()
        let rule = {
            Name = "sierpinski"
            BasePattern = "triangle"
            RecursivePattern = Some "triangle triangle triangle"
            TerminationCondition = "max_depth_5"
            Transformations = [Scale 0.5]
            Properties = { engine.CreateDefaultProperties() with RecursionLimit = 3 }
            Dependencies = []
        }
        
        let fractalGrammar = 
            FractalGrammarBuilder()
                .WithName("TestFractal")
                .WithVersion("1.0")
                .AddFractalRule(rule)
                .Build()
        
        // Act
        let result = engine.GenerateFractalGrammar(fractalGrammar)
        
        // Assert
        Assert.True(result.Success)
        Assert.NotEmpty(result.GeneratedGrammar)
        Assert.True(result.IterationsPerformed > 0)
        Assert.True(result.ComputationTime.TotalMilliseconds >= 0.0)

    [<Fact>]
    let ``FractalGrammarEngine should analyze complexity`` () =
        // Arrange
        let engine = FractalGrammarEngine()
        let rule = {
            Name = "test"
            BasePattern = "pattern"
            RecursivePattern = None
            TerminationCondition = "max_depth_5"
            Transformations = []
            Properties = engine.CreateDefaultProperties()
            Dependencies = []
        }
        
        let fractalGrammar = 
            FractalGrammarBuilder()
                .WithName("TestComplexity")
                .AddFractalRule(rule)
                .Build()
        
        // Act
        let complexity = engine.AnalyzeFractalComplexity(fractalGrammar)
        
        // Assert
        Assert.True(complexity.ContainsKey("total_rules"))
        Assert.True(complexity.ContainsKey("average_dimension"))
        Assert.True(complexity.ContainsKey("max_recursion_depth"))

    [<Fact>]
    let ``FractalGrammarBuilder should build valid grammar`` () =
        // Arrange
        let builder = FractalGrammarBuilder()
        let rule = {
            Name = "test_rule"
            BasePattern = "test"
            RecursivePattern = None
            TerminationCondition = "always_terminate"
            Transformations = []
            Properties = FractalGrammarEngine().CreateDefaultProperties()
            Dependencies = []
        }
        
        // Act
        let grammar = 
            builder
                .WithName("TestGrammar")
                .WithVersion("2.0")
                .AddFractalRule(rule)
                .Build()
        
        // Assert
        Assert.Equal("TestGrammar", grammar.Name)
        Assert.Equal("2.0", grammar.Version)
        Assert.Single(grammar.FractalRules)
        Assert.Equal("test_rule", grammar.FractalRules.Head.Name)

    [<Fact>]
    let ``FractalGrammarService should create simple fractal rule`` () =
        // Arrange
        let service = FractalGrammarService()
        
        // Act
        let rule = service.CreateSimpleFractalRule("simple", "base_pattern")
        
        // Assert
        Assert.Equal("simple", rule.Name)
        Assert.Equal("base_pattern", rule.BasePattern)
        Assert.True(rule.RecursivePattern.IsSome)
        Assert.NotEmpty(rule.Transformations)

    [<Fact>]
    let ``FractalGrammarService should generate from specification`` () =
        // Arrange
        let service = FractalGrammarService()
        let patterns = ["pattern1"; "pattern2"; "pattern3"]
        
        // Act
        let result = service.GenerateFromSpec("TestSpec", patterns)
        
        // Assert
        Assert.True(result.Success)
        Assert.NotEmpty(result.GeneratedGrammar)

    [<Fact>]
    let ``FractalGrammarParser should parse simple fractal grammar`` () =
        // Arrange
        let parser = FractalGrammarParser()
        let input = """
            fractal test_rule {
                pattern = "base pattern"
                dimension = 1.5
                depth = 5
                transform scale 0.8
            }
        """
        
        // Act
        let result = parser.ParseFractalGrammar(input)
        
        // Assert
        Assert.True(result.Success)
        Assert.Single(result.ParsedRules)
        Assert.Equal("test_rule", result.ParsedRules.Head.Name)
        Assert.Equal("base pattern", result.ParsedRules.Head.BasePattern)

    [<Fact>]
    let ``FractalGrammarParser should handle parse errors`` () =
        // Arrange
        let parser = FractalGrammarParser()
        let invalidInput = "invalid fractal syntax"
        
        // Act
        let result = parser.ParseFractalGrammar(invalidInput)
        
        // Assert
        Assert.False(result.Success)
        Assert.True(result.FractalGrammar.IsNone)

    [<Fact>]
    let ``FractalGrammarManager should execute fractal grammar`` () =
        // Arrange
        let manager = FractalGrammarManager()
        let rule = {
            Name = "execution_test"
            BasePattern = "test pattern"
            RecursivePattern = None
            TerminationCondition = "always_terminate"
            Transformations = [Scale 1.0]
            Properties = FractalGrammarEngine().CreateDefaultProperties()
            Dependencies = []
        }
        
        let fractalGrammar = 
            FractalGrammarBuilder()
                .WithName("ExecutionTest")
                .AddFractalRule(rule)
                .Build()
        
        let context = manager.CreateDefaultContext()
        
        // Act
        let result = manager.ExecuteFractalGrammar(fractalGrammar, context)
        
        // Assert
        Assert.True(result.Success)
        Assert.NotEmpty(result.GeneratedGrammar)
        Assert.True(result.ExecutionTime.TotalMilliseconds >= 0.0)
        Assert.True(result.FractalDimension >= 0.0)

    [<Fact>]
    let ``FractalGrammarManager should format grammar to different outputs`` () =
        // Arrange
        let manager = FractalGrammarManager()
        let rule = {
            Name = "format_test"
            BasePattern = "test = pattern"
            RecursivePattern = None
            TerminationCondition = "always_terminate"
            Transformations = []
            Properties = FractalGrammarEngine().CreateDefaultProperties()
            Dependencies = []
        }
        
        let fractalGrammar = 
            FractalGrammarBuilder()
                .WithName("FormatTest")
                .AddFractalRule(rule)
                .Build()
        
        // Act & Assert
        let formats = [EBNF; ANTLR; JSON; XML; GraphViz; SVG]
        for format in formats do
            let context = { manager.CreateDefaultContext() with OutputFormat = format }
            let result = manager.ExecuteFractalGrammar(fractalGrammar, context)
            Assert.True(result.Success, sprintf "Format %A should succeed" format)
            Assert.NotEmpty(result.GeneratedGrammar)

    [<Fact>]
    let ``FractalGrammarManager should generate visualization`` () =
        // Arrange
        let manager = FractalGrammarManager()
        let rule = {
            Name = "viz_test"
            BasePattern = "visual pattern"
            RecursivePattern = None
            TerminationCondition = "always_terminate"
            Transformations = []
            Properties = FractalGrammarEngine().CreateDefaultProperties()
            Dependencies = []
        }
        
        let fractalGrammar = 
            FractalGrammarBuilder()
                .WithName("VizTest")
                .AddFractalRule(rule)
                .Build()
        
        let context = { manager.CreateDefaultContext() with EnableVisualization = true; OutputFormat = SVG }
        
        // Act
        let result = manager.ExecuteFractalGrammar(fractalGrammar, context)
        
        // Assert
        Assert.True(result.Success)
        Assert.True(result.VisualizationData.IsSome)
        Assert.Contains("svg", result.VisualizationData.Value.ToLowerInvariant())

    [<Theory>]
    [<InlineData(1.0, 1.0)>]
    [<InlineData(2.0, 2.0)>]
    [<InlineData(0.5, 0.5)>]
    let ``FractalGrammarEngine should handle different scaling factors`` (scaleFactor: float, expectedRatio: float) =
        // Arrange
        let engine = FractalGrammarEngine()
        let pattern = "test pattern"
        let transformation = Scale scaleFactor
        
        // Act
        let result = engine.ApplyTransformation(pattern, transformation)
        
        // Assert
        Assert.NotNull(result)
        if scaleFactor > 1.0 then
            Assert.True(result.Length >= pattern.Length)
        elif scaleFactor < 1.0 then
            Assert.True(result.Length <= pattern.Length)

    [<Fact>]
    let ``Sierpinski Triangle example should generate valid fractal`` () =
        // Arrange
        let sierpinski = FractalGrammarParser.Examples.sierpinskiTriangle
        let engine = FractalGrammarEngine()
        
        // Act
        let result = engine.GenerateFractalGrammar(sierpinski)
        
        // Assert
        Assert.True(result.Success)
        Assert.Contains("triangle", result.GeneratedGrammar.ToLowerInvariant())
        Assert.True(result.IterationsPerformed > 0)

    [<Fact>]
    let ``Koch Snowflake example should generate valid fractal`` () =
        // Arrange
        let koch = FractalGrammarParser.Examples.kochSnowflake
        let engine = FractalGrammarEngine()
        
        // Act
        let result = engine.GenerateFractalGrammar(koch)
        
        // Assert
        Assert.True(result.Success)
        Assert.Contains("line", result.GeneratedGrammar.ToLowerInvariant())
        Assert.True(result.IterationsPerformed > 0)

    [<Fact>]
    let ``FractalDSL should create grammar using computation expression`` () =
        // Arrange & Act
        let grammar = FractalGrammarParser.FractalDSL.fractalGrammar {
            FractalGrammarParser.FractalDSL.fractalRule "test1" "pattern1"
            FractalGrammarParser.FractalDSL.fractalRule "test2" "pattern2"
        }
        
        // Assert
        Assert.Equal(2, grammar.FractalRules.Length)
        Assert.Equal("test1", grammar.FractalRules.[0].Name)
        Assert.Equal("test2", grammar.FractalRules.[1].Name)
