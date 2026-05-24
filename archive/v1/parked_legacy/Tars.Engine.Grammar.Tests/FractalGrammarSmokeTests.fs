namespace Tars.Engine.Grammar.Tests

open Xunit
open Tars.Engine.Grammar.FractalGrammar
open Tars.Engine.Grammar.FractalGrammarIntegration

module FractalGrammarSmokeTests =

    [<Fact>]
    let ``CreateDefaultProperties yields positive defaults`` () =
        let properties = FractalGrammarEngine().CreateDefaultProperties()
        Assert.True(properties.Dimension > 0.0)
        Assert.True(properties.IterationDepth > 0)
        Assert.NotEmpty(properties.CompositionRules)

    [<Fact>]
    let ``Builder can compose simple fractal rule`` () =
        let engine = FractalGrammarEngine()
        let rule =
            { Name = "test"
              BasePattern = "A"
              RecursivePattern = Some "A A"
              TerminationCondition = "max_depth_3"
              Transformations = [Scale 1.2]
              Properties = engine.CreateDefaultProperties()
              Dependencies = [] }
        let grammar =
            FractalGrammarBuilder()
                .WithName("Smoke")
                .AddFractalRule(rule)
                .Build()

        Assert.Equal("Smoke", grammar.Name)
        Assert.Single(grammar.FractalRules)

    [<Fact>]
    let ``Manager executes fractal grammar`` () =
        let engine = FractalGrammarEngine()
        let rule =
            { Name = "koch"
              BasePattern = "line"
              RecursivePattern = Some "line line line"
              TerminationCondition = "max_depth_2"
              Transformations = [Scale 1.1]
              Properties = engine.CreateDefaultProperties()
              Dependencies = [] }

        let grammar =
            FractalGrammarBuilder()
                .WithName("Koch")
                .AddFractalRule(rule)
                .Build()

        let manager = FractalGrammarManager()
        let result = manager.ExecuteFractalGrammar(grammar, manager.CreateDefaultContext())
        Assert.True(result.Success)
        Assert.NotEmpty(result.GeneratedGrammar)
