module GrammarDistillationTests

open System
open Xunit
open Tars.Core
open Tars.Core.TemporalKnowledgeGraph
open Tars.Core.GrammarDistillation

[<Fact>]
let ``Distills rule when sufficient patterns exist`` () =
    let graph = TemporalGraph()
    let distiller = GrammarDistiller(graph)
    
    // Create 3 structural patterns
    let p1 = CodePatternE { Name = "Singleton"; Category = Structural; Signature = "class S { static instance }"; Occurrences = 5; FirstSeen = DateTime.UtcNow; LastSeen = DateTime.UtcNow }
    let p2 = CodePatternE { Name = "Adapter"; Category = Structural; Signature = "class A : ITarget"; Occurrences = 3; FirstSeen = DateTime.UtcNow; LastSeen = DateTime.UtcNow }
    let p3 = CodePatternE { Name = "Facade"; Category = Structural; Signature = "class F { sub1, sub2 }"; Occurrences = 2; FirstSeen = DateTime.UtcNow; LastSeen = DateTime.UtcNow }
    
    let entities = [p1; p2; p3]
    
    let rules = distiller.DistillRules(entities)
    
    let rule = Assert.Single(rules)
    Assert.Equal("Standard Structural Pattern", rule.Name)
    Assert.Equal(3, rule.Examples.Length)
    Assert.Contains("Singleton", rule.Examples)

[<Fact>]
let ``Does not distill rule when insufficient patterns exist`` () =
    let graph = TemporalGraph()
    let distiller = GrammarDistiller(graph)
    
    // Create 2 behavioral patterns (heuristic requires > 2)
    let p1 = CodePatternE { Name = "Observer"; Category = Behavioral; Signature = "interface IObserver"; Occurrences = 5; FirstSeen = DateTime.UtcNow; LastSeen = DateTime.UtcNow }
    let p2 = CodePatternE { Name = "Strategy"; Category = Behavioral; Signature = "interface IStrategy"; Occurrences = 3; FirstSeen = DateTime.UtcNow; LastSeen = DateTime.UtcNow }
    
    let entities = [p1; p2]
    
    let rules = distiller.DistillRules(entities)
    
    Assert.Empty(rules)

[<Fact>]
let ``Distills multiple rules for different categories`` () =
    let graph = TemporalGraph()
    let distiller = GrammarDistiller(graph)
    
    // 3 Structural
    let s1 = CodePatternE { Name = "S1"; Category = Structural; Signature = ""; Occurrences = 1; FirstSeen = DateTime.UtcNow; LastSeen = DateTime.UtcNow }
    let s2 = CodePatternE { Name = "S2"; Category = Structural; Signature = ""; Occurrences = 1; FirstSeen = DateTime.UtcNow; LastSeen = DateTime.UtcNow }
    let s3 = CodePatternE { Name = "S3"; Category = Structural; Signature = ""; Occurrences = 1; FirstSeen = DateTime.UtcNow; LastSeen = DateTime.UtcNow }
    
    // 3 Creational
    let c1 = CodePatternE { Name = "C1"; Category = Creational; Signature = ""; Occurrences = 1; FirstSeen = DateTime.UtcNow; LastSeen = DateTime.UtcNow }
    let c2 = CodePatternE { Name = "C2"; Category = Creational; Signature = ""; Occurrences = 1; FirstSeen = DateTime.UtcNow; LastSeen = DateTime.UtcNow }
    let c3 = CodePatternE { Name = "C3"; Category = Creational; Signature = ""; Occurrences = 1; FirstSeen = DateTime.UtcNow; LastSeen = DateTime.UtcNow }
    
    let entities = [s1; s2; s3; c1; c2; c3]
    
    let rules = distiller.DistillRules(entities)
    
    Assert.Equal(2, rules.Length)
    Assert.True(rules |> List.exists (fun r -> r.Name = "Standard Structural Pattern"))
    Assert.True(rules |> List.exists (fun r -> r.Name = "Factory/Builder Pattern"))

[<Fact>]
let ``Generates DU from grammar rule`` () =
    let generator = CodeGenerator()
    let rule = {
        Name = "My Pattern"
        Production = ""
        Examples = ["CaseA"; "CaseB"]
        DistilledFrom = []
        Version = 1
    }
    
    let code = generator.GenerateDU(rule)
    
    Assert.Contains("type MyPattern =", code)
    Assert.Contains("| CaseA", code)
    Assert.Contains("| CaseB", code)

[<Fact>]
let ``Generates CE from grammar rule`` () =
    let generator = CodeGenerator()
    let rule = {
        Name = "My Pattern"
        Production = ""
        Examples = []
        DistilledFrom = []
        Version = 1
    }
    
    let code = generator.GenerateCE(rule)
    
    Assert.Contains("type MyPatternBuilder() =", code)
    Assert.Contains("let mypatternBuilder = MyPatternBuilder()", code)

[<Fact>]
let ``HotReloadManager detects and evolves new rules`` () =
    let graph = TemporalGraph()
    let distiller = GrammarDistiller(graph)
    let generator = CodeGenerator()
    let manager = HotReloadManager(distiller, generator)
    
    // Initially empty
    Assert.Empty(manager.Evolve())
    
    // Add patterns to graph (via graph manipulation or mocking)
    // Since we don't have easy graph injection here without a lot of setup,
    // we'll rely on the fact that DistillFromGraph calls GetNodes.
    // We'll manually inject entities into the graph.
    
    // 3 Structural patterns
    let p1 = CodePatternE { Name = "Singleton"; Category = Structural; Signature = ""; Occurrences = 5; FirstSeen = DateTime.UtcNow; LastSeen = DateTime.UtcNow }
    let p2 = CodePatternE { Name = "Adapter"; Category = Structural; Signature = ""; Occurrences = 3; FirstSeen = DateTime.UtcNow; LastSeen = DateTime.UtcNow }
    let p3 = CodePatternE { Name = "Facade"; Category = Structural; Signature = ""; Occurrences = 2; FirstSeen = DateTime.UtcNow; LastSeen = DateTime.UtcNow }
    
    graph.AddFact(DerivedFrom(p1, p1)) |> ignore // Hack to add nodes to graph
    graph.AddFact(DerivedFrom(p2, p2)) |> ignore
    graph.AddFact(DerivedFrom(p3, p3)) |> ignore
    
    let updates = manager.Evolve()
    
    let updatesList = Assert.Single(updates)
    let (name, code) = updatesList
    Assert.Equal("Standard Structural Pattern", name)
    Assert.Contains("type StandardStructuralPattern =", code)
    Assert.Contains("type StandardStructuralPatternBuilder() =", code)
    
    // Evolve again - should be empty as nothing changed
    Assert.Empty(manager.Evolve())
