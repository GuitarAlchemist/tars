module TarsEngine.AutoImprovement.Tests.TieredGrammarTests

open System
open System.Collections.Generic
open Xunit
open FsUnit.Xunit
open FsCheck.Xunit

// === TIERED GRAMMARS & FRACTAL DESIGN TESTS ===

type GrammarRule = {
    RuleId: string
    LeftSide: string
    RightSide: string list
    Weight: float
    EvolutionGeneration: int
}

type GrammarTier = {
    TierLevel: int
    Rules: Map<string, GrammarRule list>
    Complexity: float
    EvolutionGeneration: int
    FractalDepth: int
    ParentTier: int option
    ChildTiers: int list
}

type FractalGrammarSystem() =
    let tiers = Dictionary<int, GrammarTier>()
    let maxTiers = 16
    let mutable currentGeneration = 1
    
    member _.CreateTier(level: int, parentTier: int option) =
        if level > maxTiers then failwith $"Maximum {maxTiers} tiers allowed"
        
        let baseRules = [
            { RuleId = Guid.NewGuid().ToString("N").[..7]; LeftSide = "IMPROVEMENT"; RightSide = ["ANALYZE"; "OPTIMIZE"; "VALIDATE"]; Weight = 1.0; EvolutionGeneration = 1 }
            { RuleId = Guid.NewGuid().ToString("N").[..7]; LeftSide = "ANALYZE"; RightSide = ["SCAN_CODE"; "IDENTIFY_PATTERNS"]; Weight = 0.8; EvolutionGeneration = 1 }
            { RuleId = Guid.NewGuid().ToString("N").[..7]; LeftSide = "OPTIMIZE"; RightSide = ["REFACTOR"; "PARALLELIZE"]; Weight = 0.9; EvolutionGeneration = 1 }
        ]
        
        let rulesMap = baseRules |> List.groupBy (fun r -> r.LeftSide) |> Map.ofList
        let complexity = float level * 0.1 + (float baseRules.Length * 0.05)
        let fractalDepth = level
        
        let tier = {
            TierLevel = level
            Rules = rulesMap
            Complexity = complexity
            EvolutionGeneration = currentGeneration
            FractalDepth = fractalDepth
            ParentTier = parentTier
            ChildTiers = []
        }
        
        tiers.[level] <- tier
        tier
    
    member _.EvolveTier(tierLevel: int) =
        match tiers.TryGetValue(tierLevel) with
        | true, tier ->
            let evolvedRules = 
                tier.Rules 
                |> Map.map (fun leftSide rules ->
                    rules |> List.map (fun rule ->
                        let newRightSide = rule.RightSide @ [sprintf "EVOLVED_%s_%d" leftSide currentGeneration]
                        { rule with 
                            RightSide = newRightSide
                            Weight = rule.Weight * 1.1
                            EvolutionGeneration = currentGeneration + 1 }))
            
            let evolvedTier = {
                tier with
                    Rules = evolvedRules
                    Complexity = tier.Complexity * 1.2
                    EvolutionGeneration = currentGeneration + 1
            }
            
            tiers.[tierLevel] <- evolvedTier
            currentGeneration <- currentGeneration + 1
            evolvedTier
        | false, _ -> failwith $"Tier {tierLevel} not found"
    
    member _.CreateFractalHierarchy() =
        // Create 16-tier fractal hierarchy
        let mutable createdTiers = []
        
        // Level 1: Root tier
        let rootTier = _.CreateTier(1, None)
        createdTiers <- rootTier :: createdTiers
        
        // Levels 2-4: Primary branches
        for level in 2..4 do
            let tier = _.CreateTier(level, Some 1)
            createdTiers <- tier :: createdTiers
        
        // Levels 5-8: Secondary branches
        for level in 5..8 do
            let parentLevel = ((level - 5) % 4) + 2
            let tier = _.CreateTier(level, Some parentLevel)
            createdTiers <- tier :: createdTiers
        
        // Levels 9-16: Tertiary branches
        for level in 9..16 do
            let parentLevel = ((level - 9) % 8) + 5
            let tier = _.CreateTier(level, Some parentLevel)
            createdTiers <- tier :: createdTiers
        
        createdTiers |> List.rev
    
    member _.GenerateFromGrammar(tierLevel: int, startSymbol: string, maxDepth: int) =
        match tiers.TryGetValue(tierLevel) with
        | true, tier ->
            let rec generate symbol depth =
                if depth >= maxDepth then [symbol]
                else
                    match tier.Rules.TryFind(symbol) with
                    | Some rules ->
                        let rule = rules |> List.maxBy (fun r -> r.Weight)
                        rule.RightSide |> List.collect (fun s -> generate s (depth + 1))
                    | None -> [symbol]
            
            generate startSymbol 0
        | false, _ -> [startSymbol]
    
    member _.GetTierCount() = tiers.Count
    member _.GetTier(level: int) = 
        match tiers.TryGetValue(level) with
        | true, tier -> Some tier
        | false, _ -> None
    
    member _.GetCurrentGeneration() = currentGeneration

[<Fact>]
let ``Fractal Grammar System should create tiered hierarchy`` () =
    // Arrange
    let system = FractalGrammarSystem()
    
    // Act
    let tiers = system.CreateFractalHierarchy()
    
    // Assert
    tiers.Length |> should equal 16
    system.GetTierCount() |> should equal 16
    
    let rootTier = system.GetTier(1).Value
    rootTier.TierLevel |> should equal 1
    rootTier.ParentTier |> should equal None

[<Fact>]
let ``Fractal Grammar should support 16 maximum tiers`` () =
    // Arrange
    let system = FractalGrammarSystem()
    
    // Act & Assert
    for level in 1..16 do
        let tier = system.CreateTier(level, None)
        tier.TierLevel |> should equal level
    
    // Should fail on 17th tier
    (fun () -> system.CreateTier(17, None) |> ignore) 
    |> should throw typeof<System.Exception>

[<Fact>]
let ``Grammar tiers should evolve with increasing complexity`` () =
    // Arrange
    let system = FractalGrammarSystem()
    let tier = system.CreateTier(1, None)
    let initialComplexity = tier.Complexity
    let initialGeneration = tier.EvolutionGeneration
    
    // Act
    let evolvedTier = system.EvolveTier(1)
    
    // Assert
    evolvedTier.Complexity |> should be (greaterThan initialComplexity)
    evolvedTier.EvolutionGeneration |> should be (greaterThan initialGeneration)
    
    // Rules should have evolved
    let originalRuleCount = tier.Rules |> Map.fold (fun acc _ rules -> acc + rules.Length) 0
    let evolvedRuleCount = evolvedTier.Rules |> Map.fold (fun acc _ rules -> acc + (rules |> List.sumBy (fun r -> r.RightSide.Length))) 0
    evolvedRuleCount |> should be (greaterThan originalRuleCount)

[<Fact>]
let ``Grammar should generate valid derivations`` () =
    // Arrange
    let system = FractalGrammarSystem()
    let tier = system.CreateTier(1, None)
    
    // Act
    let derivation = system.GenerateFromGrammar(1, "IMPROVEMENT", 3)
    
    // Assert
    derivation |> should not' (be Empty)
    derivation |> should contain "ANALYZE"
    derivation |> should contain "OPTIMIZE"
    derivation |> should contain "VALIDATE"

[<Property>]
let ``All grammar tiers should have valid parent-child relationships`` (levels: int list) =
    let system = FractalGrammarSystem()
    let validLevels = levels |> List.filter (fun l -> l >= 1 && l <= 16) |> List.distinct
    
    if validLevels.Length > 0 then
        for level in validLevels do
            let parentLevel = if level > 1 then Some (level - 1) else None
            let tier = system.CreateTier(level, parentLevel)
            
            match tier.ParentTier with
            | Some parent -> parent < tier.TierLevel
            | None -> tier.TierLevel = 1
    else
        true

[<Fact>]
let ``Fractal depth should correspond to tier level`` () =
    // Arrange
    let system = FractalGrammarSystem()
    
    // Act
    let tiers = [1..8] |> List.map (fun level -> system.CreateTier(level, None))
    
    // Assert
    for tier in tiers do
        tier.FractalDepth |> should equal tier.TierLevel

[<Fact>]
let ``Grammar evolution should preserve rule structure`` () =
    // Arrange
    let system = FractalGrammarSystem()
    let tier = system.CreateTier(1, None)
    let originalRuleKeys = tier.Rules |> Map.keys |> Set.ofSeq
    
    // Act
    let evolvedTier = system.EvolveTier(1)
    let evolvedRuleKeys = evolvedTier.Rules |> Map.keys |> Set.ofSeq
    
    // Assert
    evolvedRuleKeys |> should equal originalRuleKeys
    
    // Each rule should have more right-hand side options
    for (key, originalRules) in tier.Rules |> Map.toList do
        let evolvedRules = evolvedTier.Rules.[key]
        for (original, evolved) in List.zip originalRules evolvedRules do
            evolved.RightSide.Length |> should be (greaterThan original.RightSide.Length)

[<Fact>]
let ``Grammar system should handle concurrent tier creation`` () =
    // Arrange
    let system = FractalGrammarSystem()
    
    // Act
    let tasks = [1..8] |> List.map (fun level ->
        System.Threading.Tasks.Task.Run(fun () -> system.CreateTier(level, None)))
    
    System.Threading.Tasks.Task.WaitAll(tasks.ToArray())
    
    // Assert
    system.GetTierCount() |> should equal 8
    for level in 1..8 do
        let tier = system.GetTier(level)
        tier.IsSome |> should equal true

[<Fact>]
let ``Fractal grammar should support factorization`` () =
    // Arrange
    let system = FractalGrammarSystem()
    let tier = system.CreateTier(1, None)
    
    // Act - Factorize by extracting common patterns
    let commonPatterns = 
        tier.Rules 
        |> Map.values 
        |> Seq.collect id
        |> Seq.collect (fun rule -> rule.RightSide)
        |> Seq.groupBy id
        |> Seq.filter (fun (_, occurrences) -> Seq.length occurrences > 1)
        |> Seq.map fst
        |> Seq.toList
    
    // Assert
    commonPatterns |> should not' (be Empty)

[<Fact>]
let ``Grammar system should support rule weight optimization`` () =
    // Arrange
    let system = FractalGrammarSystem()
    let tier = system.CreateTier(1, None)
    
    // Act
    let evolvedTier = system.EvolveTier(1)
    
    // Assert
    for (_, rules) in evolvedTier.Rules |> Map.toList do
        for rule in rules do
            rule.Weight |> should be (greaterThan 0.8)  // Weights should increase through evolution
