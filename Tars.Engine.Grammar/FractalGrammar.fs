namespace Tars.Engine.Grammar

open System
open System.Collections.Generic
open Tars.Engine.Grammar.GrammarSource

/// Fractal Grammar System for TARS
/// Implements self-similar, recursive, and dynamically composable grammars
module FractalGrammar =

    /// Fractal dimension and scaling properties
    type FractalProperties = {
        Dimension: float
        ScalingFactor: float
        IterationDepth: int
        SelfSimilarityRatio: float
        RecursionLimit: int
        CompositionRules: string list
    }

    /// Fractal transformation types
    type FractalTransformation =
        | Scale of factor: float
        | Rotate of angle: float
        | Translate of x: float * y: float
        | Compose of transformations: FractalTransformation list
        | Recursive of depth: int * transformation: FractalTransformation
        | Conditional of predicate: string * ifTrue: FractalTransformation * ifFalse: FractalTransformation

    /// Fractal grammar node representing a self-similar structure
    type FractalNode = {
        Id: string
        Name: string
        Pattern: string
        Children: FractalNode list
        Properties: FractalProperties
        Transformations: FractalTransformation list
        Metadata: Map<string, obj>
        Level: int
        ParentId: string option
    }

    /// Fractal grammar rule with recursive expansion capabilities
    type FractalRule = {
        Name: string
        BasePattern: string
        RecursivePattern: string option
        TerminationCondition: string
        Transformations: FractalTransformation list
        Properties: FractalProperties
        Dependencies: string list
    }

    /// Complete fractal grammar definition
    type FractalGrammar = {
        Name: string
        Version: string
        BaseGrammar: Grammar
        FractalRules: FractalRule list
        GlobalProperties: FractalProperties
        CompositionGraph: Map<string, string list>
        GenerationHistory: FractalNode list
        Metadata: GrammarMetadata
    }

    /// Fractal grammar generation result
    type FractalGenerationResult = {
        Success: bool
        GeneratedGrammar: string
        FractalTree: FractalNode
        IterationsPerformed: int
        ComputationTime: TimeSpan
        MemoryUsage: int64
        ErrorMessages: string list
        Warnings: string list
    }

    /// Fractal grammar engine
    type FractalGrammarEngine() =
        
        /// Create default fractal properties
        member this.CreateDefaultProperties() : FractalProperties =
            {
                Dimension = 1.5
                ScalingFactor = 0.618  // Golden ratio
                IterationDepth = 5
                SelfSimilarityRatio = 0.5
                RecursionLimit = 10
                CompositionRules = ["scale"; "compose"; "recursive"]
            }

        /// Apply fractal transformation to a pattern
        member this.ApplyTransformation(pattern: string, transformation: FractalTransformation) : string =
            match transformation with
            | Scale factor ->
                // Scale pattern complexity by factor
                if factor > 1.0 then
                    pattern + " " + pattern.Substring(0, Math.Min(pattern.Length, int(factor * 10.0)))
                else
                    pattern.Substring(0, Math.Max(1, int(float pattern.Length * factor)))
            
            | Rotate angle ->
                // Rotate pattern elements (conceptual rotation in grammar space)
                let words = pattern.Split(' ')
                let rotatedWords = 
                    words 
                    |> Array.mapi (fun i word -> 
                        let rotationIndex = (i + int(angle / 45.0)) % words.Length
                        words.[rotationIndex])
                String.Join(" ", rotatedWords)
            
            | Translate (x, y) ->
                // Translate pattern by adding positional modifiers
                sprintf "(%s) [offset: %.2f, %.2f]" pattern x y
            
            | Compose transformations ->
                // Apply multiple transformations sequentially
                transformations |> List.fold this.ApplyTransformation pattern
            
            | Recursive (depth, innerTransformation) ->
                // Apply transformation recursively
                let mutable result = pattern
                for i in 1..depth do
                    result <- this.ApplyTransformation(result, innerTransformation)
                result
            
            | Conditional (predicate, ifTrue, ifFalse) ->
                // Apply conditional transformation based on pattern properties
                let shouldApplyTrue = 
                    match predicate with
                    | "length_gt_10" -> pattern.Length > 10
                    | "contains_recursive" -> pattern.Contains("recursive")
                    | "is_complex" -> pattern.Split(' ').Length > 3
                    | _ -> true
                
                if shouldApplyTrue then
                    this.ApplyTransformation(pattern, ifTrue)
                else
                    this.ApplyTransformation(pattern, ifFalse)

        /// Generate fractal node from rule
        member this.GenerateFractalNode(rule: FractalRule, level: int, parentId: string option) : FractalNode =
            let nodeId = sprintf "%s_L%d_%s" rule.Name level (Guid.NewGuid().ToString("N").[..7])
            
            // Apply transformations to base pattern
            let transformedPattern = 
                rule.Transformations 
                |> List.fold this.ApplyTransformation rule.BasePattern
            
            // Generate children if recursion conditions are met
            let children = 
                if level < rule.Properties.RecursionLimit && 
                   not (this.EvaluateTerminationCondition(rule.TerminationCondition, level, transformedPattern)) then
                    
                    // Create recursive children with scaled properties
                    let childProperties = {
                        rule.Properties with 
                            IterationDepth = rule.Properties.IterationDepth - 1
                            ScalingFactor = rule.Properties.ScalingFactor * rule.Properties.SelfSimilarityRatio
                    }
                    
                    let childRule = { rule with Properties = childProperties }
                    [this.GenerateFractalNode(childRule, level + 1, Some nodeId)]
                else
                    []
            
            {
                Id = nodeId
                Name = rule.Name
                Pattern = transformedPattern
                Children = children
                Properties = rule.Properties
                Transformations = rule.Transformations
                Metadata = Map.ofList [
                    ("level", box level)
                    ("generated_at", box DateTime.UtcNow)
                    ("pattern_length", box transformedPattern.Length)
                ]
                Level = level
                ParentId = parentId
            }

        /// Evaluate termination condition for fractal recursion
        member private this.EvaluateTerminationCondition(condition: string, level: int, pattern: string) : bool =
            match condition with
            | "max_depth_5" -> level >= 5
            | "pattern_too_long" -> pattern.Length > 1000
            | "complexity_threshold" -> pattern.Split(' ').Length > 50
            | "always_terminate" -> true
            | "never_terminate" -> false
            | _ -> level >= 10  // Default safety limit

        /// Generate complete fractal grammar
        member this.GenerateFractalGrammar(fractalGrammar: FractalGrammar) : FractalGenerationResult =
            let startTime = DateTime.UtcNow
            let mutable totalIterations = 0
            let errors = ResizeArray<string>()
            let warnings = ResizeArray<string>()
            
            try
                // Generate fractal tree from all rules
                let fractalTrees = 
                    fractalGrammar.FractalRules
                    |> List.map (fun rule ->
                        totalIterations <- totalIterations + 1
                        this.GenerateFractalNode(rule, 0, None))
                
                // Compose generated patterns into complete grammar
                let generatedGrammar = this.ComposeFractalGrammar(fractalGrammar, fractalTrees)
                
                // Create root fractal tree
                let rootTree = {
                    Id = "root"
                    Name = fractalGrammar.Name
                    Pattern = generatedGrammar
                    Children = fractalTrees
                    Properties = fractalGrammar.GlobalProperties
                    Transformations = []
                    Metadata = Map.ofList [
                        ("total_rules", box fractalGrammar.FractalRules.Length)
                        ("generation_time", box (DateTime.UtcNow - startTime))
                    ]
                    Level = 0
                    ParentId = None
                }
                
                {
                    Success = true
                    GeneratedGrammar = generatedGrammar
                    FractalTree = rootTree
                    IterationsPerformed = totalIterations
                    ComputationTime = DateTime.UtcNow - startTime
                    MemoryUsage = int64 (generatedGrammar.Length * 2) // Rough estimate
                    ErrorMessages = errors |> Seq.toList
                    Warnings = warnings |> Seq.toList
                }
            
            with
            | ex ->
                errors.Add(sprintf "Fractal generation failed: %s" ex.Message)
                {
                    Success = false
                    GeneratedGrammar = ""
                    FractalTree = {
                        Id = "error"
                        Name = "Error"
                        Pattern = ""
                        Children = []
                        Properties = this.CreateDefaultProperties()
                        Transformations = []
                        Metadata = Map.empty
                        Level = 0
                        ParentId = None
                    }
                    IterationsPerformed = totalIterations
                    ComputationTime = DateTime.UtcNow - startTime
                    MemoryUsage = 0L
                    ErrorMessages = errors |> Seq.toList
                    Warnings = warnings |> Seq.toList
                }

        /// Compose fractal trees into complete grammar
        member private this.ComposeFractalGrammar(fractalGrammar: FractalGrammar, trees: FractalNode list) : string =
            let header = sprintf "// Fractal Grammar: %s v%s\n// Generated: %s\n\n" 
                            fractalGrammar.Name 
                            fractalGrammar.Version 
                            (DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss"))
            
            let baseGrammarContent = fractalGrammar.BaseGrammar.Content
            
            let fractalRules = 
                trees
                |> List.map (fun tree -> this.TreeToGrammarRules(tree))
                |> String.concat "\n\n"
            
            header + baseGrammarContent + "\n\n// Fractal Extensions\n" + fractalRules

        /// Convert fractal tree to grammar rules
        member private this.TreeToGrammarRules(tree: FractalNode) : string =
            let mainRule = sprintf "%s = %s ;" tree.Name tree.Pattern
            
            let childRules = 
                tree.Children
                |> List.map this.TreeToGrammarRules
                |> String.concat "\n"
            
            if String.IsNullOrEmpty(childRules) then
                mainRule
            else
                mainRule + "\n" + childRules

        /// Analyze fractal grammar complexity
        member this.AnalyzeFractalComplexity(fractalGrammar: FractalGrammar) : Map<string, obj> =
            let totalRules = fractalGrammar.FractalRules.Length
            let avgDimension = 
                fractalGrammar.FractalRules 
                |> List.map (fun r -> r.Properties.Dimension) 
                |> List.average
            
            let maxDepth = 
                fractalGrammar.FractalRules 
                |> List.map (fun r -> r.Properties.RecursionLimit) 
                |> List.max
            
            Map.ofList [
                ("total_rules", box totalRules)
                ("average_dimension", box avgDimension)
                ("max_recursion_depth", box maxDepth)
                ("composition_complexity", box fractalGrammar.CompositionGraph.Count)
                ("estimated_generation_time", box (totalRules * maxDepth * 10)) // milliseconds
            ]

    /// Fractal grammar builder for fluent API
    type FractalGrammarBuilder() =
        let mutable name = "FractalGrammar"
        let mutable version = "1.0"
        let mutable baseGrammar = Grammar.createInline "base" ""
        let mutable rules = []
        let mutable properties = FractalGrammarEngine().CreateDefaultProperties()
        
        member this.WithName(grammarName: string) =
            name <- grammarName
            this
        
        member this.WithVersion(grammarVersion: string) =
            version <- grammarVersion
            this
        
        member this.WithBaseGrammar(grammar: Grammar) =
            baseGrammar <- grammar
            this
        
        member this.AddFractalRule(rule: FractalRule) =
            rules <- rule :: rules
            this
        
        member this.WithGlobalProperties(props: FractalProperties) =
            properties <- props
            this
        
        member this.Build() : FractalGrammar =
            {
                Name = name
                Version = version
                BaseGrammar = baseGrammar
                FractalRules = List.rev rules
                GlobalProperties = properties
                CompositionGraph = Map.empty
                GenerationHistory = []
                Metadata = GrammarMetadata.createDefault name
            }

    /// Fractal grammar service
    type FractalGrammarService() =
        let engine = FractalGrammarEngine()
        
        /// Create a simple fractal rule
        member this.CreateSimpleFractalRule(name: string, basePattern: string) : FractalRule =
            {
                Name = name
                BasePattern = basePattern
                RecursivePattern = Some (sprintf "(%s)*" basePattern)
                TerminationCondition = "max_depth_5"
                Transformations = [Scale 0.8; Recursive (3, Scale 0.9)]
                Properties = engine.CreateDefaultProperties()
                Dependencies = []
            }
        
        /// Generate fractal grammar from simple specification
        member this.GenerateFromSpec(name: string, patterns: string list) : FractalGenerationResult =
            let builder = FractalGrammarBuilder()
            let fractalGrammar = 
                builder
                    .WithName(name)
                    .WithVersion("1.0")
                    .WithBaseGrammar(Grammar.createInline "base" "// Base grammar")
                
            // Add fractal rules for each pattern
            for pattern in patterns do
                let rule = this.CreateSimpleFractalRule(sprintf "rule_%s" (pattern.Replace(" ", "_")), pattern)
                fractalGrammar.AddFractalRule(rule) |> ignore
            
            let finalGrammar = fractalGrammar.Build()
            engine.GenerateFractalGrammar(finalGrammar)
