namespace Tars.Core

open System
open Tars.Core.TemporalKnowledgeGraph

module GrammarDistillation =

    /// Distill grammar rules from communities of code patterns
    type GrammarDistiller(graph: TemporalGraph) =
        
        /// Distill rules from a list of entities (representing a community)
        member this.DistillRules(entities: TarsEntity list) : GrammarRuleEntity list =
            // Filter for CodePattern entities
            let patterns = 
                entities 
                |> List.choose (function 
                    | CodePatternE p -> Some p 
                    | _ -> None)
            
            // Group by category
            patterns
            |> List.groupBy (fun p -> p.Category)
            |> List.choose (fun (category, group) ->
                // Heuristic: If we have > 2 patterns of the same category, distill a rule
                if group.Length > 2 then
                    let name = 
                        match category with
                        | Structural -> "Standard Structural Pattern"
                        | Behavioral -> "Common Behavioral Pattern"
                        | Creational -> "Factory/Builder Pattern"
                        | Agentic -> "Agent Interaction Pattern"
                        | Architectural -> "Architectural Style"
                        | CustomCategory c -> $"Custom {c} Pattern"
                        
                    Some {
                        Name = name
                        Production = $"// TODO: Synthesize production from {group.Length} examples"
                        Examples = group |> List.map (fun p -> p.Name)
                        DistilledFrom = group |> List.map (fun p -> p.Name)
                        Version = 1
                    }
                else
                    None
            )

        /// Auto-generate rules for all communities in the graph
        /// Note: Requires community detection to have run first
        member this.DistillFromGraph() : GrammarRuleEntity list =
            // In a real impl, we'd query communities. 
            // For now, we'll just look at all nodes as one "global" community
            // to demonstrate the logic.
            let nodes = graph.GetNodes()
            this.DistillRules(nodes)

    /// Generate F# code from grammar rules
    type CodeGenerator() =
        
        /// Generate a Discriminated Union from a grammar rule
        member this.GenerateDU(rule: GrammarRuleEntity) : string =
            let cases = 
                rule.Examples 
                |> List.map (fun e -> $"    | {e}")
                |> String.concat Environment.NewLine
            
            let typeName = rule.Name.Replace(" ", "")
            
            $"type {typeName} ={Environment.NewLine}{cases}"

        /// Generate a Computation Expression builder from a grammar rule
        member this.GenerateCE(rule: GrammarRuleEntity) : string =
            let typeName = rule.Name.Replace(" ", "")
            let builderName = $"{typeName.ToLowerInvariant()}Builder"
            
            $"""
type {typeName}Builder() =
    member _.Yield(_) = ()
    member _.Run(state) = state

let {builderName} = {typeName}Builder()
"""

    /// Manages the lifecycle of grammar evolution
    type HotReloadManager(distiller: GrammarDistiller, generator: CodeGenerator) =
        
        let mutable currentVersion = 0
        let mutable activeRules: Map<string, GrammarRuleEntity> = Map.empty
        
        /// Check for new patterns and evolve grammar if needed
        member this.Evolve() : (string * string) list =
            let newRules = distiller.DistillFromGraph()
            
            // Find rules that are new or changed
            let updates = 
                newRules 
                |> List.filter (fun r -> 
                    match activeRules.TryFind r.Name with
                    | Some existing -> existing.Version < r.Version // In a real system, we'd check content too
                    | None -> true
                )
            
            if updates.IsEmpty then
                []
            else
                // Update active rules
                updates |> List.iter (fun r -> 
                    activeRules <- activeRules |> Map.add r.Name r
                )
                currentVersion <- currentVersion + 1
                
                // Generate code for updates
                updates |> List.map (fun r ->
                    let du = generator.GenerateDU(r)
                    let ce = generator.GenerateCE(r)
                    (r.Name, $"{du}{Environment.NewLine}{ce}")
                )
