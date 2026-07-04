namespace Tars.Tests

open Xunit
open Tars.Tools
open Tars.Core

module RegistryDiscoveryTests =
    [<Fact>]
    let ``Discovery finds annotated skills`` () =
        // Force loading of Tars.Evolution
        let _ = Tars.Evolution.McpGrammarTools.createTools ()

        let skills = SkillRegistry.All.Value

        // Let's print all skills found for debugging if it fails
        if not (skills |> Seq.exists (fun s -> s.Name = "standard.run_command")) then
            printfn "Skills found: %A" (skills |> Seq.map (fun s -> s.Name) |> Seq.toList)

        Assert.Contains(skills, (fun s -> s.Name = "standard.run_command"))
        Assert.Contains(skills, (fun s -> s.Name = "evolution.ingest_ga_traces"))
