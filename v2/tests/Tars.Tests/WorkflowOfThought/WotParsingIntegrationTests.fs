namespace Tars.Tests.WorkflowOfThought

open Xunit
open Tars.DSL.Wot
open Tars.DSL.Wot.WotParser
open Tars.DSL.Wot.WotCompiler
open System.IO

module WotParsingIntegrationTests =

    [<Fact>]
    let ``parse .wot.trsx then compile - golden path ok`` () =
        // Use a relative path from the test execution directory, or copy the file
        // Assuming test run from root or tests dir, let's look for known location
        // Adjust this if needed based on test runner's CWD
        let repoRoot = "../../../../../" 
        let path = Path.Combine(repoRoot, "sample.wot.trsx")
        
        let path = Path.GetFullPath(path) // normalize
        
        // Ensure file exists for the test
        if not (File.Exists path) then
            // Fallback: try finding it relative to solution if running from bin
            // Or just skip/fail with clear message
            failwith $"Test file not found at: {path}. CWD: {Directory.GetCurrentDirectory()}"

        match WotParser.parseFile path with
        | Error es ->
            let msg = es |> List.map (fun e -> $"L{e.Line}: {e.Message}") |> String.concat "\n"
            failwith $"Parse failed:\n{msg}"
        | Ok wf ->
            match compileWorkflowToPlanParsed wf with
            | Error errs ->
                let msg = errs |> List.map string |> String.concat "\n"
                failwith $"Compile failed:\n{msg}"
            | Ok plan ->
                // Assertions for correctness
                Assert.Equal(5, plan.Steps.Length)
                Assert.Equal("sample.wot.trsx", plan.Goal)
