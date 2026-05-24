namespace Tars.Tests.WorkflowOfThought

open Xunit
open Tars.DSL.Wot
open Tars.DSL.Wot.WotCompiler
open System.IO

module WotParsingIntegrationTests =

    [<Fact>]
    let ``parse .wot.trsx then compile - golden path ok`` () =
        // Fixture is copied to output via Content item in fsproj
        let asmDir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location)
        let path = Path.Combine(asmDir, "WorkflowOfThought", "fixtures", "sample.wot.trsx")

        if not (File.Exists path) then
            failwith $"Test fixture not found at: {path}. CWD: {Directory.GetCurrentDirectory()}"

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
                Assert.Equal(5, plan.Steps.Length)
                Assert.Equal("sample_integration_test", plan.Goal)
