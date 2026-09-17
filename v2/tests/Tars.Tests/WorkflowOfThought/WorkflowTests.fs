namespace Tars.Tests.WorkflowOfThought

open System.IO
open Xunit
open Tars.Core.WorkflowOfThought
open Tars.DSL.Wot
open Tars.Cortex.WoTTypes
open Tars.Interface.Cli.Commands

// Issue #44: loading and running a .wot.trsx lived inside WotCommand, reachable only through the CLI.
module WorkflowTests =

    let private fixture name =
        let asmDir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location)
        Path.Combine(asmDir, "WorkflowOfThought", "fixtures", name)

    let private loadSample () =
        match Workflow.load (fixture "sample.wot.trsx") with
        | Ok plan -> plan
        | Error err -> failwith (String.concat "; " (Workflow.describe err))

    let private node id =
        [ $"  node \"{id}\" kind=\"reason\" {{"
          $"    goal = \"Think about {id}\""
          $"    output = \"{id}_out\""
          "  }" ]

    [<Fact>]
    let ``load compiles a workflow file into an ordered plan`` () =
        let plan = loadSample ()

        Assert.Equal("sample_integration_test", plan.Goal)
        Assert.Equal<string>([ "analyse"; "research"; "evaluate"; "refine"; "summarise" ], plan.Steps |> List.map (fun s -> s.Id))

    [<Fact>]
    let ``load reports a missing file as a parse failure`` () =
        match Workflow.load (fixture "does-not-exist.wot.trsx") with
        | Error(ParseFailed [ e ]) ->
            Assert.Equal(0, e.Line)
            Assert.Contains("File not found", e.Message)
        | other -> failwith $"Expected a single parse error, got {other}"

    [<Fact>]
    let ``loadLines reports a graph that is not a chain as a compile failure`` () =
        let lines =
            [ "workflow \"branchy\" {"
              yield! node "a"
              yield! node "b"
              yield! node "c"
              "  edge \"a\" -> \"b\""
              "  edge \"b\" -> \"c\""
              "  edge \"a\" -> \"c\""
              "}" ]

        match Workflow.loadLines lines with
        | Error(CompileFailed _ as err) ->
            Assert.All(Workflow.describe err, fun line -> Assert.StartsWith("Compile error", line))
        | other -> failwith $"Expected a compile failure, got {other}"

    [<Fact>]
    let ``toCortexPlan keeps step order as a chain of edges`` () =
        let plan = loadSample ()
        let cortex = WotExecution.toCortexPlan plan

        Assert.Equal("analyse", cortex.EntryNode)
        Assert.Equal("sample_integration_test", cortex.Metadata.SourceGoal)
        Assert.Equal<string>(plan.Steps |> List.map (fun s -> s.Id), cortex.Nodes |> List.map (fun n -> n.Id))
        Assert.All(cortex.Nodes, fun n -> Assert.Equal(WoTNodeKind.Reason, n.Kind))

        Assert.Equal<string>(
            [ "analyse->research"; "research->evaluate"; "evaluate->refine"; "refine->summarise" ],
            cortex.Edges |> List.map (fun e -> $"{e.From}->{e.To}")
        )

    [<Fact>]
    let ``runV0 executes a loaded workflow with a stub reasoner`` () =
        if not (Tars.Tests.TestHelpers.requireTools ()) then
            ()
        else
            let reasoner =
                { new IReasoner with
                    member _.Reason(_, _, _, _, _) =
                        async { return Ok { Content = "stub"; Usage = None } } }

            let result =
                WotExecution.runV0 (WotExecution.v0Tools ()) reasoner None None ReasonStepMode.Stub (loadSample ())
                |> Async.RunSynchronously

            match result with
            | Ok(ctx, _, traces) ->
                Assert.Equal(5, traces.Length)
                Assert.Contains("summary", ctx.Vars |> Map.keys)
            | Error(e, _) -> failwith e
