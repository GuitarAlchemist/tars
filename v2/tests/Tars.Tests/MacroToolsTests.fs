namespace Tars.Tests

open Xunit
open Tars.Core
open Tars.Metascript
open Tars.Tools

module MacroToolsTests =

    // Mock Registry for testing
    // Alias to avoid conflict with Tars.Core.Domain
    module MetaDomain = Tars.Metascript.Domain

    type MockRegistry() =
        let store =
            System.Collections.Concurrent.ConcurrentDictionary<string, MetaDomain.Workflow>()

        interface IMacroRegistry with
            member _.Register(workflow: MetaDomain.Workflow) =
                task { store.AddOrUpdate(workflow.Name, workflow, (fun _ _ -> workflow)) |> ignore }

            member _.Get(name: string) =
                task {
                    if store.ContainsKey(name) then
                        return Some store.[name]
                    else
                        return None
                }

            member _.List() =
                task { return store.Values |> Seq.toList }

            member _.Delete(name: string) =
                task {
                    let mutable removed = Unchecked.defaultof<MetaDomain.Workflow>
                    return store.TryRemove(name, &removed)
                }

    let createTestWorkflow name =
        ({ Name = name
           Description = "Test Macro"
           Version = "1.0.0"
           Inputs = []
           Steps = [] }
        : MetaDomain.Workflow)

    [<Fact>]
    let ``register_macro: valid json registers macro`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            let registry = MockRegistry()
            let tools = MacroTools.getTools registry
            let registerTool = tools |> List.find (fun t -> t.Name = "register_macro")

            let validJson =
                """
            {
                "name": "test_macro",
                "description": "A test macro",
                "version": "1.0.0",
                "inputs": [],
                "steps": [
                    {
                        "id": "step1",
                        "type": "agent",
                        "dependsOn": null,
                        "agent": "test",
                        "tool": null,
                        "instruction": "do something",
                        "params": null,
                        "context": null,
                        "outputs": null,
                        "tools": null
                    }
                ]
            }
            """

            let! result = registerTool.Execute validJson |> Async.StartAsTask

            match result with
            | Result.Ok msg -> Assert.Contains("registered successfully", msg)
            | Result.Error e ->
                System.IO.File.WriteAllText("debug_macros.log", e)
                failwith $"Tool execute failed: {e}"

            // Verify it's in registry
            let iRegistry = registry :> IMacroRegistry
            let! stored = iRegistry.Get("test_macro")
            Assert.True(stored.IsSome)
            Assert.Equal("test_macro", stored.Value.Name)
        }

    [<Fact>]
    let ``register_macro: invalid json returns error`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            let registry = MockRegistry()
            let tools = MacroTools.getTools registry
            let registerTool = tools |> List.find (fun t -> t.Name = "register_macro")

            let invalidJson = "{ invalid: json }"

            let! result = registerTool.Execute invalidJson |> Async.StartAsTask

            match result with
            | Result.Error msg -> Assert.Contains("Parse error", msg)
            | Result.Ok _ -> Assert.Fail("Should have failed parsing")
        }

    [<Fact>]
    let ``list_macros: returns formatted list`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            let registry = MockRegistry()
            let iRegistry = registry :> IMacroRegistry
            let! _ = iRegistry.Register(createTestWorkflow "macro1")
            let! _ = iRegistry.Register(createTestWorkflow "macro2")

            let tools = MacroTools.getTools registry
            let listTool = tools |> List.find (fun t -> t.Name = "list_macros")

            let! result = listTool.Execute "" |> Async.StartAsTask

            match result with
            | Result.Ok msg ->
                Assert.Contains("macro1", msg)
                Assert.Contains("macro2", msg)
            | Result.Error e -> Assert.Fail($"Tool failed: {e}")
        }

    [<Fact>]
    let ``get_macro: returns json definition`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            let registry = MockRegistry()
            let iRegistry = registry :> IMacroRegistry
            let! _ = iRegistry.Register(createTestWorkflow "my_macro")

            let tools = MacroTools.getTools registry
            let getTool = tools |> List.find (fun t -> t.Name = "get_macro")

            let! result = getTool.Execute "my_macro" |> Async.StartAsTask

            match result with
            | Result.Ok json ->
                Assert.Contains("my_macro", json)
                Assert.Contains("Test Macro", json)
            | Result.Error e -> Assert.Fail($"Tool failed: {e}")
        }

    [<Fact>]
    let ``get_macro: returns error if missing`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            let registry = MockRegistry()
            let tools = MacroTools.getTools registry
            let getTool = tools |> List.find (fun t -> t.Name = "get_macro")

            let! result = getTool.Execute "missing_macro" |> Async.StartAsTask

            match result with
            | Result.Error msg -> Assert.Contains("not found", msg)
            | Result.Ok _ -> Assert.Fail("Should have failed")
        }
