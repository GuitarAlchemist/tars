namespace Tars.Registry.Tests

open System.Text.Json.Nodes
open Xunit
open Tars.Registry

/// Sample annotated skill exercised by the registry discovery tests.
/// Lives in a public module so reflection can find it.
module SampleSkills =

    [<TarsSkill("test.echo", "test")>]
    let echo (input: JsonNode) : JsonNode =
        let result = JsonObject()
        result.["echo"] <- (if isNull input then JsonValue.Create("") :> JsonNode else input.DeepClone())
        result :> JsonNode

    /// Deliberately NOT annotated — discovery must skip this.
    let unannotated (input: JsonNode) : JsonNode = input

module RegistryTests =

    [<Fact>]
    let ``discoverFindsAnnotatedMethod`` () =
        let found = Registry.byName "test.echo"
        Assert.True(Option.isSome found, "test.echo should be discovered by reflection")
        let skill = found.Value
        Assert.Equal("test.echo", skill.Name)
        Assert.Equal("test", skill.Domain)

    [<Fact>]
    let ``discoverInvokesHandler`` () =
        let skill =
            match Registry.byName "test.echo" with
            | Some s -> s
            | None -> failwith "test.echo not registered"

        let payload : JsonNode = JsonValue.Create("hello") :> JsonNode
        match skill.Handler payload with
        | Ok node ->
            Assert.NotNull(node)
            let echoed = node.["echo"]
            Assert.NotNull(echoed)
            Assert.Equal("hello", echoed.GetValue<string>())
        | Error msg ->
            failwithf "Handler returned Error: %s" msg

    [<Fact>]
    let ``unannotatedMethodNotInRegistry`` () =
        let all = Registry.all ()
        // No skill in the registry should point to the unannotated method
        // (we only have one annotated test skill, so any extra entry from
        // this module would be a discovery bug).
        let testDomainSkills =
            all |> Array.filter (fun s -> s.Domain = "test")
        Assert.Equal(1, testDomainSkills.Length)
        Assert.Equal("test.echo", testDomainSkills.[0].Name)
