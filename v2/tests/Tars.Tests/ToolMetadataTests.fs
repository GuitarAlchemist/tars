namespace Tars.Tests

open System
open System.Text.Json
open Xunit
open Tars.Core
open Tars.Core.ToolMetadata

/// The sidecar that says what a tool expects and what running it does.
///
/// `Tool` carries a name, a description and `Execute: string -> ...`. That is enough
/// to call a tool you already understand and not enough to offer one in an editor:
/// nothing says what the string should contain, which fields are required, or whether
/// running it writes to disk, spends money or leaves the machine.
/// The metadata registry is process-wide, so the modules that write to it share a
/// collection: xUnit runs collections in parallel, and one test calling `clear()`
/// while another counts is a race, not a failure of either test.
[<Xunit.Collection("ToolMetadata registry")>]
module ToolMetadataTests =

    let private descriptor name required approval : ToolDescriptor =
        { Name = name
          InputSchema = objectSchema (required |> List.map (fun r -> r, "string", "A field.")) required
          Required = required
          Approval = approval }

    // =========================================================================
    // The registry
    // =========================================================================

    [<Fact>]
    let ``a described tool can be found again`` () =
        clear ()
        let written = descriptor "read_file" [ "path" ] (Approval.observes "reads a file")
        describe written

        Assert.Equal(Some written, tryFind "read_file")
        Assert.True(isDescribed "read_file")

    [<Fact>]
    let ``a tool nobody described is absent, not wrong`` () =
        clear ()

        // The whole point of a sidecar keyed by name: a tool with no description is
        // simply not offered yet. Adding a field to `Tool` would instead have made
        // every undescribed tool wrong on the day the type changed, and there are
        // hundreds of them.
        Assert.Equal(None, tryFind "never_described")
        Assert.False(isDescribed "never_described")

    [<Fact>]
    let ``casing does not make a different tool`` () =
        clear ()
        describe (descriptor "git_commit" [] (Approval.mutates "commits"))

        // Names are typed by hand into graphs, so `Git_Commit` has to be the same tool.
        Assert.True((tryFind "Git_Commit").IsSome)
        Assert.True((tryFind "GIT_COMMIT").IsSome)
        Assert.True((tryFind "  git_commit  ").IsSome)

    [<Fact>]
    let ``a name that is not a name finds nothing rather than throwing`` () =
        clear ()
        Assert.Equal(None, tryFind null)
        Assert.Equal(None, tryFind "")
        Assert.Equal(None, tryFind "   ")

    [<Fact>]
    let ``describing the same tool again replaces the description`` () =
        clear ()
        describe (descriptor "tool" [] (Approval.observes "the first answer"))
        describe (descriptor "tool" [] (Approval.mutates "the second answer"))

        Assert.Equal(1, describedCount ())
        Assert.Equal("the second answer", (tryFind "tool").Value.Approval.Effect)

    [<Fact>]
    let ``the count is what an editor shows against the registry's total`` () =
        clear ()
        Assert.Equal(0, describedCount ())

        describeAll
            [ descriptor "a" [] (Approval.observes "a")
              descriptor "b" [] (Approval.observes "b") ]

        Assert.Equal(2, describedCount ())
        Assert.Equal<string>([ "a"; "b" ], all () |> List.map (fun d -> d.Name))

    // =========================================================================
    // Tiers
    // =========================================================================

    [<Fact>]
    let ``the wire spelling of a tier is fixed, not derived from the case name`` () =
        // Derived from the case name, renaming a case would silently change a
        // published contract that other repos match on.
        Assert.Equal("read_only", ReadOnly.Wire)
        Assert.Equal("mutates", Mutates.Wire)
        Assert.Equal("escapes", Escapes.Wire)

    [<Fact>]
    let ``only a tool that reaches nothing runs without being asked about`` () =
        Assert.True((Approval.observes "reads a file").AutoApproved)
        Assert.False((Approval.mutates "writes a file").AutoApproved)
        Assert.False((Approval.escapes "calls an API").AutoApproved)

        Assert.Equal(ReadOnly, (Approval.observes "x").Tier)
        Assert.Equal(Mutates, (Approval.mutates "x").Tier)
        Assert.Equal(Escapes, (Approval.escapes "x").Tier)

    // =========================================================================
    // Schemas
    // =========================================================================

    [<Fact>]
    let ``the generated schema says what it claims to say`` () =
        let schema =
            objectSchema
                [ "path", "string", "The file to read."
                  "lines", "integer", "How many lines." ]
                [ "path" ]

        use doc = JsonDocument.Parse schema
        let root = doc.RootElement

        Assert.Equal("object", root.GetProperty("type").GetString())
        Assert.Equal("string", root.GetProperty("properties").GetProperty("path").GetProperty("type").GetString())
        Assert.Equal("integer", root.GetProperty("properties").GetProperty("lines").GetProperty("type").GetString())
        Assert.False(root.GetProperty("additionalProperties").GetBoolean())

        let required =
            root.GetProperty("required").EnumerateArray()
            |> Seq.map (fun e -> e.GetString())
            |> Seq.toList

        Assert.Equal<string>([ "path" ], required)

    [<Fact>]
    let ``a quote in a description does not break the schema`` () =
        // The schema is built by string concatenation, so anything a describer writes
        // has to survive being put inside JSON.
        let schema =
            objectSchema [ "path", "string", """A "quoted" word and a \backslash.""" ] [ "path" ]

        use doc = JsonDocument.Parse schema

        Assert.Equal(
            """A "quoted" word and a \backslash.""",
            doc.RootElement.GetProperty("properties").GetProperty("path").GetProperty("description").GetString()
        )

    [<Fact>]
    let ``a schema with no properties is still valid JSON`` () =
        use doc = JsonDocument.Parse(objectSchema [] [])
        Assert.Equal("object", doc.RootElement.GetProperty("type").GetString())
        Assert.Empty(doc.RootElement.GetProperty("required").EnumerateArray())
