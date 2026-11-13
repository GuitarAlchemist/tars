namespace Tars.Tests

open System
open System.IO
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
        Testing.reset ()
        let written = descriptor "read_file" [ "path" ] (Approval.observes "reads a file")
        describe written

        Assert.Equal(Some written, tryFind "read_file")
        Assert.True(isDescribed "read_file")

    [<Fact>]
    let ``a tool nobody described is absent, not wrong`` () =
        Testing.reset ()

        // The whole point of a sidecar keyed by name: a tool with no description is
        // simply not offered yet. Adding a field to `Tool` would instead have made
        // every undescribed tool wrong on the day the type changed, and there are
        // hundreds of them.
        Assert.Equal(None, tryFind "never_described")
        Assert.False(isDescribed "never_described")

    [<Fact>]
    let ``casing does not make a different tool`` () =
        Testing.reset ()
        describe (descriptor "git_commit" [] (Approval.mutates "commits"))

        // Names are typed by hand into graphs, so `Git_Commit` has to be the same tool.
        Assert.True((tryFind "Git_Commit").IsSome)
        Assert.True((tryFind "GIT_COMMIT").IsSome)
        Assert.True((tryFind "  git_commit  ").IsSome)

    [<Fact>]
    let ``a name that is not a name finds nothing rather than throwing`` () =
        Testing.reset ()
        Assert.Equal(None, tryFind null)
        Assert.Equal(None, tryFind "")
        Assert.Equal(None, tryFind "   ")

    [<Fact>]
    let ``describing the same tool again replaces the description`` () =
        Testing.reset ()
        describe (descriptor "tool" [] (Approval.observes "the first answer"))
        describe (descriptor "tool" [] (Approval.mutates "the second answer"))

        Assert.Equal(1, describedCount ())
        Assert.Equal("the second answer", (tryFind "tool").Value.Approval.Effect)

    [<Fact>]
    let ``the count is what an editor shows against the registry's total`` () =
        Testing.reset ()
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
            objectSchema [ "path", "string", "The file to read."; "lines", "integer", "How many lines." ] [ "path" ]

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
        // `objectSchema` writes through `Utf8JsonWriter`, which escapes for us. It used
        // to concatenate strings, which is what these three tests were written to catch;
        // they stay because the property is the contract, not the implementation.
        let schema =
            objectSchema [ "path", "string", """A "quoted" word and a \backslash.""" ] [ "path" ]

        use doc = JsonDocument.Parse schema

        Assert.Equal(
            """A "quoted" word and a \backslash.""",
            doc.RootElement.GetProperty("properties").GetProperty("path").GetProperty("description").GetString()
        )

    [<Fact>]
    let ``a control character in a description does not break the schema`` () =
        // The concatenated version escaped backslashes and quotes and nothing else, so
        // a newline or a tab produced invalid JSON — and these schemas are what an
        // editor parses. A description someone writes later must not break the catalog.
        let awkward = "First line.\nSecond\tline.\r\nAnd a \u0007 bell."
        let schema = objectSchema [ "path", "string", awkward ] [ "path" ]

        use doc = JsonDocument.Parse schema

        Assert.Equal(
            awkward,
            doc.RootElement.GetProperty("properties").GetProperty("path").GetProperty("description").GetString()
        )

    [<Fact>]
    let ``a required name that needs escaping is escaped`` () =
        // Required names used to be interpolated with no escaping at all, so a quote
        // or a backslash in one produced invalid JSON.
        let odd = """a "quoted\name"""
        let schema = objectSchema [ odd, "string", "A field." ] [ odd ]

        use doc = JsonDocument.Parse schema

        let required =
            doc.RootElement.GetProperty("required").EnumerateArray()
            |> Seq.map (fun e -> e.GetString())
            |> Seq.toList

        Assert.Equal<string>([ odd ], required)
        Assert.True(doc.RootElement.GetProperty("properties").TryGetProperty(odd) |> fst)

    [<Fact>]
    let ``a schema with no properties is still valid JSON`` () =
        use doc = JsonDocument.Parse(objectSchema [] [])
        Assert.Equal("object", doc.RootElement.GetProperty("type").GetString())
        Assert.Empty(doc.RootElement.GetProperty("required").EnumerateArray())

    [<Fact>]
    let ``nothing in the running system resets the registry`` () =
        // `ToolMetadata.Testing.reset` is public so that `Tars.Core.fsproj` does not
        // need `InternalsVisibleTo Tars.Tests`, which opened every internal of the
        // assembly to reach this one function. The compiler used to keep production
        // code out; this does, by reading the source tree.
        //
        // It is a weaker guarantee and it is meant to be a cheaper one. If this ever
        // fails, the answer is to delete the call, not to widen the test.
        //
        // It is a text scan, so even a comment naming `ToolMetadata.Testing` in a source
        // file trips it. That is how this test was checked not to be vacuous, and it
        // fails in the safe direction: a false alarm costs a minute, a missed call costs
        // a production path that can wipe the registry.
        let rec v2Root (dir: DirectoryInfo) =
            if isNull (box dir) then failwith "could not find v2/ above the test directory"
            elif dir.GetFiles("Tars.sln").Length > 0 then dir
            else v2Root dir.Parent

        let src = Path.Combine((v2Root (DirectoryInfo(Directory.GetCurrentDirectory()))).FullName, "src")

        let callers =
            Directory.EnumerateFiles(src, "*.fs", SearchOption.AllDirectories)
            |> Seq.filter (fun f ->
                let parts = f.Replace('\\', '/').Split('/')
                not (parts |> Array.exists (fun p -> p = "obj" || p = "bin")))
            |> Seq.filter (fun f -> File.ReadAllText(f).Contains "ToolMetadata.Testing")
            |> Seq.map (fun f -> Path.GetRelativePath(src, f))
            |> Seq.toList

        Assert.True(
            callers.IsEmpty,
            "production code calls the test-only registry reset: " + String.concat ", " callers
        )
