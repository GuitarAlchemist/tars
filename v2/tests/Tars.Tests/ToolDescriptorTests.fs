namespace Tars.Tests

open System
open System.Text.Json
open Xunit
open Tars.Core
open Tars.Core.ToolMetadata
open Tars.Tools

/// The first tools described, checked against the schemas they publish.
///
/// A descriptor is a promise made to whoever reads the catalog. These hold the
/// promises internally consistent; whether each one matches the tool it describes was
/// settled by reading that tool's definition, and is recorded in the descriptor file.
/// The metadata registry is process-wide, so the modules that write to it share a
/// collection: xUnit runs collections in parallel, and one test calling `clear()`
/// while another counts is a race, not a failure of either test.
[<Xunit.Collection("ToolMetadata registry")>]
module ToolDescriptorTests =

    let private schemaOf (d: ToolDescriptor) = JsonDocument.Parse d.InputSchema

    let private propertyNames (d: ToolDescriptor) =
        use doc = schemaOf d

        match doc.RootElement.TryGetProperty "properties" with
        | true, props -> props.EnumerateObject() |> Seq.map (fun p -> p.Name) |> Seq.toList
        | _ -> []

    [<Fact>]
    let ``every built-in descriptor publishes a schema that parses`` () =
        for d in ToolDescriptors.builtIn do
            use doc = schemaOf d
            Assert.Equal("object", doc.RootElement.GetProperty("type").GetString())

    [<Fact>]
    let ``nothing is required that the schema does not declare`` () =
        // A required field missing from `properties` would be a field no caller could
        // ever supply correctly, since the schemas say additionalProperties: false.
        for d in ToolDescriptors.builtIn do
            let declared = propertyNames d

            for required in d.Required do
                Assert.True(
                    List.contains required declared,
                    $"{d.Name} requires '{required}', which its schema does not declare"
                )

    [<Fact>]
    let ``the required list and the schema's required list agree`` () =
        // `Required` is kept beside the schema so a caller can check without a schema
        // parser. Two copies of the same fact drift unless something holds them equal.
        for d in ToolDescriptors.builtIn do
            use doc = schemaOf d

            let inSchema =
                doc.RootElement.GetProperty("required").EnumerateArray()
                |> Seq.map (fun e -> e.GetString())
                |> Seq.toList

            Assert.Equal<string>(d.Required, inSchema)

    [<Fact>]
    let ``every described tool says what running it does`` () =
        for d in ToolDescriptors.builtIn do
            Assert.False(String.IsNullOrWhiteSpace d.Approval.Effect, $"{d.Name} states no effect")

            // The effect is read by a person deciding whether to allow it, so it is a
            // sentence about what happens, not a category.
            Assert.True(d.Approval.Effect.Contains " ", $"{d.Name}'s effect is not a sentence")

    [<Fact>]
    let ``only the tools that reach nothing are automatically approved`` () =
        for d in ToolDescriptors.builtIn do
            match d.Approval.Tier with
            | ReadOnly -> ()
            | Mutates
            | Escapes ->
                Assert.False(
                    d.Approval.AutoApproved,
                    $"{d.Name} changes or escapes and would run without being asked about"
                )

    [<Fact>]
    let ``a shell is classed by its reach, not by the fact that it runs locally`` () =
        // The tier answers "what is the worst this can do". A shell command can run
        // `curl`, so it reaches the network as easily as it writes a file.
        let shell = ToolDescriptors.builtIn |> List.find (fun d -> d.Name = "run_shell")
        Assert.Equal(Escapes, shell.Approval.Tier)

    [<Fact>]
    let ``a field is required only when the tool has no usable default`` () =
        // These two look inconsistent and are not, so the asymmetry is pinned before
        // someone tidies it away.
        //
        // `count_lines` reads `path` off the JSON itself and defaults it to ".", so
        // omitting it works. `list_dir` goes through `ToolHelpers.parseStringArg`,
        // which on `{}` finds no property and falls back to the raw input — making the
        // tool look for a directory literally named "{}". Its optional case does not
        // work, so the descriptor does not offer one.
        let requiredOf name =
            ToolDescriptors.builtIn |> List.find (fun d -> d.Name = name) |> (fun d -> d.Required)

        Assert.Empty(requiredOf "count_lines")
        Assert.Equal<string>([ "path" ], requiredOf "list_dir")

    [<Fact>]
    let ``no tool is described twice`` () =
        let names = ToolDescriptors.builtIn |> List.map (fun d -> d.Name)
        Assert.Equal<string>(List.distinct names, names)

    [<Fact>]
    let ``registering the built-ins is safe to do more than once`` () =
        clear ()
        ToolDescriptors.registerBuiltIn ()
        let first = describedCount ()

        ToolDescriptors.registerBuiltIn ()

        Assert.Equal(first, describedCount ())
        Assert.Equal(ToolDescriptors.builtIn.Length, first)

    [<Fact>]
    let ``every described name is a tool that exists, and hash_text is left out on purpose`` () =
        if not (TestHelpers.requireTools ()) then
            () // WDAC blocks Tars.Tools; there is no assembly to scan
        else
            // Descriptors name their tools as strings, and nothing else in the stack
            // checks those strings against anything: `describe` accepts any name, and the
            // bridge only ever looks a descriptor up *starting from* a registered tool. So
            // a typo or a renamed tool does not fail — it leaves a descriptor nobody ever
            // consults, the tool silently goes back to being undescribed, and the editor
            // drops it from the catalog. This is the only test that would notice.
            let registry = ToolRegistry()
            registry.RegisterAssembly(Reflection.Assembly.GetAssembly(typeof<ToolRegistry>))

            let registered = registry.GetAll() |> List.map (fun t -> t.Name) |> Set.ofList
            Assert.NotEmpty(registered)

            let missing =
                ToolDescriptors.builtIn
                |> List.map (fun d -> d.Name)
                |> List.filter (registered.Contains >> not)

            Assert.True(
                missing.IsEmpty,
                "these descriptors name tools that do not exist: " + String.concat ", " missing
            )

            // `hash_text` hashes whatever it is handed, and the executor serializes a
            // node's arguments to JSON before calling a tool, so describing it as an object
            // would make it hash `{"text":"..."}` rather than the text. Until there is a
            // convention for "this one takes the bare string", absent is the honest answer.
            //
            // Asserting that it *is* registered is what makes the omission mean something:
            // without it the test would also pass once the tool was renamed or deleted.
            Assert.Contains("hash_text", registered)

            Assert.DoesNotContain("hash_text", ToolDescriptors.builtIn |> List.map (fun d -> d.Name))
