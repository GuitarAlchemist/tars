module Tars.Tests.PersonaTests

open Xunit
open Tars.Core.Persona
open Tars.Core.PersonaRegistry

// ============================================================================
// Persona Type Tests
// ============================================================================

[<Fact>]
let ``createPersona creates minimal persona`` () =
    let p = createPersona "test" "Test Persona" "Act as a test"
    Assert.Equal("test", p.Id)
    Assert.Equal("Test Persona", p.Name)
    Assert.Equal("Act as a test", p.Role)
    Assert.Equal(Markdown, p.DefaultFormat)
    Assert.Empty(p.Constraints)

[<Fact>]
let ``formatInstruction returns correct string for each format`` () =
    Assert.Contains("markdown", formatInstruction Markdown)
    Assert.Contains("JSON", formatInstruction JSON)
    Assert.Contains("table", formatInstruction Table)
    Assert.Contains("bullet", formatInstruction BulletPoints)
    Assert.Equal("", formatInstruction Prose)
    Assert.Equal("custom format", formatInstruction (Custom "custom format"))

[<Fact>]
let ``buildRtfdPrompt includes role and task`` () =
    let persona = createPersona "test" "Tester" "Act as an expert tester"

    let rtfd =
        { Persona = persona
          Task = "Test this code"
          Format = None
          Details = None }

    let prompt = buildRtfdPrompt rtfd

    Assert.Contains("Act as an expert tester", prompt)
    Assert.Contains("Test this code", prompt)

[<Fact>]
let ``buildRtfdPrompt includes constraints when present`` () =
    let persona =
        { createPersona "test" "Tester" "Role" with
            Constraints = [ "Be thorough"; "Check edge cases" ] }

    let rtfd =
        { Persona = persona
          Task = "Test"
          Format = None
          Details = None }

    let prompt = buildRtfdPrompt rtfd

    Assert.Contains("Constraints", prompt)
    Assert.Contains("Be thorough", prompt)
    Assert.Contains("Check edge cases", prompt)

[<Fact>]
let ``buildRtfdPrompt includes details when present`` () =
    let persona = createPersona "test" "Tester" "Role"

    let rtfd =
        { Persona = persona
          Task = "Test"
          Format = None
          Details = Some "Focus on security" }

    let prompt = buildRtfdPrompt rtfd

    Assert.Contains("Focus on security", prompt)

// ============================================================================
// PersonaRegistry Tests
// ============================================================================

[<Fact>]
let ``PersonaRegistry registers and retrieves persona`` () =
    let registry = PersonaRegistry()
    let persona = createPersona "test-id" "Test" "Role"

    let result = registry.Register(persona)
    Assert.True(Result.isOk result)

    let retrieved = registry.Get("test-id")
    Assert.True(retrieved.IsSome)
    Assert.Equal("Test", retrieved.Value.Name)

[<Fact>]
let ``PersonaRegistry rejects empty ID`` () =
    let registry = PersonaRegistry()
    let persona = createPersona "" "Test" "Role"

    let result = registry.Register(persona)
    Assert.True(Result.isError result)

[<Fact>]
let ``PersonaRegistry rejects empty Role`` () =
    let registry = PersonaRegistry()
    let persona = createPersona "id" "Test" ""

    let result = registry.Register(persona)
    Assert.True(Result.isError result)

[<Fact>]
let ``PersonaRegistry removes persona`` () =
    let registry = PersonaRegistry()
    let persona = createPersona "test-id" "Test" "Role"
    registry.Register(persona) |> ignore

    Assert.True(registry.Exists("test-id"))
    let removed = registry.Remove("test-id")
    Assert.True(removed)
    Assert.False(registry.Exists("test-id"))

[<Fact>]
let ``PersonaRegistry lists all personas`` () =
    let registry = PersonaRegistry()
    registry.Register(createPersona "p1" "Persona1" "Role1") |> ignore
    registry.Register(createPersona "p2" "Persona2" "Role2") |> ignore

    let all = registry.List()
    Assert.Equal(2, all.Length)

[<Fact>]
let ``defaultRegistry has built-in personas`` () =
    let personas = defaultRegistry.List()
    Assert.True(personas.Length >= 5)
    Assert.True(defaultRegistry.Exists("code-reviewer"))
    Assert.True(defaultRegistry.Exists("documentation-writer"))
    Assert.True(defaultRegistry.Exists("test-engineer"))

[<Fact>]
let ``withPersona returns prompt for valid persona`` () =
    match withPersona "code-reviewer" "Review this code" None None with
    | Result.Ok prompt ->
        Assert.Contains("code", prompt.ToLower())
        Assert.Contains("Review this code", prompt)
    | Result.Error e -> Assert.Fail($"Expected Ok, got Error: {e}")

[<Fact>]
let ``withPersona returns error for unknown persona`` () =
    match withPersona "nonexistent-persona" "Task" None None with
    | Result.Ok _ -> Assert.Fail("Expected Error for unknown persona")
    | Result.Error e -> Assert.Contains("not found", e)

// ============================================================================
// Built-in Persona Tests
// ============================================================================

[<Fact>]
let ``Built-in code-reviewer has security focus`` () =
    let reviewer = BuiltIn.codeReviewer
    Assert.Equal("code-reviewer", reviewer.Id)
    Assert.True(reviewer.Constraints |> List.exists (fun c -> c.ToLower().Contains("security")))

[<Fact>]
let ``Built-in personas have unique IDs`` () =
    let ids = BuiltIn.all |> List.map (fun p -> p.Id)
    let unique = ids |> List.distinct
    Assert.Equal(ids.Length, unique.Length)
