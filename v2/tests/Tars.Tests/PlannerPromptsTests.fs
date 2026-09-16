module Tars.Tests.PlannerPromptsTests

open System
open System.IO
open Xunit
open Tars.Cortex

// Issue #251: a missing template silently fell back to a placeholder prompt
// ("... (truncated for brevity in actual replacement)") that was sent to the LLM.

let private withTemplate (content: string) (f: string -> unit) =
    let path = Path.Combine(Path.GetTempPath(), $"tars-planner-{Guid.NewGuid():N}.md")
    File.WriteAllText(path, content)

    try
        f path
    finally
        File.Delete path

let private overlay (path: string) =
    Some(Map.ofList [ "prompts/planner.md", path ])

[<Fact>]
let ``Planner prompt substitutes goal and date from the template`` () =
    withTemplate "Plan {goal} on {currentDate}." (fun path ->
        match PlannerPrompts.generatePlanPrompt "ship it" (overlay path) with
        | Ok prompt -> Assert.Equal("Plan ship it on " + DateTime.Now.ToString("yyyy-MM-dd") + ".", prompt)
        | Error e -> failwithf "Expected a prompt, got error: %s" e)

[<Fact>]
let ``Planner prompt fails loudly naming the path when the template is missing`` () =
    let missing =
        Path.Combine(Path.GetTempPath(), $"tars-planner-missing-{Guid.NewGuid():N}.md")

    match PlannerPrompts.generatePlanPrompt "ship it" (overlay missing) with
    | Error e -> Assert.Contains(missing, e)
    | Ok prompt -> failwithf "Expected an error, got a prompt: %s" prompt

