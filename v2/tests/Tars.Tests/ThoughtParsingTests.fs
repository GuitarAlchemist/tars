module Tars.Tests.ThoughtParsingTests

open System.Text.Json
open Xunit
open Tars.Cortex.ThoughtParsing

// Issue #280: these helpers were private to Patterns.fs, so a bug in one (#263) could only
// be reached by running a whole Graph/Tree/Workflow-of-Thought workflow.

[<Fact>]
let ``parseThoughts splits on emphasised headers and skips the preamble`` () =
    let reply =
        """Here are three distinct thoughts about the problem:

**THOUGHT 1:** Cache the parsed grammar between requests to avoid reparsing.
**THOUGHT 2:** Stream results to the client instead of buffering everything.
## THOUGHT 3: Batch the embedding calls so the model is hit once per step."""

    Assert.Equal<string>(
        [ "Cache the parsed grammar between requests to avoid reparsing."
          "Stream results to the client instead of buffering everything."
          "Batch the embedding calls so the model is hit once per step." ],
        parseThoughts reply
    )

[<Fact>]
let ``parseThoughts joins continuation lines into the current thought`` () =
    let reply =
        """THOUGHT 1: Split the module by pattern
so each one gets its own test file."""

    Assert.Equal<string>(
        [ "Split the module by pattern so each one gets its own test file." ],
        parseThoughts reply
    )

[<Fact>]
let ``parseThoughts without headers keeps long lines and drops short fragments`` () =
    let reply =
        """ok
This line is long enough to count as a thought on its own."""

    Assert.Equal<string>([ "This line is long enough to count as a thought on its own." ], parseThoughts reply)

[<Fact>]
let ``stripCodeFence removes a fenced block with a language tag`` () =
    Assert.Equal("{\"score\": 0.8}", stripCodeFence "```json\n{\"score\": 0.8}\n```")
    Assert.Equal("plain", stripCodeFence "  plain  ")

[<Fact>]
let ``tryParseJsonWithFallback recovers JSON wrapped in prose`` () =
    match tryParseJsonWithFallback "Sure! Here is the score: {\"score\": 0.7} Hope that helps." with
    | Ok elem -> Assert.Equal(0.7, elem.GetProperty("score").GetDouble())
    | Error e -> failwithf "Expected JSON, got error: %s" e

[<Fact>]
let ``tryParseJsonWithFallback rejects an empty reply`` () =
    Assert.True(Result.isError (tryParseJsonWithFallback "```\n```"))

[<Fact>]
let ``getDoubleWithDefault tries names in order, case-insensitively, and accepts numeric strings`` () =
    use doc = JsonDocument.Parse("""{"Confidence": "0.35", "other": true}""")
    let elem = doc.RootElement

    Assert.Equal(0.35, getDoubleWithDefault [ "score"; "confidence" ] 0.5 elem)
    Assert.Equal(0.5, getDoubleWithDefault [ "other"; "missing" ] 0.5 elem)

[<Fact>]
let ``getStringList accepts an array or a single string`` () =
    use doc =
        JsonDocument.Parse("""{"risks": ["slow", 3, "costly"], "reason": "fits"}""")

    let elem = doc.RootElement

    Assert.Equal<string>([ "slow"; "costly" ], getStringList "RISKS" elem)
    Assert.Equal<string>([ "fits" ], getStringList "reason" elem)
    Assert.Equal<string>([], getStringList "missing" elem)

[<Fact>]
let ``clamp01 bounds a score to the unit interval`` () =
    Assert.Equal(0.0, clamp01 -0.2)
    Assert.Equal(1.0, clamp01 1.7)
    Assert.Equal(0.4, clamp01 0.4)
