module Tars.Tests.RoutingClassifierTests

open Xunit
open Tars.Llm.Routing

// The de-stringly-typed routing classifiers are now pure functions over a closed
// DU — directly testable without constructing a service or hitting a backend.

[<Theory>]
[<InlineData("gpt-4o", "OpenAIFamily")>]
[<InlineData("GPT-3.5-turbo", "OpenAIFamily")>]
[<InlineData("claude-opus-4-8", "AnthropicFamily")>]
[<InlineData("gemini-2.0-flash", "GeminiFamily")>]
[<InlineData("llama3.2", "LocalFamily")>]
[<InlineData("qwen2.5-coder", "LocalFamily")>]
[<InlineData("", "LocalFamily")>]
let ``ModelFamily.classify maps model names to provider families`` (model: string) (expected: string) =
    let actual =
        match ModelFamily.classify model with
        | OpenAIFamily -> "OpenAIFamily"
        | AnthropicFamily -> "AnthropicFamily"
        | GeminiFamily -> "GeminiFamily"
        | LocalFamily -> "LocalFamily"

    Assert.Equal(expected, actual)

[<Theory>]
[<InlineData("code", "CodeHint")>]
[<InlineData("cheap", "CheapHint")>]
[<InlineData("reasoning", "ReasoningHint")>]
[<InlineData("analysis", "ReasoningHint")>]
[<InlineData("think-hard", "ReasoningHint")>]
[<InlineData("docker", "DockerHint")>]
[<InlineData("llamacpp", "LlamaCppHint")>]
[<InlineData("perf", "LlamaCppHint")>]
[<InlineData("fast", "FastHint")>]
[<InlineData("quick", "FastHint")>]
[<InlineData("", "DefaultHint")>]
[<InlineData("whatever", "DefaultHint")>]
let ``RoutingHint.classify maps hints to routing intents`` (hint: string) (expected: string) =
    let actual =
        match RoutingHint.classify hint with
        | CodeHint -> "CodeHint"
        | CheapHint -> "CheapHint"
        | ReasoningHint -> "ReasoningHint"
        | DockerHint -> "DockerHint"
        | LlamaCppHint -> "LlamaCppHint"
        | FastHint -> "FastHint"
        | DefaultHint -> "DefaultHint"

    Assert.Equal(expected, actual)

[<Fact>]
let ``RoutingHint.classify preserves original precedence: code before reasoning`` () =
    // A hint containing both "code" and "reason" must classify as CodeHint,
    // matching the original sequential matching order.
    Assert.Equal(CodeHint, RoutingHint.classify "code-reasoning")
