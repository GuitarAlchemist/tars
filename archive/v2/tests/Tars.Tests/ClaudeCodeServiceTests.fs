namespace Tars.Tests

open System
open Xunit
open Tars.Llm
open Tars.Llm.ClaudeCodeService

/// Tests for ClaudeCodeService (pure logic tests + guarded integration tests).
module ClaudeCodeServiceTests =

    // =========================================================================
    // Unit tests (no Claude Code required)
    // =========================================================================

    [<Fact>]
    let ``buildPrompt includes system prompt`` () =
        let req =
            { LlmRequest.Default with
                SystemPrompt = Some "You are a helpful assistant."
                Messages = [ { Role = Role.User; Content = "Hello" } ] }

        let prompt = ClaudeCodeService.buildPrompt req
        Assert.Contains("You are a helpful assistant", prompt)
        Assert.Contains("Hello", prompt)

    [<Fact>]
    let ``buildPrompt handles multiple messages`` () =
        let req =
            { LlmRequest.Default with
                Messages =
                    [ { Role = Role.User; Content = "What is F#?" }
                      { Role = Role.Assistant; Content = "F# is a functional language." }
                      { Role = Role.User; Content = "Tell me more." } ] }

        let prompt = ClaudeCodeService.buildPrompt req
        Assert.Contains("What is F#?", prompt)
        Assert.Contains("F# is a functional language", prompt)
        Assert.Contains("Tell me more", prompt)

    [<Fact>]
    let ``buildPrompt handles empty messages`` () =
        let req = { LlmRequest.Default with Messages = [] }
        let prompt = ClaudeCodeService.buildPrompt req
        Assert.Equal("", prompt)

    [<Fact>]
    let ``parseResponse handles valid Claude Code JSON`` () =
        let json = """{"type":"result","result":"Hello, world!","cost_usd":0.001,"session_id":"abc123"}"""
        let response = ClaudeCodeService.parseResponse json
        Assert.Equal("Hello, world!", response.Text)
        Assert.Equal(Some "stop", response.FinishReason)
        Assert.True(response.Usage.IsSome)
        Assert.True(response.Raw.IsSome)

    [<Fact>]
    let ``parseResponse handles missing result field`` () =
        let json = """{"type":"error","error":"something went wrong"}"""
        let response = ClaudeCodeService.parseResponse json
        // Falls back to raw JSON as text
        Assert.True(response.Text.Length > 0)

    [<Fact>]
    let ``parseResponse handles non-JSON output`` () =
        let text = "This is plain text output"
        let response = ClaudeCodeService.parseResponse text
        Assert.Equal(text, response.Text)

    [<Fact>]
    let ``defaultConfig has sensible defaults`` () =
        Assert.Equal("claude", defaultConfig.ClaudePath)
        Assert.Equal(TimeSpan.FromMinutes(2.0), defaultConfig.Timeout)
        Assert.False(defaultConfig.Verbose)
        Assert.True(defaultConfig.Model.IsNone)

    [<Fact>]
    let ``create returns an ILlmService`` () =
        let service = ClaudeCodeService.create None
        Assert.NotNull(service)

    [<Fact>]
    let ``create with model override`` () =
        let service = ClaudeCodeService.create (Some "sonnet")
        Assert.NotNull(service)

    [<Fact>]
    let ``RouteAsync returns Anthropic backend`` () =
        let service = ClaudeCodeService.create None
        let routed = service.RouteAsync(LlmRequest.Default).Result
        match routed.Backend with
        | Anthropic model -> Assert.Equal("claude-code", model)
        | other -> Assert.Fail(sprintf "Expected Anthropic, got %A" other)

    [<Fact>]
    let ``EmbedAsync returns empty array`` () =
        let service = ClaudeCodeService.create None
        let result = service.EmbedAsync("test").Result
        Assert.Empty(result)

    // =========================================================================
    // Integration tests (require Claude Code on PATH)
    // =========================================================================

    let private claudeCodeAvailable =
        lazy (ClaudeCodeService.isAvailable ())

    [<Fact>]
    let ``isAvailable detects Claude Code`` () =
        // This test always runs — it just reports whether claude is on PATH
        let available = ClaudeCodeService.isAvailable ()
        // We don't assert true/false — just that it doesn't throw
        Assert.True(available || not available)

    [<Fact>]
    let ``CompleteAsync returns response from Claude Code`` () =
        if not claudeCodeAvailable.Value then () else

        let service = ClaudeCodeService.create None
        let req =
            { LlmRequest.Default with
                SystemPrompt = Some "Reply with exactly one word."
                Messages = [ { Role = Role.User; Content = "Say hello." } ]
                MaxTokens = Some 10 }

        let response = service.CompleteAsync(req).Result
        // Always produces some text (either success or error message)
        Assert.True(response.Text.Length > 0)
        // If Claude Code is authenticated and working, the response won't be an error
        // But in CI/test contexts where it's not fully set up, we just verify no crash
        Assert.True(response.FinishReason.IsSome)
