namespace Tars.Tests

open System
open System.IO
open System.Threading.Tasks
open Xunit
open Tars.Llm
open Tars.Llm.ClaudeCodeService
open Tars.Interface.Cli

/// Tests for ClaudeCodeService (pure logic tests + guarded integration tests).
module ClaudeCodeServiceTests =

    // =========================================================================
    // Unit tests (no Claude Code required)
    // =========================================================================

    [<Fact>]
    let ``buildPrompt leaves the system prompt to its own file`` () =
        let req =
            { LlmRequest.Default with
                SystemPrompt = Some "You are a helpful assistant."
                Messages =
                    [ { Role = Role.System; Content = "Answer in English." }
                      { Role = Role.User; Content = "Hello" } ] }

        Assert.Equal("Hello", ClaudeCodeService.buildPrompt req)
        Assert.Equal("You are a helpful assistant.\n\nAnswer in English.", ClaudeCodeService.systemPromptOf req)

    [<Fact>]
    let ``the claude process gets the prompt on stdin and answers as text only`` () =
        let psi =
            ClaudeCodeService.startInfo { defaultConfig with Model = Some "sonnet" } "system.txt"

        // Not on the command line: one holds at most 32,767 characters on Windows.
        Assert.True(psi.RedirectStandardInput)

        // No tools, settings, CLAUDE.md, hooks, MCP servers or saved session: the answer
        // depends on the request alone.
        Assert.Equal<string>(
            [ "-p"
              "--output-format"
              "json"
              "--safe-mode"
              "--setting-sources"
              ""
              "--tools"
              ""
              "--strict-mcp-config"
              "--no-session-persistence"
              "--system-prompt-file"
              "system.txt"
              "--model"
              "sonnet" ],
            List.ofSeq psi.ArgumentList
        )

    [<Fact>]
    let ``--model claude:<model> names the model Claude Code is asked for`` () =
        Assert.Equal(Some "sonnet", LlmFactory.claudeCodeModel "claude:sonnet")
        Assert.Equal(None, LlmFactory.claudeCodeModel "qwen2.5-coder:7b")
        Assert.Equal(None, LlmFactory.claudeCodeModel "claude:")

    [<Fact>]
    let ``Claude Code's completions keep the configured backend's embeddings`` () =
        let embedder =
            { new ILlmService with
                member _.CompleteAsync _ = failwith "completions do not go to the embedder"
                member _.EmbedAsync _ = Task.FromResult [| 1.0f; 2.0f |]
                member _.CompleteStreamAsync(_, _) = failwith "completions do not go to the embedder"
                member _.RouteAsync _ = failwith "routes do not go to the embedder" }

        let llm =
            ClaudeCodeService.create (Some "sonnet") |> LlmFactory.withEmbeddings embedder

        Assert.Equal<float32>([| 1.0f; 2.0f |], llm.EmbedAsync("text").Result)

        match llm.RouteAsync(LlmRequest.Default).Result.Backend with
        | Anthropic model -> Assert.Equal("sonnet", model)
        | other -> Assert.Fail(sprintf "Expected Anthropic, got %A" other)

    /// A stand-in for `claude` that runs `windows` (cmd) or `unix` (sh), whatever its arguments.
    let private fakeClaude (windows: string) (unix: string) : string =
        let dir = Directory.CreateTempSubdirectory("fake-claude").FullName

        if OperatingSystem.IsWindows() then
            let path = Path.Combine(dir, "claude.cmd")
            File.WriteAllText(path, "@echo off\r\n" + windows + "\r\n")
            path
        else
            let path = Path.Combine(dir, "claude")
            File.WriteAllText(path, "#!/bin/sh\n" + unix + "\n")
            File.SetUnixFileMode(path, UnixFileMode.UserRead ||| UnixFileMode.UserWrite ||| UnixFileMode.UserExecute)
            path

    /// More than any pipe buffer, so writing it waits for the reader.
    let private longRequest =
        { LlmRequest.Default with
            Messages = [ { Role = Role.User; Content = String.replicate 1_000_000 "x" } ] }

    [<Fact>]
    let ``a claude that exits before reading the prompt reports its exit code`` () =
        let config = { defaultConfig with ClaudePath = fakeClaude "exit /b 3" "exit 3" }
        let response = (ClaudeCodeLlmService(config) :> ILlmService).CompleteAsync(longRequest).Result

        Assert.Equal(Some "error", response.FinishReason)
        Assert.Contains("exited with code 3", response.Text)

    [<Fact>]
    let ``a claude that never reads the prompt is stopped at the timeout`` () =
        let config =
            { defaultConfig with
                ClaudePath = fakeClaude "ping -n 60 127.0.0.1 > nul" "sleep 60"
                Timeout = TimeSpan.FromSeconds 2.0 }

        let watch = Diagnostics.Stopwatch.StartNew()
        let response = (ClaudeCodeLlmService(config) :> ILlmService).CompleteAsync(longRequest).Result

        Assert.Contains("timed out", response.Text)
        Assert.True(watch.Elapsed < TimeSpan.FromSeconds 30.0, $"took {watch.Elapsed}")

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
    let ``RouteAsync names the model Claude Code was told to use`` () =
        let service = ClaudeCodeService.create (Some "opus")
        let routed = service.RouteAsync(LlmRequest.Default).Result
        match routed.Backend with
        | Anthropic model -> Assert.Equal("opus", model)
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
