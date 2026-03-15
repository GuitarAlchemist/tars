module Tars.Tests.AgentDefinitionTests

open Xunit
open Tars.Core

[<Fact>]
let ``Parse valid agent definition from markdown`` () =
    let md = """---
id: test-agent
name: Test Agent
role: Tester
description: A test agent for unit tests
model_hint: reasoning
temperature: 0.3
capabilities: [reasoning, planning]
version: "1.0"
---

You are a test agent. Be precise and helpful.
"""
    let result = AgentDefinitionParser.parse md None
    match result with
    | Result.Ok def ->
        Assert.Equal("test-agent", def.Id)
        Assert.Equal("Test Agent", def.Name)
        Assert.Equal("Tester", def.Role)
        Assert.Equal("A test agent for unit tests", def.Description)
        Assert.Equal(Some "reasoning", def.ModelHint)
        Assert.Equal(Some 0.3, def.Temperature)
        Assert.Equal(2, def.Capabilities.Length)
        Assert.Contains(AgentSkill.Reasoning, def.Capabilities)
        Assert.Contains(AgentSkill.Planning, def.Capabilities)
        Assert.Equal(Some "1.0", def.Version)
        Assert.Contains("test agent", def.SystemPrompt)
    | Result.Error e -> Assert.Fail($"Parse failed: {e}")

[<Fact>]
let ``Parse agent definition with multiline capabilities`` () =
    let md = """---
id: multi-cap
name: Multi
role: Multi
description: Agent with multiline caps
capabilities:
- reasoning
- coding
- critique
---

System prompt here.
"""
    let result = AgentDefinitionParser.parse md None
    match result with
    | Result.Ok def ->
        Assert.Equal(3, def.Capabilities.Length)
        Assert.Contains(AgentSkill.Reasoning, def.Capabilities)
        Assert.Contains(AgentSkill.Coding, def.Capabilities)
        Assert.Contains(AgentSkill.Critique, def.Capabilities)
    | Result.Error e -> Assert.Fail($"Parse failed: {e}")

[<Fact>]
let ``Parse agent definition with custom capability`` () =
    let md = """---
id: custom
name: Custom
role: Custom
description: Custom caps
capabilities: [music-theory, tab-generation]
---

Prompt.
"""
    let result = AgentDefinitionParser.parse md None
    match result with
    | Result.Ok def ->
        Assert.Equal(2, def.Capabilities.Length)
        match def.Capabilities.[0] with
        | AgentSkill.Custom "music-theory" -> ()
        | other -> Assert.Fail($"Expected Custom 'music-theory', got {other}")
    | Result.Error e -> Assert.Fail($"Parse failed: {e}")

[<Fact>]
let ``Parse fails without frontmatter`` () =
    let md = "Just a plain markdown file."
    let result = AgentDefinitionParser.parse md None
    match result with
    | Result.Error _ -> ()
    | Result.Ok _ -> Assert.Fail("Expected parse error for missing frontmatter")

[<Fact>]
let ``Parse fails with missing required fields`` () =
    let md = """---
name: NoId
role: NoId
---

Prompt.
"""
    let result = AgentDefinitionParser.parse md None
    match result with
    | Result.Error e -> Assert.Contains("id", e.ToLowerInvariant())
    | Result.Ok _ -> Assert.Fail("Expected parse error for missing id")

[<Fact>]
let ``Parse handles optional fields gracefully`` () =
    let md = """---
id: minimal
name: Minimal
role: Minimal
---

Minimal prompt.
"""
    let result = AgentDefinitionParser.parse md None
    match result with
    | Result.Ok def ->
        Assert.Equal(None, def.ModelHint)
        Assert.Equal(None, def.Temperature)
        Assert.Equal(None, def.Version)
        Assert.Empty(def.Capabilities)
        Assert.Equal("", def.Description)
    | Result.Error e -> Assert.Fail($"Parse failed: {e}")

[<Fact>]
let ``Parse handles quoted values`` () =
    let md = """---
id: "quoted-id"
name: 'Quoted Name'
role: QuotedRole
description: "Description with spaces"
---

Prompt.
"""
    let result = AgentDefinitionParser.parse md None
    match result with
    | Result.Ok def ->
        Assert.Equal("quoted-id", def.Id)
        Assert.Equal("Quoted Name", def.Name)
    | Result.Error e -> Assert.Fail($"Parse failed: {e}")

[<Fact>]
let ``System prompt preserves multiline content`` () =
    let md = """---
id: multi
name: Multi
role: Multi
---

Line one.

Line two.

Line three.
"""
    let result = AgentDefinitionParser.parse md None
    match result with
    | Result.Ok def ->
        Assert.Contains("Line one.", def.SystemPrompt)
        Assert.Contains("Line two.", def.SystemPrompt)
        Assert.Contains("Line three.", def.SystemPrompt)
    | Result.Error e -> Assert.Fail($"Parse failed: {e}")

[<Fact>]
let ``AgentRegistry has all 8 builtin agents`` () =
    let agents = Tars.Core.WorkflowOfThought.AgentRegistry.allDefinitions ()
    Assert.True(agents.Length >= 8, $"Expected at least 8 agents, got {agents.Length}")

    let roles = agents |> List.map (fun a -> a.Role)
    Assert.Contains("Planner", roles)
    Assert.Contains("QAEngineer", roles)
    Assert.Contains("Skeptic", roles)
    Assert.Contains("Verifier", roles)
    Assert.Contains("Comms", roles)
    Assert.Contains("Regulatory", roles)
    Assert.Contains("SupplyChain", roles)
    Assert.Contains("Default", roles)

[<Fact>]
let ``AgentRegistry get returns correct agent`` () =
    let planner = Tars.Core.WorkflowOfThought.AgentRegistry.get "Planner"
    Assert.True(planner.IsSome)
    Assert.Equal("Planner", planner.Value.Role)
    Assert.Equal(Some "reasoning", planner.Value.ModelHint)

[<Fact>]
let ``AgentRegistry get is case-insensitive`` () =
    let upper = Tars.Core.WorkflowOfThought.AgentRegistry.get "PLANNER"
    let lower = Tars.Core.WorkflowOfThought.AgentRegistry.get "planner"
    Assert.True(upper.IsSome)
    Assert.True(lower.IsSome)
    Assert.Equal(upper.Value.Role, lower.Value.Role)

[<Fact>]
let ``AgentRegistry getOrDefault falls back to Default`` () =
    let unknown = Tars.Core.WorkflowOfThought.AgentRegistry.getOrDefault "nonexistent"
    Assert.Equal("Default", unknown.Role)

[<Fact>]
let ``Parse accepts GA-compatible fields without error`` () =
    let md = """---
id: ga-theory
name: Theory Agent
role: theory
description: Music theory analysis
capabilities: [reasoning, analysis]
routing_keywords: [chord, scale, key]
use_critique: true
delegates_to: critic
---

You are Theory Agent.
"""
    let result = AgentDefinitionParser.parse md None
    match result with
    | Result.Ok def ->
        Assert.Equal("ga-theory", def.Id)
        Assert.Equal("theory", def.Role)
    | Result.Error e -> Assert.Fail($"Parse failed: {e}")

[<Fact>]
let ``Discovery searches cross-repo paths`` () =
    let paths = AgentDefinitionDiscovery.defaultSearchPaths "/tmp/test"
    Assert.Equal(3, paths.Length)
    Assert.Contains("agents", paths.[0])
    Assert.Contains(".tars", paths.[1])
    Assert.Contains(".ga", paths.[2])
