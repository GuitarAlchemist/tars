# TARS V2 Testing Strategy

**Date:** November 22, 2025
**Status:** Design
**Priority:** P0 (CRITICAL)
**Context:** Defining how to test an intelligent, non-deterministic system.

---

## The Testing Challenge

**Traditional software:** Given input X, always produces output Y.
**Agentic software:** Given task X, produces *one of many valid solutions* Y₁, Y₂, ... Yₙ.

**Example:**

```
Task: "Fix the failing test"
Valid outcomes:
✓ Agent fixes the bug
✓ Agent modifies the test (if test was wrong)
✓ Agent adds better error handling
✗ Agent does nothing (FAIL)
✗ Agent breaks more tests (FAIL)
```

---

## Testing Pyramid for TARS

```
         Rare, Expensive
              ▲
              │
    ┌─────────────────────┐
    │  Scenario Tests     │  ← 5 tests (weekly)
    │  (End-to-End)       │
    └─────────────────────┘
              ▲
              │
    ┌─────────────────────┐
    │  Agent Behavior     │  ← 20 tests (daily)
    │  Tests              │
    └─────────────────────┘
              ▲
              │
    ┌─────────────────────┐
    │  Integration Tests  │  ← 50 tests (CI)
    └─────────────────────┘
              ▲
              │
    ┌─────────────────────┐
    │  Unit Tests         │  ← 200+ tests (CI)
    │  (Pure Functions)   │
    └─────────────────────┘
              │
              ▼
         Fast, Abundant
```

---

## Layer 1: Unit Tests (Pure F# Functions)

**What to test:** Deterministic functions with no side effects.

```fsharp
// Example: BeliefGraph
[<Fact>]
let ``addNode should add node to graph`` () =
    let graph = BeliefGraph.empty
    let newGraph = graph |> BeliefGraph.addNode "concept1" "AI"
    
    Assert.True(BeliefGraph.contains "concept1" newGraph)
    Assert.Equal("AI", BeliefGraph.getData "concept1" newGraph)

// Example: VectorStore
[<Property>]
let ``cosine similarity is symmetric`` (v1: float[]) (v2: float[]) =
    let sim1 = VectorStore.cosineSimilarity v1 v2
    let sim2 = VectorStore.cosineSimilarity v2 v1
    abs(sim1 - sim2) < 0.0001
```

**Tools:**

- **xUnit**: Standard .NET test framework
- **FsCheck**: Property-based testing (generates random inputs)
- **Coverage Target:** 80%+ for core libraries

---

## Layer 2: Integration Tests (Component Interaction)

**What to test:** MCP clients, AutoGen bridges, Docker sandboxes.

### Test 1: MCP Client → MCP Server

```fsharp
[<Fact>]
let ``McpClient can call Filesystem server`` () = async {
    // Start test MCP server (Docker)
    use! server = TestContainers.startMcpServer "filesystem"
    
    // Create MCP client
    let client = McpClient(server.Endpoint)
    
    // Call "read_file" tool
    let! result = client.CallTool("read_file", {| path = "/test.txt" |})
    
    Assert.Equal("Hello, TARS!", result.content)
}
```

### Test 2: AutoGen → F# Kernel Message Passing

```fsharp
[<Fact>]
let ``AutoGen agent can send message to F# Kernel`` () = async {
    let kernel = TarsKernel()
    let agent = AutoGenBridge(kernel.EventBus)
    
    // Agent sends "Hello"
    do! agent.SendMessage({ Role = "user"; Content = "Hello" })
    
    // Kernel should receive it
    let! receivedMessages = kernel.GetMessages()
    Assert.Single(receivedMessages)
}
```

**Tools:**

- **TestContainers**: Spin up Docker containers for tests
- **Docker Compose**: Multi-service integration tests

---

## Layer 3: Agent Behavior Tests (The Novel Part)

**What to test:** Agent decision-making and tool usage.

### Pattern: "Golden Runs"

**Concept:** Record a successful agent trace, then replay it as a regression test.

#### Step 1: Capture Golden Run

```bash
$ tars fix-build --record golden-run-001.json

┌─────────────────────────────────────────────┐
│ Recording agent trace to golden-run-001.json│
├─────────────────────────────────────────────┤
│ ✓ Ran test suite                            │
│ ✓ Identified failing test                   │
│ ✓ Read UserService.cs                       │
│ ✓ Applied fix                                │
│ ✓ Tests now pass                             │
└─────────────────────────────────────────────┘

Trace saved: 12 steps, 45 seconds
```

#### Step 2: Convert to Test

```fsharp
[<Fact>]
let ``Agent fixes NullReferenceException (Golden Run 001)`` () = async {
    // Load the recorded trace
    let! trace = GoldenRun.load "golden-run-001.json"
    
    // Replay the trace
    let! actualSteps = Agent.replay trace
    
    // Assert: Agent took same actions
    Assert.Equal(trace.Steps.Length, actualSteps.Length)
    Assert.All(actualSteps, fun step ->
        Assert.Contains(step.Action, ["RunTests"; "ReadFile"; "WriteFile"])
    )
    
    // Assert: Final outcome matches
    Assert.True(trace.FinalState.TestsPassing)
}
```

**Golden Run Schema:**

```json
{
  "runId": "golden-run-001",
  "task": "Fix failing test in UserService",
  "steps": [
    {"action": "RunTests", "output": "1 test failed"},
    {"action": "ReadFile", "args": {"path": "UserService.cs"}},
    {"action": "WriteFile", "args": {"path": "UserService.cs", "content": "..."}},
    {"action": "RunTests", "output": "All tests passed"}
  ],
  "finalState": {
    "testsPassing": true,
    "filesChanged": ["UserService.cs"]
  }
}
```

### Pattern: "Behavior Assertions"

**Instead of checking exact actions, check high-level behaviors.**

```fsharp
[<Fact>]
let ``Agent should run tests before proposing fix`` () = async {
    let task = { Goal = "Fix failing test"; Repo = "/test-repo" }
    let! agentSteps = Agent.execute task
    
    // Assert: First action should be "RunTests"
    Assert.Equal("RunTests", agentSteps.[0].Action)
    
    // Assert: Agent should not modify code before seeing error
    let firstCodeChange = agentSteps |> List.tryFindIndex (fun s -> s.Action = "WriteFile")
    let testsRun = agentSteps |> List.tryFindIndex (fun s -> s.Action = "RunTests")
    Assert.True(testsRun < firstCodeChange)
}
```

---

## Layer 4: Scenario Tests (End-to-End)

**What to test:** Full workflows from user request → agent completion.

### Scenario 1: "Generate REST API from OpenAPI Spec"

```gherkin
Feature: Generate REST API
  As a developer
  I want TARS to generate a working REST API
  So that I can save time

Scenario: Generate API from spec
  Given an OpenAPI spec file "petstore.yaml"
  When I run "tars generate api --spec petstore.yaml"
  Then TARS should:
    | Step | Expected |
    | Parse spec | ✓ |
    | Generate ASP.NET controllers | ✓ |
    | Generate models | ✓ |
    | Run `dotnet build` | ✓ |
    | Create Postman collection | ✓ |
  And the API should respond to GET /pets with 200 OK
```

**Implementation (SpecFlow):**

```csharp
[Given(@"an OpenAPI spec file ""(.*)""")]
public void GivenAnOpenAPISpec(string filename) {
    File.Copy($"testdata/{filename}", "/workspace/spec.yaml");
}

[When(@"I run ""(.*)""")]
public async Task WhenIRunCommand(string command) {
    _result = await TarsCli.Run(command);
}

[Then(@"the API should respond to GET (.*) with (\d+) OK")]
public async Task ThenAPIResponds(string endpoint, int statusCode) {
    var response = await Http.Get($"http://localhost:5000{endpoint}");
    Assert.Equal(statusCode, (int)response.StatusCode);
}
```

---

## Test Data Strategy

### Synthetic Repos

Create minimal, controlled repos for testing.

```
test-repos/
  failing-test/
    UserService.cs     # Has intentional bug
    UserServiceTests.cs # Failing test
  broken-build/
    Program.cs         # Missing semicolon
  complex-refactor/
    ... # 50 files, realistic codebase
```

### Mocked LLM Responses

For deterministic tests, mock the LLM.

```fsharp
type MockLLM() =
    interface ICognitiveProvider with
        member _.Complete(prompt) = async {
            if prompt.Contains("fix the null reference") then
                return "Add null check: if (user == null) return null;"
            else
                return "I don't know."
        }
```

---

## Metrics & Success Criteria

| Metric | Target |
| :--- | :--- |
| **Unit Test Coverage** | 80%+ |
| **Integration Test Pass Rate** | 95%+ |
| **Golden Run Regression Rate** | 0% (no regressions) |
| **Scenario Test Success** | 90%+ (allows for LLM variability) |

### Dashboard

```
┌───────────────────────────────────────┐
│ TARS Test Suite Status                │
├───────────────────────────────────────┤
│ Unit Tests:        312 ✓  2 ✗  98%   │
│ Integration:        48 ✓  1 ✗  97%   │
│ Agent Behavior:     18 ✓  0 ✗ 100%   │
│ Scenario:            4 ✓  1 ✗  80%   │
├───────────────────────────────────────┤
│ Overall: PASS ✓                       │
└───────────────────────────────────────┘
```

---

## CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: TARS Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: dotnet test --filter Category=Unit
  
  integration-tests:
    runs-on: ubuntu-latest
    services:
      docker:
        image: docker:dind
    steps:
      - run: dotnet test --filter Category=Integration
  
  golden-runs:
    runs-on: ubuntu-latest
    steps:
      - run: dotnet test --filter Category=GoldenRun
  
  scenario-tests:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'  # Only on main branch
    steps:
      - run: dotnet test --filter Category=Scenario
```

---

## Implementation Checklist

- [ ] **Week 1**: Set up xUnit + FsCheck for unit tests
- [ ] **Week 2**: Add TestContainers for integration tests
- [ ] **Week 3**: Implement Golden Run recorder + replayer
- [ ] **Week 4**: Create 5 golden run test cases
- [ ] **Week 5**: Build 3 scenario tests with SpecFlow

---

## References

- [Golden Testing (Approval Tests)](https://approvaltests.com/)
- [TestContainers](https://testcontainers.com/)
- [SpecFlow (BDD)](https://specflow.org/)
- [FsCheck (Property-Based Testing)](https://fscheck.github.io/FsCheck/)
