namespace Tars.Tools.Standard

open System
open Tars.Tools

module AgentTools =

    /// Agent registry reference (set at startup)
    let mutable private agentRegistry: obj option = None

    /// Set the agent registry reference
    let setRegistry (registry: obj) = agentRegistry <- Some registry

    [<TarsToolAttribute("delegate_task",
                        "Delegates a task to another agent. Input JSON: { \"agent\": \"Reviewer\", \"task\": \"Review this code for bugs\", \"context\": \"optional context\" }")>]
    let delegateTask (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let agent = root.GetProperty("agent").GetString()
                let taskDesc = root.GetProperty("task").GetString()

                let mutable contextProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let context =
                    if root.TryGetProperty("context", &contextProp) then
                        contextProp.GetString()
                    else
                        ""

                printfn "📤 DELEGATING to %s: %s" agent (taskDesc.Substring(0, min 50 taskDesc.Length))

                // For now, return a formatted delegation request
                // In a full implementation, this would actually invoke the target agent
                let delegation =
                    "Task Delegation Request:\n"
                    + "  Target Agent: "
                    + agent
                    + "\n"
                    + "  Task: "
                    + taskDesc
                    + "\n"
                    + (if context.Length > 0 then
                           "  Context: " + context + "\n"
                       else
                           "")
                    + "\nNote: Full agent-to-agent execution not yet implemented. "
                    + "The target agent would receive this task and execute it independently."

                return delegation
            with ex ->
                return "delegate_task error: " + ex.Message
        }

    [<TarsToolAttribute("request_review",
                        "Requests code review from the Reviewer agent. Input JSON: { \"code\": \"code to review\", \"focus\": \"bugs|style|performance\" }")>]
    let requestReview (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let code = root.GetProperty("code").GetString()

                let mutable focusProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let focus =
                    if root.TryGetProperty("focus", &focusProp) then
                        focusProp.GetString()
                    else
                        "general"

                printfn "🔍 REQUESTING REVIEW (focus: %s)" focus

                let codePreview =
                    if code.Length > 100 then
                        code.Substring(0, 100) + "..."
                    else
                        code

                // Simulate a review response
                let review =
                    "Code Review Request:\n"
                    + "  Focus: "
                    + focus
                    + "\n"
                    + "  Code preview: "
                    + codePreview
                    + "\n\n"
                    + "Review Checklist:\n"
                    + "  [ ] Check for bugs and edge cases\n"
                    + "  [ ] Verify error handling\n"
                    + "  [ ] Review code style\n"
                    + "  [ ] Check performance implications\n"
                    + "  [ ] Ensure tests are adequate\n\n"
                    + "Use analyze_code or run_tests for deeper analysis."

                return review
            with ex ->
                return "request_review error: " + ex.Message
        }

    [<TarsToolAttribute("query_agent",
                        "Queries another agent for information. Input JSON: { \"agent\": \"Curriculum\", \"question\": \"What tasks are pending?\" }")>]
    let queryAgent (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let agent = root.GetProperty("agent").GetString()
                let question = root.GetProperty("question").GetString()

                printfn "❓ QUERYING %s: %s" agent (question.Substring(0, min 40 question.Length))

                // Provide agent-specific responses
                let response =
                    match agent.ToLower() with
                    | "curriculum" ->
                        "Curriculum Agent Status:\n"
                        + "  - Generates coding tasks for skill improvement\n"
                        + "  - Tracks task difficulty progression\n"
                        + "  - Adapts based on completion success"
                    | "reviewer" ->
                        "Reviewer Agent Status:\n"
                        + "  - Reviews code for quality and correctness\n"
                        + "  - Tools: read_code, git_diff, git_status\n"
                        + "  - Can approve or request changes"
                    | "executor" ->
                        "Executor Agent Status:\n"
                        + "  - Executes coding tasks\n"
                        + "  - Has 28+ tools available\n"
                        + "  - Can write and commit code"
                    | _ -> sprintf "Unknown agent: %s. Available agents: Curriculum, Executor, Reviewer" agent

                return sprintf "Query to %s:\n  Q: %s\n\n%s" agent question response
            with ex ->
                return "query_agent error: " + ex.Message
        }

    [<TarsToolAttribute("list_agents", "Lists all available agents and their capabilities. No input required.")>]
    let listAgents (_: string) =
        task {
            printfn "📋 LISTING AGENTS"

            let agentList =
                "Available Agents:\n\n"
                + "1. Curriculum Agent\n"
                + "   - Role: Task generation and difficulty progression\n"
                + "   - Capabilities: Planning, Reasoning\n"
                + "   - Tools: None (uses LLM directly)\n\n"
                + "2. Executor Agent\n"
                + "   - Role: Task execution and code generation\n"
                + "   - Capabilities: CodeGeneration, TaskExecution, Reasoning\n"
                + "   - Tools: 28+ (explore, read, write, test, git, etc.)\n\n"
                + "3. Reviewer Agent\n"
                + "   - Role: Code review and quality assurance\n"
                + "   - Capabilities: Reasoning, Planning\n"
                + "   - Tools: read_code, git_diff, git_status"

            return agentList
        }

    [<TarsToolAttribute("agent_status",
                        "Gets the current status of an agent. Input: agent name (Curriculum, Executor, or Reviewer)")>]
    let agentStatus (agentName: string) =
        task {
            let name =
                if String.IsNullOrWhiteSpace(agentName) then
                    "Executor"
                else
                    agentName.Trim()

            printfn "📊 STATUS: %s" name

            let status =
                match name.ToLower() with
                | "curriculum"
                | "curriculum agent" ->
                    "Curriculum Agent:\n"
                    + "  State: Active\n"
                    + "  Tasks Generated: (tracked per session)\n"
                    + "  Model: (same as Executor)"
                | "executor"
                | "executor agent" ->
                    "Executor Agent:\n"
                    + "  State: Active\n"
                    + "  Tools: 28+\n"
                    + "  Current Task: (varies)"
                | "reviewer"
                | "reviewer agent" ->
                    "Reviewer Agent:\n"
                    + "  State: Available\n"
                    + "  Tools: read_code, git_diff, git_status\n"
                    + "  Reviews Pending: 0"
                | _ -> sprintf "Unknown agent: %s. Try: Curriculum, Executor, or Reviewer" name

            return status
        }
