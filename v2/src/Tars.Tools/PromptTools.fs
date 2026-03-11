namespace Tars.Tools.Standard

open System
open System.Text.Json
open Tars.Tools

module PromptTools =

    [<TarsToolAttribute("improve_prompt",
                        "Analyzes and suggests improvements for a prompt. Input: the prompt text to improve.")>]
    let improvePrompt (prompt: string) =
        task {
            printfn $"✨ IMPROVING PROMPT ({prompt.Length} chars)"

            // Analyze prompt for common issues and suggest improvements
            let issues = ResizeArray<string>()
            let suggestions = ResizeArray<string>()

            if prompt.Length < 50 then
                issues.Add("Prompt is very short")
                suggestions.Add("Add more context and specific instructions")

            if
                not (
                    prompt.Contains("step")
                    || prompt.Contains("specific")
                    || prompt.Contains("example")
                )
            then
                suggestions.Add("Consider adding 'step by step' or specific examples")

            if not (prompt.ToLower().Contains("json") || prompt.Contains("format")) then
                suggestions.Add("Specify expected output format (JSON, code, etc.)")

            if prompt.ToUpper() = prompt then
                issues.Add("Prompt is all uppercase (may seem aggressive)")
                suggestions.Add("Use normal case for better readability")

            if prompt.Contains("don't") || prompt.Contains("never") || prompt.Contains("can't") then
                suggestions.Add("Consider rephrasing negatives as positives")

            if not (prompt.Contains("?") || prompt.Contains("Please") || prompt.Contains("should")) then
                suggestions.Add("Add clear action words or questions")

            let preview =
                if prompt.Length > 200 then
                    prompt.Substring(0, 200) + "..."
                else
                    prompt

            let issueText =
                if issues.Count > 0 then
                    "Issues Found:\n  - " + String.Join("\n  - ", issues)
                else
                    "No major issues found"

            let suggestionText = String.Join("\n  - ", suggestions)

            let analysis =
                "Prompt Analysis ("
                + string prompt.Length
                + " characters):\n\n"
                + "Original Prompt:\n"
                + preview
                + "\n\n"
                + issueText
                + "\n\n"
                + "Suggestions:\n  - "
                + suggestionText
                + "\n\n"
                + "Prompt Enhancement Tips:\n"
                + "  1. Start with clear context/role\n"
                + "  2. Be specific about desired output\n"
                + "  3. Include examples when helpful\n"
                + "  4. Specify format (JSON, code, markdown)\n"
                + "  5. End with clear action instruction"

            return analysis
        }

    [<TarsToolAttribute("create_agent_prompt",
                        "Creates a well-structured agent system prompt. Input JSON: { \"role\": \"coder\", \"style\": \"concise\" }")>]
    let createAgentPrompt (args: string) =
        task {
            try
                let doc = JsonDocument.Parse(args)
                let root = doc.RootElement

                let mutable roleProp = Unchecked.defaultof<JsonElement>

                let role =
                    if root.TryGetProperty("role", &roleProp) then
                        roleProp.GetString()
                    else
                        "assistant"

                let mutable styleProp = Unchecked.defaultof<JsonElement>

                let style =
                    if root.TryGetProperty("style", &styleProp) then
                        styleProp.GetString()
                    else
                        "helpful"

                printfn $"🤖 CREATING AGENT PROMPT for role: {role}"

                let prompt =
                    "You are an expert "
                    + role
                    + " assistant.\n\n"
                    + "## Communication Style\n"
                    + "- Be "
                    + style
                    + " and professional\n"
                    + "- Provide clear, actionable responses\n"
                    + "- Use code examples when helpful\n\n"
                    + "## Guidelines\n"
                    + "1. Think step-by-step before acting\n"
                    + "2. Use available tools effectively\n"
                    + "3. Verify your work before completing\n"
                    + "4. Ask clarifying questions if needed"

                return
                    "Generated Agent Prompt:\n\n"
                    + prompt
                    + "\n\nUse this as a system prompt for a new agent."
            with ex ->
                return $"create_agent_prompt error: {ex.Message}"
        }

    [<TarsToolAttribute("reflect_on_task",
                        "Reflects on task completion and identifies improvements. Input: description of what was done.")>]
    let reflectOnTask (description: string) =
        task {
            printfn "🔍 REFLECTING on task..."

            let reflection =
                "Task Reflection:\n\n"
                + "What was done:\n"
                + description
                + "\n\n"
                + "Questions to consider:\n"
                + "1. Did the solution fully address the requirements?\n"
                + "2. Are there edge cases not handled?\n"
                + "3. Could the code be more efficient?\n"
                + "4. Is the code readable and maintainable?\n"
                + "5. Are there tests that should be added?\n\n"
                + "Next steps:\n"
                + "- Review the solution for completeness\n"
                + "- Consider adding error handling\n"
                + "- Think about documentation"

            return reflection
        }

    [<TarsToolAttribute("report_progress", "Reports current progress on a task. Input: progress description.")>]
    let reportProgress (description: string) =
        task {
            let timestamp = DateTime.Now.ToString("HH:mm:ss")

            let preview =
                if description.Length > 80 then
                    description.Substring(0, 80)
                else
                    description

            printfn $"📊 [{timestamp}] PROGRESS: {preview}"
            return $"Progress recorded at {timestamp}: {description}"
        }
