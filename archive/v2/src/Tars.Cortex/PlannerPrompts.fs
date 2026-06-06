namespace Tars.Cortex

open System

module PlannerPrompts =

    let generatePlanPrompt (goal: string) (options: Map<string, string> option) : string =
        let currentDate = DateTime.Now.ToString("yyyy-MM-dd")
        let originalPath = "prompts/planner.md"

        let promptPath =
            match options with
            | Some overlays -> Map.tryFind originalPath overlays |> Option.defaultValue originalPath
            | None -> originalPath

        let template =
            if System.IO.File.Exists promptPath then
                System.IO.File.ReadAllText promptPath
            else
                // Legacy hardcoded fallback
                """You are TARS-Architect, an expert in designing autonomous agent workflows using the TARS .trsx DSL.
The current date is {currentDate}.
Your goal is to convert a user's intent into a valid, executable .wot.trsx workflow file.

### DSL Guide
... (truncated for brevity in actual replacement)
"""

        template
            .Replace("{currentDate}", currentDate)
            .Replace("{goal}", goal)
            .Replace("{{", "{") // Handle double braces if present in file or legacy
            .Replace("}}", "}")
