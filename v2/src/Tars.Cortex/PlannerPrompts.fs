namespace Tars.Cortex

open System
open System.IO

module PlannerPrompts =

    [<Literal>]
    let TemplatePath = "prompts/planner.md"

    /// Candidate locations for the planner template. A configured overlay is used as-is;
    /// otherwise the template is looked up relative to the working directory, then next to
    /// the executable (the CLI copies prompts/planner.md into its output).
    let private candidates (overlays: Map<string, string> option) =
        match overlays |> Option.bind (Map.tryFind TemplatePath) with
        | Some overlay -> [ overlay ]
        | None ->
            [ Path.GetFullPath TemplatePath
              Path.Combine(AppContext.BaseDirectory, TemplatePath) ]

    /// Build the planner prompt for `goal`. Returns an Error naming every path tried when
    /// the template is missing, rather than a placeholder prompt (#251).
    let generatePlanPrompt (goal: string) (overlays: Map<string, string> option) : Result<string, string> =
        let tried = candidates overlays

        match tried |> List.tryFind File.Exists with
        | None ->
            let paths = String.Join(", ", tried)
            Error $"Planner prompt template not found. Looked in: {paths}"
        | Some path ->
            let currentDate = DateTime.Now.ToString("yyyy-MM-dd")

            File
                .ReadAllText(path)
                .Replace("{currentDate}", currentDate)
                .Replace("{goal}", goal)
                .Replace("{{", "{") // Handle double braces if present in the file
                .Replace("}}", "}")
            |> Ok
