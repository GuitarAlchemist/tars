/// Demo Generator - Creates presentations from pipeline outputs
/// Supports multiple output formats for showcasing project progress
module Tars.Core.DemoGenerator

open System
open System.IO
open System.Text
open Tars.Core.Project
open Tars.Core.Artifact

// ============================================================================
// Demo Types
// ============================================================================

/// Demo output format
type DemoFormat =
    | MarkdownReport
    | HtmlPresentation
    | JsonSummary
    | PlainText

/// Demo section for a stage
type DemoSection =
    { Stage: string
      Title: string
      Summary: string
      Artifacts: string list
      Highlights: string list }

/// Complete demo output
type Demo =
    { ProjectId: string
      ProjectName: string
      Title: string
      Subtitle: string option
      Sections: DemoSection list
      GeneratedAt: DateTime
      Format: DemoFormat }

// ============================================================================
// Demo Builder
// ============================================================================

type DemoBuilder() =
    let mutable sections: DemoSection list = []
    let mutable projectName = ""
    let mutable projectId = ""
    let mutable title = "Project Demo"
    let mutable subtitle: string option = None

    member _.SetProject(id: string, name: string) =
        projectId <- id
        projectName <- name

    member _.SetTitle(t: string) = title <- t
    member _.SetSubtitle(s: string) = subtitle <- Some s

    member _.AddSection(stage: string, sectionTitle: string, summary: string) =
        let section =
            { Stage = stage
              Title = sectionTitle
              Summary = summary
              Artifacts = []
              Highlights = [] }

        sections <- sections @ [ section ]

    member _.AddArtifact(stage: string, artifactName: string) =
        sections <-
            sections
            |> List.map (fun s ->
                if s.Stage = stage then
                    { s with
                        Artifacts = s.Artifacts @ [ artifactName ] }
                else
                    s)

    member _.AddHighlight(stage: string, highlight: string) =
        sections <-
            sections
            |> List.map (fun s ->
                if s.Stage = stage then
                    { s with
                        Highlights = s.Highlights @ [ highlight ] }
                else
                    s)

    member _.Build(format: DemoFormat) : Demo =
        { ProjectId = projectId
          ProjectName = projectName
          Title = title
          Subtitle = subtitle
          Sections = sections
          GeneratedAt = DateTime.UtcNow
          Format = format }

// ============================================================================
// Demo Renderers
// ============================================================================

/// Render demo as Markdown
let renderMarkdown (demo: Demo) : string =
    let sb = StringBuilder()

    sb.AppendLine($"# {demo.Title}") |> ignore
    demo.Subtitle |> Option.iter (fun s -> sb.AppendLine($"*{s}*") |> ignore)
    sb.AppendLine() |> ignore
    sb.AppendLine($"**Project:** {demo.ProjectName} (`{demo.ProjectId}`)") |> ignore
    let genTime = demo.GeneratedAt.ToString("yyyy-MM-dd HH:mm:ss")
    sb.AppendLine($"**Generated:** {genTime} UTC") |> ignore
    sb.AppendLine() |> ignore
    sb.AppendLine("---") |> ignore
    sb.AppendLine() |> ignore

    for section in demo.Sections do
        sb.AppendLine($"## {section.Title}") |> ignore
        sb.AppendLine() |> ignore
        sb.AppendLine(section.Summary) |> ignore
        sb.AppendLine() |> ignore

        if not section.Highlights.IsEmpty then
            sb.AppendLine("### Key Highlights") |> ignore

            for h in section.Highlights do
                sb.AppendLine($"- {h}") |> ignore

            sb.AppendLine() |> ignore

        if not section.Artifacts.IsEmpty then
            sb.AppendLine("### Artifacts") |> ignore

            for a in section.Artifacts do
                sb.AppendLine($"- `{a}`") |> ignore

            sb.AppendLine() |> ignore

    sb.ToString()

/// Render demo as HTML
let renderHtml (demo: Demo) : string =
    let sb = StringBuilder()

    sb.AppendLine("<!DOCTYPE html>") |> ignore
    sb.AppendLine("<html><head>") |> ignore
    sb.AppendLine($"<title>{demo.Title}</title>") |> ignore
    sb.AppendLine("<style>") |> ignore

    sb.AppendLine(
        "body { font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 2rem; }"
    )
    |> ignore

    sb.AppendLine(
        "h1 { color: #1a1a2e; } h2 { color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 0.5rem; }"
    )
    |> ignore

    sb.AppendLine(
        ".meta { color: #666; font-size: 0.9rem; } .highlight { background: #e8f5e9; padding: 0.5rem; border-radius: 4px; margin: 0.25rem 0; }"
    )
    |> ignore

    sb.AppendLine(
        ".artifact { font-family: monospace; background: #f5f5f5; padding: 0.25rem 0.5rem; border-radius: 4px; }"
    )
    |> ignore

    sb.AppendLine("</style></head><body>") |> ignore

    sb.AppendLine($"<h1>{demo.Title}</h1>") |> ignore

    demo.Subtitle
    |> Option.iter (fun s -> sb.AppendLine($"<p><em>{s}</em></p>") |> ignore)

    let genTimeHtml = demo.GeneratedAt.ToString("yyyy-MM-dd HH:mm")

    sb.AppendLine($"<p class=\"meta\">Project: {demo.ProjectName} | Generated: {genTimeHtml}</p>")
    |> ignore

    sb.AppendLine("<hr/>") |> ignore

    for section in demo.Sections do
        sb.AppendLine($"<h2>{section.Title}</h2>") |> ignore
        sb.AppendLine($"<p>{section.Summary}</p>") |> ignore

        if not section.Highlights.IsEmpty then
            sb.AppendLine("<h3>Key Highlights</h3>") |> ignore

            for h in section.Highlights do
                sb.AppendLine($"<div class=\"highlight\">✓ {h}</div>") |> ignore

        if not section.Artifacts.IsEmpty then
            sb.AppendLine("<h3>Artifacts</h3><ul>") |> ignore

            for a in section.Artifacts do
                sb.AppendLine($"<li><span class=\"artifact\">{a}</span></li>") |> ignore

            sb.AppendLine("</ul>") |> ignore

    sb.AppendLine("</body></html>") |> ignore
    sb.ToString()

/// Render demo as JSON
let renderJson (demo: Demo) : string =
    let formatSection (s: DemoSection) =
        let artifacts = s.Artifacts |> List.map (sprintf "\"%s\"") |> String.concat ","
        let highlights = s.Highlights |> List.map (sprintf "\"%s\"") |> String.concat ","

        $"{{\"stage\":\"%s{s.Stage}\",\"title\":\"%s{s.Title}\",\"summary\":\"%s{s.Summary}\",\"artifacts\":[%s{artifacts}],\"highlights\":[%s{highlights}]}}"

    let sectionsJson = demo.Sections |> List.map formatSection |> String.concat ","
    let timestamp = demo.GeneratedAt.ToString("o")

    $"{{\"projectId\":\"%s{demo.ProjectId}\",\"projectName\":\"%s{demo.ProjectName}\",\"title\":\"%s{demo.Title}\",\"generatedAt\":\"%s{timestamp}\",\"sections\":[%s{sectionsJson}]}}"

/// Render demo as plain text
let renderPlainText (demo: Demo) : string =
    let sb = StringBuilder()

    sb.AppendLine(String.replicate 60 "=") |> ignore
    sb.AppendLine(demo.Title.ToUpper()) |> ignore
    demo.Subtitle |> Option.iter (fun s -> sb.AppendLine(s) |> ignore)
    sb.AppendLine(String.replicate 60 "=") |> ignore
    sb.AppendLine() |> ignore
    sb.AppendLine($"Project: {demo.ProjectName}") |> ignore
    let genTimePlain = demo.GeneratedAt.ToString("yyyy-MM-dd HH:mm:ss")
    sb.AppendLine($"Generated: {genTimePlain}") |> ignore
    sb.AppendLine() |> ignore

    for section in demo.Sections do
        sb.AppendLine(String.replicate 40 "-") |> ignore
        sb.AppendLine(section.Title.ToUpper()) |> ignore
        sb.AppendLine(String.replicate 40 "-") |> ignore
        sb.AppendLine(section.Summary) |> ignore
        sb.AppendLine() |> ignore

        if not section.Highlights.IsEmpty then
            sb.AppendLine("HIGHLIGHTS:") |> ignore

            for h in section.Highlights do
                sb.AppendLine($"  * {h}") |> ignore

            sb.AppendLine() |> ignore

    sb.ToString()

/// Render demo in specified format
let render (demo: Demo) : string =
    match demo.Format with
    | MarkdownReport -> renderMarkdown demo
    | HtmlPresentation -> renderHtml demo
    | JsonSummary -> renderJson demo
    | PlainText -> renderPlainText demo

// ============================================================================
// Demo Generation from Project
// ============================================================================

/// Generate demo from project state
let generateFromProject (project: Project) (stageArtifacts: Map<string, StageArtifacts>) (format: DemoFormat) : Demo =
    let builder = DemoBuilder()
    builder.SetProject(project.Id, project.Name)
    builder.SetTitle($"{project.Name} - Pipeline Demo")
    builder.SetSubtitle($"Template: {project.Template}")

    for stage in templateStages project.Template do
        let stageNameStr = stageName stage
        builder.AddSection(stageNameStr, $"{stageNameStr} Phase", $"Completed the {stageNameStr} phase.")

        match Map.tryFind stageNameStr stageArtifacts with
        | Some artifacts ->
            for artifact in artifacts.Artifacts do
                builder.AddArtifact(stageNameStr, artifact.Name)

            artifacts.Summary
            |> Option.iter (fun s -> builder.AddHighlight(stageNameStr, s))
        | None -> ()

    builder.Build(format)

/// Save demo to file
let saveDemo (demo: Demo) (outputPath: string) : Result<string, string> =
    try
        let content = render demo
        let dir = Path.GetDirectoryName(outputPath)

        if not (String.IsNullOrEmpty dir) && not (Directory.Exists dir) then
            Directory.CreateDirectory dir |> ignore

        File.WriteAllText(outputPath, content)
        Result.Ok outputPath
    with ex ->
        Result.Error ex.Message
